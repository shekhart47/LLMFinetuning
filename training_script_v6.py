import os
import torch
import warnings
warnings.filterwarnings(“ignore”)

# Fix for distributed training - MUST be before any other imports

os.environ[“TOKENIZERS_PARALLELISM”] = “false”
os.environ[“CUDA_LAUNCH_BLOCKING”] = “1”  # Better error messages
os.environ[“OMP_NUM_THREADS”] = “1”

# DISABLE MLFLOW LOGGING FOR AZURE ML - COMPREHENSIVE

os.environ[“MLFLOW_TRACKING_URI”] = “”
os.environ[“DISABLE_MLFLOW_INTEGRATION”] = “TRUE”
os.environ[“HF_MLFLOW_LOG_ARTIFACTS”] = “FALSE”
os.environ[“AZUREML_LOG_MODELS”] = “FALSE”
os.environ[“AZUREML_MLFLOW_LOG_ARTIFACTS”] = “FALSE”
os.environ[“MLFLOW_EXPERIMENT_ID”] = “”
os.environ[“MLFLOW_RUN_ID”] = “”

# Only set these if not already set by torchrun/accelerate

if “RANK” not in os.environ:
os.environ[“MASTER_ADDR”] = “localhost”
os.environ[“MASTER_PORT”] = “29500”

# Set multiprocessing start method

import multiprocessing
try:
multiprocessing.set_start_method(‘spawn’, force=True)
except RuntimeError:
pass
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
TrainingArguments,
Trainer,
BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_from_disk

# import wandb  # REMOVED - Don’t import wandb at all

from typing import Optional
import json

class TrainingConfig:
def **init**(self):
# Model configuration
self.MODEL_NAME = “dmis-lab/llama-3-meerkat-8b-v1.0”
self.OUTPUT_DIR = “./results”
self.DATASET_DIR = “./datasets”

```
    # LoRA configuration
    self.LORA_RANK = 8
    self.LORA_ALPHA = 16
    self.LORA_DROPOUT = 0.1
    self.LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # Quantization configuration
    self.USE_4BIT = True
    self.BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
    self.BNB_4BIT_QUANT_TYPE = "nf4"
    self.BNB_4BIT_USE_DOUBLE_QUANT = True
    
    # Training hyperparameters
    self.NUM_EPOCHS = 3
    self.BATCH_SIZE = 4
    self.GRADIENT_ACCUMULATION_STEPS = 4
    self.LEARNING_RATE = 2e-4
    self.WARMUP_RATIO = 0.1
    self.WEIGHT_DECAY = 0.01
    self.MAX_SEQ_LENGTH = 2048
    
    # Optimization
    self.OPTIM = "paged_adamw_8bit"
    self.LR_SCHEDULER_TYPE = "cosine"
    self.MAX_GRAD_NORM = 1.0
    
    # Logging and evaluation
    self.LOGGING_STEPS = 10
    self.EVAL_STEPS = 500
    self.SAVE_STEPS = 500
    self.SAVE_TOTAL_LIMIT = 3
    self.EVAL_STRATEGY = "steps"
    self.LOAD_BEST_MODEL_AT_END = True
    self.METRIC_FOR_BEST_MODEL = "eval_loss"
    self.GREATER_IS_BETTER = False
    
    # Hardware optimization
    self.DATALOADER_NUM_WORKERS = 4
    self.GROUP_BY_LENGTH = True
    self.DDPO_ENABLED = True  # Distributed training
    
    # Wandb configuration - DISABLED FOR AZURE ML
    self.USE_WANDB = False  # Turn off Wandb to avoid conflicts
    self.WANDB_PROJECT = "meerkat-medical-qa"
    self.WANDB_RUN_NAME = None
    
    # Seed for reproducibility
    self.SEED = 42
```

class EvalConfig:
def **init**(self):
# Evaluation configuration
self.EVAL_BATCH_SIZE = 8
self.EVAL_STEPS = 500
self.EVAL_ACCUMULATION_STEPS = 1
self.PREDICTION_LOSS_ONLY = False

def setup_model_and_tokenizer(config: TrainingConfig):
“””
Setup model and tokenizer with QLoRA configuration.
“””
print(“Setting up model and tokenizer…”)

```
# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config.USE_4BIT,
    bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model with additional training-specific settings
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
    trust_remote_code=True,
    use_cache=False,  # Required for gradient checkpointing
    attn_implementation="eager",  # Use eager attention for better compatibility
)

# Remove the early gradient checkpointing - do it after LoRA
# model.gradient_checkpointing_enable()  # MOVED TO AFTER LORA

# Ensure model is in training mode
model.train()

# LoRA configuration with better target modules
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.LORA_RANK,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    bias="none",
    use_rslora=False,  # Disable RSLoRA for compatibility
    modules_to_save=None,  # Avoid saving extra modules
)

# Apply LoRA with error handling
try:
    model = get_peft_model(model, lora_config)
except Exception as e:
    print(f"Error applying PEFT: {e}")
    print("Trying with simplified LoRA config...")
    
    # Simplified LoRA config
    lora_config_simple = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # Only basic modules
        bias="none",
    )
    model = get_peft_model(model, lora_config_simple)

# CRITICAL: Prepare model for training after LoRA application
model.train()

# Enable input gradients for the embeddings
if hasattr(model, 'get_input_embeddings'):
    model.get_input_embeddings().weight.requires_grad_(True)

# Ensure all trainable parameters require gradients
for name, param in model.named_parameters():
    if param.requires_grad:
        param.requires_grad_(True)

# Additional fix for gradient checkpointing with LoRA
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Enable gradients for LoRA parameters specifically
for name, module in model.named_modules():
    if 'lora_' in name.lower():
        for param in module.parameters():
            param.requires_grad_(True)

# Print model info
if hasattr(model, 'print_trainable_parameters'):
    model.print_trainable_parameters()

return model, tokenizer
```

def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int):
“””
Tokenize the dataset for training with proper padding and truncation.
“””
def tokenize_function(examples):
# Use the formatted_chat column for training
tokenized = tokenizer(
examples[“formatted_chat”],
truncation=True,
padding=“max_length”,  # Pad all sequences to max_length
max_length=max_length,
return_tensors=None,
add_special_tokens=True,
)

```
    # Set labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
    num_proc=1,  # Single process to avoid issues
)

return tokenized_dataset
```

class DataCollatorForLanguageModeling:
“””
Custom data collator to ensure proper padding and attention masks.
“””

```
def __init__(self, tokenizer, max_length=2048):
    self.tokenizer = tokenizer
    self.max_length = max_length

def __call__(self, features):
    # Extract input_ids and labels
    batch = {}
    
    # Pad sequences to the same length
    input_ids = [f["input_ids"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Pad to max length in batch or global max_length
    max_len = min(max(len(seq) for seq in input_ids), self.max_length)
    
    # Pad input_ids
    padded_input_ids = []
    attention_masks = []
    padded_labels = []
    
    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        # Truncate if too long
        if len(inp) > max_len:
            inp = inp[:max_len]
            lab = lab[:max_len]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(inp)
        
        # Pad sequences
        padding_length = max_len - len(inp)
        inp = inp + [self.tokenizer.pad_token_id] * padding_length
        lab = lab + [-100] * padding_length  # -100 is ignored in loss
        attention_mask = attention_mask + [0] * padding_length
        
        padded_input_ids.append(inp)
        padded_labels.append(lab)
        attention_masks.append(attention_mask)
    
    batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
    batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    
    return batch
```

def setup_training_arguments(config: TrainingConfig, eval_config: EvalConfig):
“””
Setup training arguments for Hugging Face Trainer.
“””
return TrainingArguments(
output_dir=config.OUTPUT_DIR,
overwrite_output_dir=True,
num_train_epochs=config.NUM_EPOCHS,
per_device_train_batch_size=config.BATCH_SIZE,
per_device_eval_batch_size=eval_config.EVAL_BATCH_SIZE,
gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
learning_rate=config.LEARNING_RATE,
warmup_ratio=config.WARMUP_RATIO,
weight_decay=config.WEIGHT_DECAY,
optim=config.OPTIM,
lr_scheduler_type=config.LR_SCHEDULER_TYPE,
max_grad_norm=config.MAX_GRAD_NORM,

```
    # Logging and evaluation - FIXED PARAMETER NAME
    logging_steps=config.LOGGING_STEPS,
    eval_steps=config.EVAL_STEPS,
    eval_strategy=config.EVAL_STRATEGY,  # Changed from evaluation_strategy
    save_steps=config.SAVE_STEPS,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
    metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
    greater_is_better=config.GREATER_IS_BETTER,
    
    # Hardware optimization
    dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
    group_by_length=config.GROUP_BY_LENGTH,
    ddp_find_unused_parameters=False,
    
    # Mixed precision
    bf16=True,
    tf32=True,
    
    # DISABLE ALL EXTERNAL LOGGING FOR AZURE ML
    report_to=None,  # Disable all external logging
    run_name=None,
    
    # Reproducibility
    seed=config.SEED,
    data_seed=config.SEED,
    
    # Memory optimization
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    
    # Azure ML specific settings
    push_to_hub=False,  # Don't push to HF Hub
    hub_model_id=None,
    hub_strategy="every_save",
    hub_token=None,
)
```

# Removed MedicalQATrainer class - using standard Trainer instead

# This avoids compatibility issues with different Transformers versions

def disable_mlflow_completely():
“”“Completely disable MLflow integration”””
try:
import mlflow
mlflow.end_run()  # End any existing run
# Monkey patch mlflow to do nothing
mlflow.log_params = lambda x: None
mlflow.log_metrics = lambda x: None
mlflow.log_artifacts = lambda x: None
mlflow.start_run = lambda: None
mlflow.end_run = lambda: None
print(“MLflow disabled successfully”)
except ImportError:
print(“MLflow not installed - no action needed”)
except Exception as e:
print(f”Error disabling MLflow: {e}”)

def main():
“””
Main training function.
“””
# FIRST: Disable MLflow completely
disable_mlflow_completely()

```
# Initialize configurations
config = TrainingConfig()
eval_config = EvalConfig()

# DON'T initialize wandb at all - completely removed
# No wandb logic needed since USE_WANDB = False

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer(config)

# Load datasets
print("Loading datasets...")
train_dataset = load_from_disk(os.path.join(config.DATASET_DIR, "train_dataset"))
eval_dataset = load_from_disk(os.path.join(config.DATASET_DIR, "eval_dataset"))

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Eval dataset: {len(eval_dataset)} samples")

# Tokenize datasets
train_dataset_tokenized = tokenize_dataset(train_dataset, tokenizer, config.MAX_SEQ_LENGTH)
eval_dataset_tokenized = tokenize_dataset(eval_dataset, tokenizer, config.MAX_SEQ_LENGTH)

# Setup training arguments
training_args = setup_training_arguments(config, eval_config)

# Create custom data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, max_length=config.MAX_SEQ_LENGTH)

# Create trainer - Using standard Trainer to avoid compatibility issues
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=eval_dataset_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model()
tokenizer.save_pretrained(config.OUTPUT_DIR)

# Save training config
with open(os.path.join(config.OUTPUT_DIR, "training_config.json"), "w") as f:
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    json.dump(config_dict, f, indent=2, default=str)

print(f"Training completed! Model saved to {config.OUTPUT_DIR}")

# No wandb.finish() needed since we don't use wandb
```

if **name** == “**main**”:
main()
