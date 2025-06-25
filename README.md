# Medical QA Fine-tuning Pipeline for Llama-3-Meerkat-8B

This repository contains a complete pipeline for fine-tuning the Llama-3-Meerkat-8B model on medical QA datasets with progressive training capabilities and comprehensive evaluation.

## 🚀 Overview

The pipeline consists of three main components:

1. **Dataset Preparation**: Processes medical specialty JSON files into HuggingFace datasets with holdout/train/eval splits
1. **Fine-tuning**: QLoRA-based fine-tuning with configurable hyperparameters
1. **Evaluation**: Comprehensive evaluation with automatic metrics and specialty-wise analysis

## 📋 Requirements

### Dependencies

```bash
pip install torch transformers datasets peft bitsandbytes
pip install nltk rouge-score bert-score matplotlib seaborn pandas numpy
pip install accelerate wandb tqdm scikit-learn
```

### Required Models for Evaluation

The evaluation script requires additional models that need to be downloaded automatically:

1. **BERTScore**: Uses `microsoft/deberta-xlarge-mnli` (downloaded automatically)
1. **NLTK Data**: Downloads `punkt` and `wordnet` automatically
1. **ROUGE**: Built into `rouge-score` package

**Note**: The first run of evaluation will download these models automatically. Ensure you have sufficient disk space (~2GB for BERTScore model).

## 🗂️ Directory Structure

```
project/
├── dataset_preparation.py      # Dataset processing script
├── fine_tuning.py             # Training script with QLoRA
├── evaluation.py              # Evaluation script with metrics
├── README.md                  # This file
├── datasets/                  # Generated datasets
│   ├── holdout_dataset/       # 10 samples per specialty (constant)
│   ├── train_dataset/         # Training data (progressive)
│   ├── eval_dataset/          # Evaluation data (from train split)
│   └── eval_pool_dataset/     # Remaining data for future iterations
├── results/                   # Fine-tuned model outputs
└── evaluation_results/        # Evaluation outputs and analysis
```

## 🔧 Setup and Configuration

### 1. Update Data Paths

In `dataset_preparation.py`, update the paths:

```python
config = DatasetConfig()
config.DATA_DIR = "/path/to/your/json/files"  # Directory with medical JSON files
config.OUTPUT_DIR = "/path/to/output/datasets"  # Where to save processed datasets
```

### 2. Configuration Parameters

The pipeline uses `SimpleNamespace` classes for configuration:

**Dataset Configuration** (`DatasetConfig`):

- `HOLDOUT_SAMPLES_PER_SPECIALTY`: 10 (constant test set)
- `INITIAL_TRAIN_PERCENTAGE`: 0.5 (start with 50% of data)
- `EVAL_SPLIT_SIZE`: 0.2 (20% of training data for evaluation)

**Training Configuration** (`TrainingConfig`):

- `LORA_RANK`: 8
- `LORA_ALPHA`: 16
- `LORA_TARGET_MODULES`: [“q_proj”, “v_proj”]
- `NUM_EPOCHS`: 3
- `BATCH_SIZE`: 4
- `LEARNING_RATE`: 2e-4

**Evaluation Configuration** (`EvaluationConfig`):

- `MAX_NEW_TOKENS`: 512
- `TEMPERATURE`: 0.1
- `BATCH_SIZE`: 4

## 🏃 Running the Pipeline

### Step 1: Dataset Preparation

```bash
python dataset_preparation.py
```

This creates:

- Holdout dataset (420 samples: 10 per specialty)
- Initial training dataset (50% of remaining data)
- Evaluation dataset (20% of training data)
- Pool dataset (remaining data for future iterations)

### Step 2: Fine-tuning

```bash
python fine_tuning.py
```

**GPU Requirements**:

- Recommended: 4 A100 or 2 H100 GPUs
- Minimum: 1 GPU with 24GB+ VRAM
- Uses BF16 precision and QLoRA for memory efficiency

**Wandb Integration** (Optional):

```bash
export WANDB_API_KEY=your_key
# Set USE_WANDB=True in TrainingConfig
```

### Step 3: Evaluation

```bash
python evaluation.py
```

Update model paths in `evaluation.py`:

```python
config = EvaluationConfig()
config.FINETUNED_MODEL_PATH = "./results"  # Path to your fine-tuned model
```

## 📊 Evaluation Metrics

The evaluation script computes all metrics from your specification:

### Automatic Metrics (Baseline Filtering)

- **BLEU (1-4 gram)**: Surface-level n-gram overlap with reference answers
- **ROUGE (1, 2, L)**: Recall-oriented evaluation for text summarization
- **METEOR**: Semantic similarity using contextual embeddings
- **BERTScore**: Semantic similarity using BERT embeddings (preferred for QA)
- **Exact Match**: For factual spans (medical entities, dosages)
- **Token-level F1**: Overlap in medical terms (symptoms, treatments, diagnoses)

### Specialty-wise Analysis

- Performance breakdown by medical specialty
- Identification of low-performing specialties
- Recommendations for next training iteration

## 🔄 Progressive Training Workflow

### Initial Training

1. Run dataset preparation with 50% of training data
1. Fine-tune model
1. Evaluate and identify low-performance specialties

### Subsequent Iterations

1. Update specialty sample percentages based on evaluation:

```python
# In dataset_preparation.py
specialty_adjustments = {
    'cardiology': 0.8,      # Increase to 80% for low-performing specialty
    'neurology': 0.6,       # Increase to 60%
    'dermatology': 0.5      # Keep at 50% (good performance)
}

# Recreate training dataset
holdout_ds, train_ds, eval_ds, pool_ds = update_training_data(config, specialty_adjustments)
```

1. Re-run fine-tuning with new training data
1. Evaluate improvement

### Automated Recommendations

The evaluation script generates `training_recommendations.json`:

```json
{
  "low_performance_specialties": ["cardiology", "neurology"],
  "recommended_sample_increases": {
    "cardiology": 0.8,
    "neurology": 0.8
  },
  "overall_improvement": {
    "ROUGE-L": 0.045,
    "bertscore_f1": 0.032
  }
}
```

## 📈 Output Files

### Dataset Preparation

- `holdout_dataset/`: Constant test set
- `train_dataset/`: Current training data
- `eval_dataset/`: Evaluation split
- `eval_pool_dataset/`: Remaining data
- `dataset_stats.json`: Dataset statistics

### Training

- `results/`: Fine-tuned model and tokenizer
- `training_config.json`: Training configuration used
- Wandb logs (if enabled)

### Evaluation

- `base_model_results.csv`: Detailed base model predictions
- `finetuned_model_results.csv`: Detailed fine-tuned model predictions
- `model_comparison.csv`: Overall performance comparison
- `specialty_performance.csv`: Performance by medical specialty
- `specialty_analysis.png`: Visualization of specialty performance
- `model_comparison.png`: Model comparison chart
- `training_recommendations.json`: Recommendations for next iteration

## 🛠️ Customization

### Adding New Metrics

To add custom evaluation metrics, modify the `AutomaticMetrics` class in `evaluation.py`:

```python
def compute_custom_metric(self, reference: str, hypothesis: str) -> float:
    # Your custom metric implementation
    return score
```

### Modifying Chat Template

Update the template in `dataset_preparation.py` and `evaluation.py`:

```python
def apply_chat_template(question: str, answer: str, system_message: str) -> str:
    # Modify template format here
    return formatted_template
```

### Adjusting Training Parameters

Modify the `TrainingConfig` class in `fine_tuning.py`:

```python
class TrainingConfig:
    def __init__(self):
        self.LORA_RANK = 16          # Increase for more capacity
        self.LORA_ALPHA = 32         # Adjust learning rate scaling
        self.NUM_EPOCHS = 5          # More epochs
        self.BATCH_SIZE = 8          # Larger batch size if memory allows
```

## 🐛 Troubleshooting

### Memory Issues

- Reduce `BATCH_SIZE` in training config
- Enable gradient checkpointing (already enabled)
- Use `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size

### Model Loading Issues

- Ensure PEFT model was saved correctly
- Check that base model path is correct
- Verify GPU memory is sufficient

### Evaluation Errors

- Ensure NLTK data is downloaded (runs automatically)
- Check that BERTScore model downloads successfully
- Verify input data format matches expected structure

### Dataset Issues

- Ensure JSON files follow the expected format
- Check that all specialty files are present in the directory
- Verify file naming convention: `dataset_with_answers_{specialty}.json`

## 📝 Notes

1. **First Run**: The evaluation script will download required models (~2GB), so ensure good internet connectivity.
1. **Reproducibility**: All random operations use seeds specified in config classes.
1. **GPU Memory**: The pipeline is optimized for A100/H100 GPUs. For smaller GPUs, reduce batch sizes and consider using gradient checkpointing.
1. **Progressive Training**: The holdout and eval datasets remain constant across iterations, only the training dataset changes based on specialty performance.
1. **Template Consistency**: Ensure the same chat template is used in dataset preparation, training, and evaluation.

## 🤝 Support

For issues related to:

- **Hugging Face Integration**: Check transformers and datasets documentation
- **PEFT/LoRA**: Refer to PEFT library documentation
- **Evaluation Metrics**: Check individual metric library documentation (NLTK, ROUGE, BERTScore)

## 📚 References

- [Llama-3-Meerkat-8B Model](https://huggingface.co/dmis-lab/llama-3-meerkat-8b-v1.0)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [Medical QA Evaluation Best Practices](https://arxiv.org/abs/2010.11610)
