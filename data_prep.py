import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


class DatasetConfig:
    def __init__(self):
        # Data paths
        self.DATA_DIR = ""  # Path to directory containing JSON files
        self.OUTPUT_DIR = ""  # Path to save processed datasets
        
        # Dataset split parameters
        self.HOLDOUT_SAMPLES_PER_SPECIALTY = 10
        self.INITIAL_TRAIN_PERCENTAGE = 0.5  # Use 50% of available training data initially
        self.EVAL_SPLIT_SIZE = 0.2  # 20% of training data for evaluation
        self.RANDOM_SEED = 42
        
        # Specialty-specific sampling (for progressive training)
        self.SPECIALTY_SAMPLE_PERCENTAGES = {}  # Can be updated for specific specialties
        
        # Template configuration
        self.SYSTEM_MESSAGE = (
            "You are a helpful medical AI assistant specializing in providing accurate, "
            "evidence-based answers to medical questions. Provide clear, comprehensive, "
            "and scientifically-grounded responses while maintaining appropriate medical "
            "disclaimers when necessary."
        )


def load_medical_datasets(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Load all JSON files from the directory and extract specialty-wise QA pairs.
    
    Args:
        data_dir: Path to directory containing JSON files
        
    Returns:
        Dictionary with specialty as key and QA pairs as value
    """
    specialty_data = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'dataset_with_answers_' in filename:
            # Extract specialty name from filename
            specialty = filename.replace('dataset_with_answers_', '').replace('.json', '')
            
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to standard format if needed
            if isinstance(data, dict):
                specialty_data[specialty] = data
            else:
                print(f"Warning: Unexpected data format in {filename}")
                
    print(f"Loaded {len(specialty_data)} specialties")
    for specialty, qa_pairs in specialty_data.items():
        print(f"  {specialty}: {len(qa_pairs)} QA pairs")
        
    return specialty_data


def sample_diverse_queries(qa_pairs: Dict[str, str], n_samples: int, seed: int = 42) -> Dict[str, str]:
    """
    Sample diverse queries from a specialty's QA pairs.
    Uses simple length-based diversity for now.
    
    Args:
        qa_pairs: Dictionary of question-answer pairs
        n_samples: Number of samples to select
        seed: Random seed
        
    Returns:
        Sampled QA pairs
    """
    random.seed(seed)
    
    if len(qa_pairs) <= n_samples:
        return qa_pairs
    
    # Sort questions by length to ensure diversity
    questions = list(qa_pairs.keys())
    questions.sort(key=len)
    
    # Sample from different length bins
    bin_size = len(questions) // n_samples
    sampled_questions = []
    
    for i in range(n_samples):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(questions))
        if start_idx < len(questions):
            bin_questions = questions[start_idx:end_idx]
            sampled_questions.append(random.choice(bin_questions))
    
    # If we need more samples, randomly sample from remaining
    if len(sampled_questions) < n_samples:
        remaining = set(questions) - set(sampled_questions)
        additional = random.sample(list(remaining), n_samples - len(sampled_questions))
        sampled_questions.extend(additional)
    
    return {q: qa_pairs[q] for q in sampled_questions[:n_samples]}


def apply_chat_template(question: str, answer: str, system_message: str) -> str:
    """
    Apply chat template to question-answer pair.
    
    Args:
        question: Medical question
        answer: Medical answer
        system_message: System message for the template
        
    Returns:
        Formatted chat template string
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    # Convert to chat format (simplified version)
    template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
    template += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    template += f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
    
    return template


def create_datasets(config: DatasetConfig):
    """
    Create holdout, train, and eval datasets with progressive training capability.
    """
    # Load all specialty data
    specialty_data = load_medical_datasets(config.DATA_DIR)
    
    # Create holdout dataset (10 samples per specialty)
    holdout_data = []
    remaining_data = {}
    
    print("Creating holdout dataset...")
    for specialty, qa_pairs in specialty_data.items():
        # Sample diverse queries for holdout
        holdout_samples = sample_diverse_queries(
            qa_pairs, 
            config.HOLDOUT_SAMPLES_PER_SPECIALTY, 
            config.RANDOM_SEED
        )
        
        # Add to holdout dataset
        for question, answer in holdout_samples.items():
            holdout_data.append({
                'specialty': specialty,
                'question': question,
                'response': answer,
                'formatted_chat': apply_chat_template(question, answer, config.SYSTEM_MESSAGE)
            })
        
        # Store remaining data
        remaining_qa = {q: a for q, a in qa_pairs.items() if q not in holdout_samples}
        remaining_data[specialty] = remaining_qa
        
        print(f"  {specialty}: {len(holdout_samples)} holdout, {len(remaining_qa)} remaining")
    
    # Create train dataset with initial percentage
    print(f"\nCreating train dataset with {config.INITIAL_TRAIN_PERCENTAGE*100}% of available data...")
    train_data = []
    eval_pool_data = []
    
    for specialty, qa_pairs in remaining_data.items():
        # Check if specialty has custom sample percentage
        specialty_percentage = config.SPECIALTY_SAMPLE_PERCENTAGES.get(
            specialty, 
            config.INITIAL_TRAIN_PERCENTAGE
        )
        
        n_samples = int(len(qa_pairs) * specialty_percentage)
        if n_samples == 0 and len(qa_pairs) > 0:
            n_samples = 1  # Ensure at least one sample per specialty
            
        # Sample for training
        questions = list(qa_pairs.keys())
        random.seed(config.RANDOM_SEED)
        train_questions = random.sample(questions, min(n_samples, len(questions)))
        
        # Add to train dataset
        for question in train_questions:
            answer = qa_pairs[question]
            train_data.append({
                'specialty': specialty,
                'question': question,
                'response': answer,
                'formatted_chat': apply_chat_template(question, answer, config.SYSTEM_MESSAGE)
            })
        
        # Store remaining for future use
        remaining_questions = [q for q in questions if q not in train_questions]
        for question in remaining_questions:
            answer = qa_pairs[question]
            eval_pool_data.append({
                'specialty': specialty,
                'question': question,
                'response': answer,
                'formatted_chat': apply_chat_template(question, answer, config.SYSTEM_MESSAGE)
            })
        
        print(f"  {specialty}: {len(train_questions)} train samples, {len(remaining_questions)} in pool")
    
    # Split train data into train/eval
    train_df = pd.DataFrame(train_data)
    eval_pool_df = pd.DataFrame(eval_pool_data)
    
    # Stratified split by specialty
    train_final, eval_final = train_test_split(
        train_df,
        test_size=config.EVAL_SPLIT_SIZE,
        stratify=train_df['specialty'],
        random_state=config.RANDOM_SEED
    )
    
    # Convert to HuggingFace datasets
    holdout_dataset = Dataset.from_pandas(pd.DataFrame(holdout_data))
    train_dataset = Dataset.from_pandas(train_final.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_final.reset_index(drop=True))
    eval_pool_dataset = Dataset.from_pandas(eval_pool_df.reset_index(drop=True))
    
    # Save datasets
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    holdout_dataset.save_to_disk(os.path.join(config.OUTPUT_DIR, "holdout_dataset"))
    train_dataset.save_to_disk(os.path.join(config.OUTPUT_DIR, "train_dataset"))
    eval_dataset.save_to_disk(os.path.join(config.OUTPUT_DIR, "eval_dataset"))
    eval_pool_dataset.save_to_disk(os.path.join(config.OUTPUT_DIR, "eval_pool_dataset"))
    
    # Save dataset statistics
    stats = {
        'total_specialties': len(specialty_data),
        'holdout_samples': len(holdout_data),
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'eval_pool_samples': len(eval_pool_dataset),
        'samples_per_specialty': {
            specialty: len([d for d in train_data if d['specialty'] == specialty])
            for specialty in specialty_data.keys()
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "dataset_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset creation completed!")
    print(f"Holdout dataset: {len(holdout_dataset)} samples")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset: {len(eval_dataset)} samples")
    print(f"Eval pool dataset: {len(eval_pool_dataset)} samples")
    print(f"Datasets saved to: {config.OUTPUT_DIR}")
    
    return holdout_dataset, train_dataset, eval_dataset, eval_pool_dataset


def update_training_data(config: DatasetConfig, specialty_adjustments: Dict[str, float]):
    """
    Update training dataset by adjusting sample percentages for specific specialties.
    
    Args:
        config: Dataset configuration
        specialty_adjustments: Dict mapping specialty to new sample percentage
    """
    print("Updating training dataset with specialty adjustments...")
    
    # Update config with new percentages
    config.SPECIALTY_SAMPLE_PERCENTAGES.update(specialty_adjustments)
    
    # Recreate datasets with new configuration
    return create_datasets(config)


if __name__ == "__main__":
    # Initialize configuration
    config = DatasetConfig()
    
    # TODO: Set these paths
    config.DATA_DIR = "/path/to/your/json/files"  # Update this path
    config.OUTPUT_DIR = "/path/to/output/datasets"  # Update this path
    
    # Create initial datasets
    holdout_ds, train_ds, eval_ds, eval_pool_ds = create_datasets(config)
    
    print("\nDataset preparation completed successfully!")
    print("\nTo update training data for specific specialties, use:")
    print("specialty_adjustments = {'cardiology': 0.8, 'neurology': 0.6}")
    print("update_training_data(config, specialty_adjustments)")
