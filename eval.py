import os
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from peft import PeftModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class EvaluationConfig:
    def __init__(self):
        # Model paths
        self.BASE_MODEL_PATH = "dmis-lab/llama-3-meerkat-8b-v1.0"
        self.FINETUNED_MODEL_PATH = "./results"  # Path to fine-tuned model
        self.DATASET_DIR = "./datasets"
        
        # Evaluation settings
        self.MAX_NEW_TOKENS = 512
        self.TEMPERATURE = 0.1
        self.TOP_P = 0.9
        self.DO_SAMPLE = True
        self.BATCH_SIZE = 4
        
        # Output settings
        self.OUTPUT_DIR = "./evaluation_results"
        self.SAVE_PREDICTIONS = True
        self.SAVE_PLOTS = True
        
        # Metrics configuration
        self.ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
        self.BLEU_WEIGHTS = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        self.BLEU_NAMES = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']


class AutomaticMetrics:
    """
    Class to compute automatic evaluation metrics for medical QA.
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BLEU scores (1-4 gram)."""
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        bleu_scores = {}
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        
        for weight, name in zip(weights, names):
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weight, smoothing_function=self.smoothing)
            bleu_scores[name] = score
            
        return bleu_scores
    
    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    
    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        return meteor_score([ref_tokens], hyp_tokens)
    
    def compute_exact_match(self, reference: str, hypothesis: str) -> float:
        """Compute exact match for factual spans."""
        # Extract medical entities (simplified - can be enhanced with NER)
        ref_entities = self._extract_medical_entities(reference)
        hyp_entities = self._extract_medical_entities(hypothesis)
        
        if not ref_entities:
            return 1.0 if not hyp_entities else 0.0
        
        matches = len(ref_entities.intersection(hyp_entities))
        return matches / len(ref_entities)
    
    def compute_token_f1(self, reference: str, hypothesis: str) -> float:
        """Compute token-level F1 score for symptom/treatment/diagnosis terms."""
        ref_tokens = set(nltk.word_tokenize(reference.lower()))
        hyp_tokens = set(nltk.word_tokenize(hypothesis.lower()))
        
        # Filter for medical terms (simplified)
        medical_terms = self._filter_medical_terms(ref_tokens.union(hyp_tokens))
        ref_medical = ref_tokens.intersection(medical_terms)
        hyp_medical = hyp_tokens.intersection(medical_terms)
        
        if not ref_medical and not hyp_medical:
            return 1.0
        if not ref_medical or not hyp_medical:
            return 0.0
        
        common = ref_medical.intersection(hyp_medical)
        precision = len(common) / len(hyp_medical) if hyp_medical else 0
        recall = len(common) / len(ref_medical) if ref_medical else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _extract_medical_entities(self, text: str) -> set:
        """Extract medical entities (simplified pattern matching)."""
        # Patterns for common medical entities
        patterns = [
            r'\b[A-Z][a-z]*itis\b',  # Conditions ending in -itis
            r'\b[A-Z][a-z]*osis\b',  # Conditions ending in -osis
            r'\b[A-Z][a-z]*emia\b',  # Blood conditions
            r'\b\d+\s*mg\b',         # Dosages
            r'\b\d+\s*ml\b',         # Volume measurements
            r'\bmg/kg\b',            # Dosage per weight
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update([m.lower() for m in matches])
        
        return entities
    
    def _filter_medical_terms(self, tokens: set) -> set:
        """Filter tokens for medical relevance (simplified)."""
        medical_indicators = {
            'pain', 'ache', 'fever', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'headache', 'dizziness', 'fatigue', 'weakness', 'inflammation', 'infection',
            'diagnosis', 'treatment', 'therapy', 'medication', 'drug', 'dose', 'dosage',
            'symptoms', 'syndrome', 'disease', 'disorder', 'condition', 'mg', 'ml',
            'twice', 'daily', 'morning', 'evening', 'hours', 'days', 'weeks'
        }
        
        return tokens.intersection(medical_indicators)


def load_model_and_tokenizer(model_path: str, is_peft: bool = False):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_peft else "dmis-lab/llama-3-meerkat-8b-v1.0"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if is_peft:
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            "dmis-lab/llama-3-meerkat-8b-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for inference
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, config: EvaluationConfig) -> str:
    """Generate response for a given question."""
    # Format input (simplified - adjust based on your template)
    system_msg = ("You are a helpful medical AI assistant specializing in providing accurate, "
                 "evidence-based answers to medical questions.")
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template (simplified)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.replace("<|eot_id|>", "").strip()
    else:
        response = full_response[len(prompt):].strip()
    
    return response


def evaluate_model(model, tokenizer, dataset, config: EvaluationConfig, model_name: str) -> Dict:
    """Evaluate model on dataset."""
    metrics_calculator = AutomaticMetrics()
    results = []
    specialty_results = defaultdict(list)
    
    print(f"Evaluating {model_name}...")
    
    for i, example in enumerate(tqdm(dataset)):
        question = example['question']
        reference = example['response']
        specialty = example['specialty']
        
        # Generate response
        hypothesis = generate_response(model, tokenizer, question, config)
        
        # Compute metrics
        bleu_scores = metrics_calculator.compute_bleu(reference, hypothesis)
        rouge_scores = metrics_calculator.compute_rouge(reference, hypothesis)
        meteor_score = metrics_calculator.compute_meteor(reference, hypothesis)
        exact_match = metrics_calculator.compute_exact_match(reference, hypothesis)
        token_f1 = metrics_calculator.compute_token_f1(reference, hypothesis)
        
        # Combine all metrics
        sample_metrics = {
            'question': question,
            'reference': reference,
            'hypothesis': hypothesis,
            'specialty': specialty,
            'exact_match': exact_match,
            'token_f1': token_f1,
            'meteor': meteor_score,
            **bleu_scores,
            **rouge_scores
        }
        
        results.append(sample_metrics)
        specialty_results[specialty].append(sample_metrics)
    
    return results, specialty_results


def compute_bert_scores(results: List[Dict]) -> List[Dict]:
    """Compute BERTScore for all predictions."""
    print("Computing BERTScores...")
    
    references = [r['reference'] for r in results]
    hypotheses = [r['hypothesis'] for r in results]
    
    # Compute BERTScore
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
    
    # Add to results
    for i, result in enumerate(results):
        result['bertscore_precision'] = P[i].item()
        result['bertscore_recall'] = R[i].item()
        result['bertscore_f1'] = F1[i].item()
    
    return results


def analyze_specialty_performance(specialty_results: Dict[str, List], output_dir: str):
    """Analyze performance by medical specialty."""
    specialty_stats = {}
    
    for specialty, results in specialty_results.items():
        metrics = {
            'BLEU-1': np.mean([r['BLEU-1'] for r in results]),
            'BLEU-4': np.mean([r['BLEU-4'] for r in results]),
            'ROUGE-1': np.mean([r['ROUGE-1'] for r in results]),
            'ROUGE-L': np.mean([r['ROUGE-L'] for r in results]),
            'METEOR': np.mean([r['meteor'] for r in results]),
            'Exact Match': np.mean([r['exact_match'] for r in results]),
            'Token F1': np.mean([r['token_f1'] for r in results]),
            'BERTScore F1': np.mean([r['bertscore_f1'] for r in results]),
            'Sample Count': len(results)
        }
        specialty_stats[specialty] = metrics
    
    # Create DataFrame for analysis
    df = pd.DataFrame(specialty_stats).T
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, "specialty_performance.csv"))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot ROUGE-L scores by specialty
    plt.subplot(2, 2, 1)
    df_sorted = df.sort_values('ROUGE-L', ascending=True)
    plt.barh(range(len(df_sorted)), df_sorted['ROUGE-L'])
    plt.yticks(range(len(df_sorted)), df_sorted.index, fontsize=8)
    plt.xlabel('ROUGE-L Score')
    plt.title('ROUGE-L Performance by Specialty')
    plt.tight_layout()
    
    # Plot BERTScore F1 by specialty
    plt.subplot(2, 2, 2)
    df_sorted = df.sort_values('BERTScore F1', ascending=True)
    plt.barh(range(len(df_sorted)), df_sorted['BERTScore F1'])
    plt.yticks(range(len(df_sorted)), df_sorted.index, fontsize=8)
    plt.xlabel('BERTScore F1')
    plt.title('BERTScore F1 Performance by Specialty')
    
    # Plot Token F1 by specialty
    plt.subplot(2, 2, 3)
    df_sorted = df.sort_values('Token F1', ascending=True)
    plt.barh(range(len(df_sorted)), df_sorted['Token F1'])
    plt.yticks(range(len(df_sorted)), df_sorted.index, fontsize=8)
    plt.xlabel('Token F1 Score')
    plt.title('Medical Token F1 Performance by Specialty')
    
    # Plot sample distribution
    plt.subplot(2, 2, 4)
    plt.bar(range(len(df)), df['Sample Count'])
    plt.xticks(range(len(df)), df.index, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution by Specialty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "specialty_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def identify_low_performance_specialties(specialty_df: pd.DataFrame, threshold: float = 0.3) -> List[str]:
    """Identify specialties with low performance for targeted improvement."""
    low_performance = []
    
    # Check multiple metrics
    metrics_to_check = ['ROUGE-L', 'BERTScore F1', 'Token F1']
    
    for specialty in specialty_df.index:
        low_scores = 0
        for metric in metrics_to_check:
            if specialty_df.loc[specialty, metric] < threshold:
                low_scores += 1
        
        if low_scores >= 2:  # If low in at least 2 metrics
            low_performance.append(specialty)
    
    return low_performance


def create_comparison_report(base_results: List[Dict], ft_results: List[Dict], output_dir: str):
    """Create comparison report between base and fine-tuned models."""
    base_metrics = {}
    ft_metrics = {}
    
    # Calculate overall metrics
    metrics = ['BLEU-1', 'BLEU-4', 'ROUGE-1', 'ROUGE-L', 'meteor', 'exact_match', 'token_f1', 'bertscore_f1']
    
    for metric in metrics:
        base_metrics[metric] = np.mean([r[metric] for r in base_results])
        ft_metrics[metric] = np.mean([r[metric] for r in ft_results])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Base Model': base_metrics,
        'Fine-tuned Model': ft_metrics,
        'Improvement': {k: ft_metrics[k] - base_metrics[k] for k in metrics}
    })
    
    # Save comparison
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], [base_metrics[m] for m in metrics], 
            width, label='Base Model', alpha=0.8)
    plt.bar([i + width/2 for i in x], [ft_metrics[m] for m in metrics], 
            width, label='Fine-tuned Model', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df


def main():
    """Main evaluation function."""
    config = EvaluationConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load holdout dataset
    print("Loading holdout dataset...")
    holdout_dataset = load_from_disk(os.path.join(config.DATASET_DIR, "holdout_dataset"))
    print(f"Evaluating on {len(holdout_dataset)} samples")
    
    # Evaluate base model
    print("\n=== Evaluating Base Model ===")
    base_model, base_tokenizer = load_model_and_tokenizer(config.BASE_MODEL_PATH, is_peft=False)
    base_results, base_specialty_results = evaluate_model(
        base_model, base_tokenizer, holdout_dataset, config, "Base Model"
    )
    
    # Compute BERTScores
    base_results = compute_bert_scores(base_results)
    
    # Clean up base model to save memory
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n=== Evaluating Fine-tuned Model ===")
    ft_model, ft_tokenizer = load_model_and_tokenizer(config.FINETUNED_MODEL_PATH, is_peft=True)
    ft_results, ft_specialty_results = evaluate_model(
        ft_model, ft_tokenizer, holdout_dataset, config, "Fine-tuned Model"
    )
    
    # Compute BERTScores
    ft_results = compute_bert_scores(ft_results)
    
    # Save detailed results
    if config.SAVE_PREDICTIONS:
        pd.DataFrame(base_results).to_csv(os.path.join(config.OUTPUT_DIR, "base_model_results.csv"), index=False)
        pd.DataFrame(ft_results).to_csv(os.path.join(config.OUTPUT_DIR, "finetuned_model_results.csv"), index=False)
    
    # Analyze specialty performance
    print("\n=== Analyzing Specialty Performance ===")
    base_specialty_df = analyze_specialty_performance(base_specialty_results, config.OUTPUT_DIR)
    base_specialty_df.to_csv(os.path.join(config.OUTPUT_DIR, "base_specialty_performance.csv"))
    
    ft_specialty_df = analyze_specialty_performance(ft_specialty_results, config.OUTPUT_DIR)
    ft_specialty_df.to_csv(os.path.join(config.OUTPUT_DIR, "finetuned_specialty_performance.csv"))
    
    # Create model comparison
    print("\n=== Creating Model Comparison ===")
    comparison_df = create_comparison_report(base_results, ft_results, config.OUTPUT_DIR)
    
    # Identify low-performance specialties
    low_performance_specialties = identify_low_performance_specialties(ft_specialty_df)
    
    # Create recommendations for next training iteration
    recommendations = {
        'low_performance_specialties': low_performance_specialties,
        'recommended_sample_increases': {
            specialty: 0.8  # Increase to 80% of available data
            for specialty in low_performance_specialties
        },
        'overall_improvement': comparison_df['Improvement'].to_dict()
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "training_recommendations.json"), 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"Low-performance specialties: {low_performance_specialties}")
    print(f"Overall ROUGE-L improvement: {comparison_df.loc['ROUGE-L', 'Improvement']:.4f}")
    print(f"Overall BERTScore F1 improvement: {comparison_df.loc['bertscore_f1', 'Improvement']:.4f}")


if __name__ == "__main__":
    main()
