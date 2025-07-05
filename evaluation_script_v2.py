import os
import json
import torch
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Core ML libraries

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_from_disk
from peft import PeftModel

# NLP evaluation libraries

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Medical domain libraries

from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data

for resource in [‘punkt’, ‘wordnet’, ‘averaged_perceptron_tagger’, ‘stopwords’]:
try:
nltk.data.find(f’tokenizers/{resource}’ if resource == ‘punkt’ else f’corpora/{resource}’)
except LookupError:
nltk.download(resource)

class MedicalEvaluationConfig:
def **init**(self):
# Model paths
self.BASE_MODEL_PATH = “dmis-lab/llama-3-meerkat-8b-v1.0”
self.FINETUNED_MODEL_PATH = “./results”
self.DATASET_DIR = “./datasets”

```
    # Medical NER model paths (LOCAL)
    self.MEDICAL_NER_MODEL_PATH = "./models/medical_ner"  # Download to this folder
    self.SENTENCE_TRANSFORMER_PATH = "./models/sentence_transformer"  # Download to this folder
    
    # Generation settings
    self.MAX_NEW_TOKENS = 512
    self.TEMPERATURE = 0.1
    self.TOP_P = 0.9
    self.DO_SAMPLE = True
    self.BATCH_SIZE = 4
    
    # Output settings
    self.OUTPUT_DIR = "./comprehensive_evaluation_results"
    self.SAVE_PREDICTIONS = True
    self.SAVE_DETAILED_ANALYSIS = True
    
    # Medical entity categories
    self.MEDICAL_ENTITIES = {
        'drugs': ['medication', 'drug', 'medicine', 'prescription', 'dosage'],
        'symptoms': ['symptom', 'pain', 'ache', 'fever', 'nausea', 'headache'],
        'doctors': ['doctor', 'physician', 'specialist', 'cardiologist', 'neurologist']
    }
```

class MedicalEntityExtractor:
“””
Comprehensive medical entity extraction using both regex and NER models.
“””

```
def __init__(self, config: MedicalEvaluationConfig):
    self.config = config
    self.setup_extractors()
    
def setup_extractors(self):
    """Setup both regex patterns and NER model."""
    # Setup regex patterns for medical entities
    self.regex_patterns = {
        'drugs': [
            r'\b\w*(?:cillin|mycin|prazole|statin|blocker)\b',  # Common drug suffixes
            r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin)\b',  # Common drugs
            r'\b\d+\s*mg\b|\b\d+\s*ml\b|\bmg/kg\b',  # Dosages
            r'\b(?:tablet|capsule|injection|syrup)s?\b',  # Drug forms
        ],
        'symptoms': [
            r'\b(?:pain|ache|fever|nausea|vomiting|diarrhea|constipation)\b',
            r'\b(?:headache|dizziness|fatigue|weakness|shortness of breath)\b',
            r'\b(?:swelling|inflammation|rash|itching|burning)\b',
            r'\b(?:chest pain|abdominal pain|back pain|joint pain)\b',
        ],
        'doctors': [
            r'\b(?:doctor|physician|specialist|MD|Dr\.)\b',
            r'\b(?:cardiologist|neurologist|dermatologist|oncologist)\b',
            r'\b(?:surgeon|psychiatrist|pediatrician|gynecologist)\b',
            r'\b(?:radiologist|pathologist|anesthesiologist)\b',
        ],
        'procedures': [
            r'\b(?:surgery|operation|procedure|biopsy|scan)\b',
            r'\b(?:MRI|CT scan|X-ray|ultrasound|endoscopy)\b',
            r'\b(?:blood test|ECG|EKG|colonoscopy)\b',
        ],
        'conditions': [
            r'\b\w*(?:itis|osis|emia|pathy|trophy)s?\b',  # Medical condition suffixes
            r'\b(?:diabetes|hypertension|cancer|infection)\b',
            r'\b(?:pneumonia|bronchitis|arthritis|hepatitis)\b',
        ]
    }
    
    # Setup medical NER model (local)
    self.setup_medical_ner()
    
def setup_medical_ner(self):
    """Setup medical NER model from local path."""
    try:
        if os.path.exists(self.config.MEDICAL_NER_MODEL_PATH):
            # Load from local path
            self.medical_ner = pipeline(
                "ner",
                model=self.config.MEDICAL_NER_MODEL_PATH,
                tokenizer=self.config.MEDICAL_NER_MODEL_PATH,
                aggregation_strategy="simple"
            )
            print(f"Loaded medical NER model from {self.config.MEDICAL_NER_MODEL_PATH}")
        else:
            print(f"Medical NER model not found at {self.config.MEDICAL_NER_MODEL_PATH}")
            print("Please download: 'Clinical-AI-Apollo/Medical-NER' to the specified folder")
            self.medical_ner = None
    except Exception as e:
        print(f"Error loading medical NER model: {e}")
        self.medical_ner = None

def extract_regex_entities(self, text: str) -> Dict[str, Set[str]]:
    """Extract medical entities using regex patterns."""
    entities = defaultdict(set)
    text_lower = text.lower()
    
    for category, patterns in self.regex_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities[category].update(matches)
    
    return dict(entities)

def extract_ner_entities(self, text: str) -> Dict[str, Set[str]]:
    """Extract medical entities using NER model."""
    entities = defaultdict(set)
    
    if self.medical_ner is None:
        return dict(entities)
    
    try:
        ner_results = self.medical_ner(text)
        
        for entity in ner_results:
            entity_text = entity['word'].lower().strip()
            entity_label = entity['entity_group'].lower()
            
            # Map NER labels to our categories
            if 'drug' in entity_label or 'medication' in entity_label:
                entities['drugs'].add(entity_text)
            elif 'symptom' in entity_label or 'sign' in entity_label:
                entities['symptoms'].add(entity_text)
            elif 'doctor' in entity_label or 'physician' in entity_label:
                entities['doctors'].add(entity_text)
            elif 'procedure' in entity_label or 'treatment' in entity_label:
                entities['procedures'].add(entity_text)
            elif 'condition' in entity_label or 'disease' in entity_label:
                entities['conditions'].add(entity_text)
    except Exception as e:
        print(f"Error in NER extraction: {e}")
    
    return dict(entities)

def extract_all_entities(self, text: str) -> Dict[str, Set[str]]:
    """Combine regex and NER entity extraction."""
    regex_entities = self.extract_regex_entities(text)
    ner_entities = self.extract_ner_entities(text)
    
    # Combine results
    combined_entities = defaultdict(set)
    all_categories = set(regex_entities.keys()) | set(ner_entities.keys())
    
    for category in all_categories:
        combined_entities[category] = (
            regex_entities.get(category, set()) | 
            ner_entities.get(category, set())
        )
    
    return dict(combined_entities)
```

class MedicalSemanticSimilarity:
“””
Medical domain semantic similarity using sentence transformers.
“””

```
def __init__(self, config: MedicalEvaluationConfig):
    self.config = config
    self.setup_model()

def setup_model(self):
    """Setup medical domain sentence transformer."""
    try:
        if os.path.exists(self.config.SENTENCE_TRANSFORMER_PATH):
            self.model = SentenceTransformer(self.config.SENTENCE_TRANSFORMER_PATH)
            print(f"Loaded sentence transformer from {self.config.SENTENCE_TRANSFORMER_PATH}")
        else:
            print(f"Sentence transformer not found at {self.config.SENTENCE_TRANSFORMER_PATH}")
            print("Please download: 'pritamdeka/S-BioBert-snli-multinli-stsb' to the specified folder")
            # Fallback to a general model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Using fallback general sentence transformer")
    except Exception as e:
        print(f"Error loading sentence transformer: {e}")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(self, text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts."""
    try:
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0
```

class MedicalQAMetrics:
“””
Comprehensive metrics for medical QA evaluation.
“””

```
def __init__(self, config: MedicalEvaluationConfig):
    self.config = config
    self.entity_extractor = MedicalEntityExtractor(config)
    self.semantic_similarity = MedicalSemanticSimilarity(config)
    self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    self.smoothing = SmoothingFunction().method1

def compute_traditional_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute traditional NLP metrics."""
    metrics = {}
    
    # BLEU scores
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    bleu_weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    bleu_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    
    for weight, name in zip(bleu_weights, bleu_names):
        score = sentence_bleu([ref_tokens], hyp_tokens, weights=weight, smoothing_function=self.smoothing)
        metrics[name] = score
    
    # ROUGE scores
    rouge_scores = self.rouge_scorer.score(reference, hypothesis)
    metrics.update({
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure
    })
    
    # METEOR score
    try:
        metrics['METEOR'] = meteor_score([ref_tokens], hyp_tokens)
    except:
        metrics['METEOR'] = 0.0
    
    return metrics

def compute_medical_entity_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute medical entity overlap metrics."""
    ref_entities = self.entity_extractor.extract_all_entities(reference)
    hyp_entities = self.entity_extractor.extract_all_entities(hypothesis)
    
    metrics = {}
    
    # Entity-wise metrics
    for category in ['drugs', 'symptoms', 'doctors', 'procedures', 'conditions']:
        ref_set = ref_entities.get(category, set())
        hyp_set = hyp_entities.get(category, set())
        
        if len(ref_set) == 0 and len(hyp_set) == 0:
            precision = recall = f1 = 1.0
        elif len(ref_set) == 0:
            precision = recall = f1 = 0.0
        elif len(hyp_set) == 0:
            precision = recall = f1 = 0.0
        else:
            intersection = len(ref_set & hyp_set)
            precision = intersection / len(hyp_set)
            recall = intersection / len(ref_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'{category}_precision'] = precision
        metrics[f'{category}_recall'] = recall
        metrics[f'{category}_f1'] = f1
    
    # Overall medical entity metrics
    all_ref_entities = set()
    all_hyp_entities = set()
    for entities in ref_entities.values():
        all_ref_entities.update(entities)
    for entities in hyp_entities.values():
        all_hyp_entities.update(entities)
    
    if len(all_ref_entities) == 0 and len(all_hyp_entities) == 0:
        metrics['medical_entity_precision'] = 1.0
        metrics['medical_entity_recall'] = 1.0
        metrics['medical_entity_f1'] = 1.0
    elif len(all_ref_entities) == 0:
        metrics['medical_entity_precision'] = 0.0
        metrics['medical_entity_recall'] = 0.0
        metrics['medical_entity_f1'] = 0.0
    elif len(all_hyp_entities) == 0:
        metrics['medical_entity_precision'] = 0.0
        metrics['medical_entity_recall'] = 0.0
        metrics['medical_entity_f1'] = 0.0
    else:
        intersection = len(all_ref_entities & all_hyp_entities)
        precision = intersection / len(all_hyp_entities)
        recall = intersection / len(all_ref_entities)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['medical_entity_precision'] = precision
        metrics['medical_entity_recall'] = recall
        metrics['medical_entity_f1'] = f1
    
    return metrics

def compute_semantic_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute semantic similarity metrics."""
    similarity = self.semantic_similarity.compute_similarity(reference, hypothesis)
    return {'semantic_similarity': similarity}

def compute_medical_quality_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute medical-specific quality metrics."""
    metrics = {}
    
    # Dosage accuracy
    ref_dosages = re.findall(r'\b\d+\s*(?:mg|ml|g|mg/kg)\b', reference.lower())
    hyp_dosages = re.findall(r'\b\d+\s*(?:mg|ml|g|mg/kg)\b', hypothesis.lower())
    
    dosage_overlap = len(set(ref_dosages) & set(hyp_dosages))
    total_ref_dosages = len(set(ref_dosages))
    
    if total_ref_dosages > 0:
        metrics['dosage_accuracy'] = dosage_overlap / total_ref_dosages
    else:
        metrics['dosage_accuracy'] = 1.0 if len(hyp_dosages) == 0 else 0.0
    
    # Safety mention (contraindications, warnings)
    safety_terms = ['contraindic', 'warning', 'caution', 'avoid', 'not recommend', 'side effect']
    ref_has_safety = any(term in reference.lower() for term in safety_terms)
    hyp_has_safety = any(term in hypothesis.lower() for term in safety_terms)
    
    if ref_has_safety:
        metrics['safety_mention_recall'] = 1.0 if hyp_has_safety else 0.0
    else:
        metrics['safety_mention_recall'] = 1.0  # No safety info needed
    
    # Medical reasoning presence
    reasoning_terms = ['because', 'due to', 'caused by', 'results in', 'leads to', 'therefore']
    ref_has_reasoning = any(term in reference.lower() for term in reasoning_terms)
    hyp_has_reasoning = any(term in hypothesis.lower() for term in reasoning_terms)
    
    if ref_has_reasoning:
        metrics['reasoning_presence'] = 1.0 if hyp_has_reasoning else 0.0
    else:
        metrics['reasoning_presence'] = 1.0  # No reasoning needed
    
    return metrics

def compute_all_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute all metrics for a single prediction."""
    all_metrics = {}
    
    # Traditional NLP metrics
    all_metrics.update(self.compute_traditional_metrics(reference, hypothesis))
    
    # Medical entity metrics
    all_metrics.update(self.compute_medical_entity_metrics(reference, hypothesis))
    
    # Semantic similarity
    all_metrics.update(self.compute_semantic_metrics(reference, hypothesis))
    
    # Medical quality metrics
    all_metrics.update(self.compute_medical_quality_metrics(reference, hypothesis))
    
    return all_metrics
```

def load_model_and_tokenizer(model_path: str, is_peft: bool = False):
“”“Load model and tokenizer for evaluation.”””
print(f”Loading model from {model_path}…”)

```
tokenizer = AutoTokenizer.from_pretrained(
    model_path if not is_peft else "dmis-lab/llama-3-meerkat-8b-v1.0"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if is_peft:
    base_model = AutoModelForCausalLM.from_pretrained(
        "dmis-lab/llama-3-meerkat-8b-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

model.eval()
return model, tokenizer
```

def generate_response(model, tokenizer, question: str, config: MedicalEvaluationConfig) -> str:
“”“Generate response for a given question.”””
system_msg = (“You are a helpful medical AI assistant specializing in providing accurate, “
“evidence-based answers to medical questions.”)

```
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

full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
    response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    response = response.replace("<|eot_id|>", "").strip()
else:
    response = full_response[len(prompt):].strip()

return response
```

def evaluate_model_comprehensive(model, tokenizer, dataset, config: MedicalEvaluationConfig,
model_name: str, metrics_calculator: MedicalQAMetrics) -> Tuple[List[Dict], Dict]:
“”“Comprehensive evaluation of a model.”””
results = []
specialty_results = defaultdict(list)

```
print(f"Evaluating {model_name} with comprehensive metrics...")

for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
    question = example['question']
    reference = example['response']
    specialty = example['specialty']
    
    # Generate response
    hypothesis = generate_response(model, tokenizer, question, config)
    
    # Compute all metrics
    all_metrics = metrics_calculator.compute_all_metrics(reference, hypothesis)
    
    # Create result record
    result = {
        'question': question,
        'reference': reference,
        'hypothesis': hypothesis,
        'specialty': specialty,
        **all_metrics
    }
    
    results.append(result)
    specialty_results[specialty].append(result)

return results, specialty_results
```

def compute_bert_scores_batch(results: List[Dict]) -> List[Dict]:
“”“Compute BERTScore for all predictions in batch.”””
print(“Computing BERTScores…”)

```
references = [r['reference'] for r in results]
hypotheses = [r['hypothesis'] for r in results]

P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)

for i, result in enumerate(results):
    result['bertscore_precision'] = P[i].item()
    result['bertscore_recall'] = R[i].item()
    result['bertscore_f1'] = F1[i].item()

return results
```

def create_detailed_error_analysis(specialty_results: Dict, output_dir: str, model_name: str):
“”“Create detailed error analysis report.”””
analysis_results = {}

```
for specialty, results in specialty_results.items():
    specialty_analysis = {
        'total_samples': len(results),
        'avg_metrics': {},
        'low_performing_samples': [],
        'missing_entities_analysis': {},
        'common_errors': []
    }
    
    # Calculate average metrics
    if results:
        metric_keys = [k for k in results[0].keys() if isinstance(results[0][k], (int, float))]
        for metric in metric_keys:
            values = [r[metric] for r in results if metric in r]
            specialty_analysis['avg_metrics'][metric] = np.mean(values) if values else 0.0
    
    # Identify low-performing samples (bottom 20% in medical entity F1)
    if results:
        sorted_by_medical_f1 = sorted(results, key=lambda x: x.get('medical_entity_f1', 0))
        low_count = max(1, len(sorted_by_medical_f1) // 5)
        
        for sample in sorted_by_medical_f1[:low_count]:
            specialty_analysis['low_performing_samples'].append({
                'question': sample['question'][:100] + "..." if len(sample['question']) > 100 else sample['question'],
                'medical_entity_f1': sample.get('medical_entity_f1', 0),
                'rouge_l': sample.get('ROUGE-L', 0),
                'semantic_similarity': sample.get('semantic_similarity', 0)
            })
    
    # Analyze missing entities
    entity_categories = ['drugs', 'symptoms', 'doctors', 'procedures', 'conditions']
    for category in entity_categories:
        recall_scores = [r.get(f'{category}_recall', 0) for r in results]
        low_recall_count = sum(1 for score in recall_scores if score < 0.5)
        specialty_analysis['missing_entities_analysis'][category] = {
            'avg_recall': np.mean(recall_scores) if recall_scores else 0.0,
            'low_recall_samples': low_recall_count,
            'percentage_problematic': (low_recall_count / len(recall_scores) * 100) if recall_scores else 0.0
        }
    
    analysis_results[specialty] = specialty_analysis

# Save detailed analysis
with open(os.path.join(output_dir, f"{model_name}_detailed_error_analysis.json"), 'w') as f:
    json.dump(analysis_results, f, indent=2, default=str)

return analysis_results
```

def create_comprehensive_visualizations(base_results: List[Dict], ft_results: List[Dict],
base_specialty_results: Dict, ft_specialty_results: Dict,
output_dir: str):
“”“Create comprehensive visualizations for evaluation results.”””

```
# 1. Overall metrics comparison
plt.figure(figsize=(16, 12))

# Define key metrics to visualize
key_metrics = [
    'ROUGE-L', 'BLEU-4', 'semantic_similarity', 'medical_entity_f1',
    'drugs_f1', 'symptoms_f1', 'doctors_f1', 'bertscore_f1'
]

base_scores = [np.mean([r.get(metric, 0) for r in base_results]) for metric in key_metrics]
ft_scores = [np.mean([r.get(metric, 0) for r in ft_results]) for metric in key_metrics]

x = np.arange(len(key_metrics))
width = 0.35

plt.subplot(2, 2, 1)
plt.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8)
plt.bar(x + width/2, ft_scores, width, label='Fine-tuned Model', alpha=0.8)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Overall Performance Comparison')
plt.xticks(x, key_metrics, rotation=45)
plt.legend()
plt.tight_layout()

# 2. Medical entity performance by specialty
plt.subplot(2, 2, 2)
specialties = list(ft_specialty_results.keys())[:10]  # Top 10 specialties
entity_f1_scores = []

for specialty in specialties:
    specialty_data = ft_specialty_results[specialty]
    avg_f1 = np.mean([r.get('medical_entity_f1', 0) for r in specialty_data])
    entity_f1_scores.append(avg_f1)

plt.barh(range(len(specialties)), entity_f1_scores)
plt.yticks(range(len(specialties)), [s.replace('_', ' ').title() for s in specialties])
plt.xlabel('Medical Entity F1 Score')
plt.title('Medical Entity Performance by Specialty')

# 3. Semantic similarity distribution
plt.subplot(2, 2, 3)
base_semantic = [r.get('semantic_similarity', 0) for r in base_results]
ft_semantic = [r.get('semantic_similarity', 0) for r in ft_results]

plt.hist(base_semantic, bins=30, alpha=0.7, label='Base Model', density=True)
plt.hist(ft_semantic, bins=30, alpha=0.7, label='Fine-tuned Model', density=True)
plt.xlabel('Semantic Similarity Score')
plt.ylabel('Density')
plt.title('Semantic Similarity Distribution')
plt.legend()

# 4. Entity category performance
plt.subplot(2, 2, 4)
entity_categories = ['drugs', 'symptoms', 'doctors', 'procedures', 'conditions']
base_entity_scores = []
ft_entity_scores = []

for category in entity_categories:
    base_avg = np.mean([r.get(f'{category}_f1', 0) for r in base_results])
    ft_avg = np.mean([r.get(f'{category}_f1', 0) for r in ft_results])
    base_entity_scores.append(base_avg)
    ft_entity_scores.append(ft_avg)

x = np.arange(len(entity_categories))
plt.bar(x - width/2, base_entity_scores, width, label='Base Model', alpha=0.8)
plt.bar(x + width/2, ft_entity_scores, width, label='Fine-tuned Model', alpha=0.8)
plt.xlabel('Entity Categories')
plt.ylabel('F1 Score')
plt.title('Entity Category Performance')
plt.xticks(x, entity_categories, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comprehensive_evaluation_results.png"), dpi=300, bbox_inches='tight')
plt.close()
```

def main():
“”“Main comprehensive evaluation function.”””
config = MedicalEvaluationConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

```
print("=== COMPREHENSIVE MEDICAL QA EVALUATION ===")
print(f"Output directory: {config.OUTPUT_DIR}")

# Initialize metrics calculator
print("Initializing medical metrics calculator...")
metrics_calculator = MedicalQAMetrics(config)

# Load holdout dataset
print("Loading holdout dataset...")
try:
    holdout_dataset = load_from_disk(os.path.join(config.DATASET_DIR, "holdout_dataset"))
    print(f"Loaded {len(holdout_dataset)} samples for evaluation")
except Exception as e:
    print(f"Error loading dataset: {e}")
    return

# Evaluate base model
print("\n=== EVALUATING BASE MODEL ===")
try:
    base_model, base_tokenizer = load_model_and_tokenizer(config.BASE_MODEL_PATH, is_peft=False)
    base_results, base_specialty_results = evaluate_model_comprehensive(
        base_model, base_tokenizer, holdout_dataset, config, "Base Model", metrics_calculator
    )
    
    # Add BERTScores
    base_results = compute_bert_scores_batch(base_results)
    
    # Clean up base model to save memory
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    
    print(f"Base model evaluation completed: {len(base_results)} samples")
    
except Exception as e:
    print(f"Error evaluating base model: {e}")
    return

# Evaluate fine-tuned model
print("\n=== EVALUATING FINE-TUNED MODEL ===")
try:
    ft_model, ft_tokenizer = load_model_and_tokenizer(config.FINETUNED_MODEL_PATH, is_peft=True)
    ft_results, ft_specialty_results = evaluate_model_comprehensive(
        ft_model, ft_tokenizer, holdout_dataset, config, "Fine-tuned Model", metrics_calculator
    )
    
    # Add BERTScores
    ft_results = compute_bert_scores_batch(ft_results)
    
    print(f"Fine-tuned model evaluation completed: {len(ft_results)} samples")
    
except Exception as e:
    print(f"Error evaluating fine-tuned model: {e}")
    return

# Save detailed predictions if requested
if config.SAVE_PREDICTIONS:
    print("\n=== SAVING DETAILED PREDICTIONS ===")
    base_df = pd.DataFrame(base_results)
    ft_df = pd.DataFrame(ft_results)
    
    base_df.to_csv(os.path.join(config.OUTPUT_DIR, "base_model_detailed_results.csv"), index=False)
    ft_df.to_csv(os.path.join(config.OUTPUT_DIR, "finetuned_model_detailed_results.csv"), index=False)
    print("Detailed predictions saved")

# Create detailed error analysis
if config.SAVE_DETAILED_ANALYSIS:
    print("\n=== CREATING DETAILED ERROR ANALYSIS ===")
    base_analysis = create_detailed_error_analysis(base_specialty_results, config.OUTPUT_DIR, "base_model")
    ft_analysis = create_detailed_error_analysis(ft_specialty_results, config.OUTPUT_DIR, "finetuned_model")
    print("Detailed error analysis completed")

# Calculate and save overall metrics comparison
print("\n=== COMPUTING OVERALL METRICS COMPARISON ===")

# Get all metric keys
metric_keys = [k for k in base_results[0].keys() if isinstance(base_results[0][k], (int, float))]

# Calculate overall averages
base_overall = {}
ft_overall = {}
improvements = {}

for metric in metric_keys:
    base_avg = np.mean([r.get(metric, 0) for r in base_results])
    ft_avg = np.mean([r.get(metric, 0) for r in ft_results])
    improvement = ft_avg - base_avg
    
    base_overall[metric] = base_avg
    ft_overall[metric] = ft_avg
    improvements[metric] = improvement

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Base_Model': base_overall,
    'Fine_tuned_Model': ft_overall,
    'Improvement': improvements,
    'Percent_Improvement': {k: (improvements[k] / base_overall[k] * 100) if base_overall[k] != 0 else 0 
                           for k in metric_keys}
})

comparison_df.to_csv(os.path.join(config.OUTPUT_DIR, "comprehensive_metrics_comparison.csv"))

# Create specialty-wise comparison
print("\n=== CREATING SPECIALTY-WISE ANALYSIS ===")
specialty_comparison = {}

for specialty in base_specialty_results.keys():
    if specialty in ft_specialty_results:
        base_spec_results = base_specialty_results[specialty]
        ft_spec_results = ft_specialty_results[specialty]
        
        specialty_metrics = {}
        for metric in ['ROUGE-L', 'semantic_similarity', 'medical_entity_f1', 'drugs_f1', 'symptoms_f1', 'doctors_f1']:
            base_avg = np.mean([r.get(metric, 0) for r in base_spec_results])
            ft_avg = np.mean([r.get(metric, 0) for r in ft_spec_results])
            
            specialty_metrics[f'{metric}_base'] = base_avg
            specialty_metrics[f'{metric}_ft'] = ft_avg
            specialty_metrics[f'{metric}_improvement'] = ft_avg - base_avg
        
        specialty_metrics['sample_count'] = len(base_spec_results)
        specialty_comparison[specialty] = specialty_metrics

specialty_df = pd.DataFrame(specialty_comparison).T
specialty_df.to_csv(os.path.join(config.OUTPUT_DIR, "specialty_wise_comparison.csv"))

# Create comprehensive visualizations
print("\n=== CREATING VISUALIZATIONS ===")
create_comprehensive_visualizations(
    base_results, ft_results, 
    base_specialty_results, ft_specialty_results, 
    config.OUTPUT_DIR
)

# Identify problematic specialties and provide recommendations
print("\n=== GENERATING RECOMMENDATIONS ===")

# Find specialties with low medical entity F1 scores
low_performing_specialties = []
for specialty, metrics in specialty_comparison.items():
    if metrics['medical_entity_f1_ft'] < 0.5:  # Threshold for low performance
        low_performing_specialties.append({
            'specialty': specialty,
            'medical_entity_f1': metrics['medical_entity_f1_ft'],
            'sample_count': metrics['sample_count'],
            'improvement': metrics['medical_entity_f1_improvement']
        })

# Sort by performance
low_performing_specialties.sort(key=lambda x: x['medical_entity_f1'])

recommendations = {
    'low_performing_specialties': low_performing_specialties[:10],  # Top 10 problematic
    'recommended_actions': {
        specialty['specialty']: {
            'increase_training_data': True,
            'focus_on_entity_extraction': True,
            'recommended_data_increase': '50-80%' if specialty['medical_entity_f1'] < 0.3 else '30-50%'
        } for specialty in low_performing_specialties[:5]
    },
    'overall_performance': {
        'total_samples_evaluated': len(base_results),
        'overall_improvement_rouge_l': improvements.get('ROUGE-L', 0),
        'overall_improvement_medical_entities': improvements.get('medical_entity_f1', 0),
        'overall_improvement_semantic': improvements.get('semantic_similarity', 0)
    }
}

with open(os.path.join(config.OUTPUT_DIR, "training_recommendations.json"), 'w') as f:
    json.dump(recommendations, f, indent=2, default=str)

# Print summary
print("\n=== EVALUATION SUMMARY ===")
print(f"Total samples evaluated: {len(base_results)}")
print(f"Number of specialties: {len(base_specialty_results)}")
print("\nKey Improvements (Fine-tuned vs Base):")
print(f"  ROUGE-L: {improvements.get('ROUGE-L', 0):.4f}")
print(f"  Semantic Similarity: {improvements.get('semantic_similarity', 0):.4f}")
print(f"  Medical Entity F1: {improvements.get('medical_entity_f1', 0):.4f}")
print(f"  Drug Entity F1: {improvements.get('drugs_f1', 0):.4f}")
print(f"  Symptom Entity F1: {improvements.get('symptoms_f1', 0):.4f}")
print(f"  Doctor Entity F1: {improvements.get('doctors_f1', 0):.4f}")
print(f"  BERTScore F1: {improvements.get('bertscore_f1', 0):.4f}")

print(f"\nLow-performing specialties: {len(low_performing_specialties)}")
for specialty in low_performing_specialties[:5]:
    print(f"  {specialty['specialty']}: {specialty['medical_entity_f1']:.3f}")

print(f"\nResults saved to: {config.OUTPUT_DIR}")
print("Files generated:")
print("  - comprehensive_metrics_comparison.csv")
print("  - specialty_wise_comparison.csv")
print("  - base_model_detailed_results.csv")
print("  - finetuned_model_detailed_results.csv")
print("  - base_model_detailed_error_analysis.json")
print("  - finetuned_model_detailed_error_analysis.json")
print("  - training_recommendations.json")
print("  - comprehensive_evaluation_results.png")
```

if **name** == “**main**”:
print(“Starting comprehensive medical QA evaluation…”)
print(”\nREQUIRED MODEL DOWNLOADS:”)
print(“1. Medical NER Model: ‘Clinical-AI-Apollo/Medical-NER’”)
print(”   Download to: ./models/medical_ner/”)
print(“2. Medical Sentence Transformer: ‘pritamdeka/S-BioBert-snli-multinli-stsb’”)
print(”   Download to: ./models/sentence_transformer/”)
print(”\nDownload commands:”)
print(“git clone https://huggingface.co/Clinical-AI-Apollo/Medical-NER ./models/medical_ner”)
print(“git clone https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb ./models/sentence_transformer”)
print(”\n” + “=”*80)

```
main()
```

