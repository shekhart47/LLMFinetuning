import os
import json
import torch
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import openai
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Core ML libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk, Dataset
from peft import PeftModel

class MedicalQAEvaluationConfig:
    """Configuration class for medical QA evaluation."""
    
    def __init__(self):
        # Model paths - Updated to match your training script
        self.BASE_MODEL_PATH = "./llama-3-meerkat-8b-v1.0"
        self.FINETUNED_MODEL_PATH = "./results"
        self.DATASET_PATH = "../../datasets/final_query_set/final_dataset/holdout_dataset"
        
        # OpenAI API settings for LLM-as-Judge
        self.OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective option
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Generation settings - Matching your training config
        self.MAX_NEW_TOKENS = 512
        self.TEMPERATURE = 0.1
        self.TOP_P = 0.9
        self.DO_SAMPLE = True
        self.MAX_SEQ_LENGTH = 2048
        
        # Quantization settings - Matching your training script
        self.USE_4BIT = True
        self.BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
        self.BNB_4BIT_QUANT_TYPE = "nf4"
        self.BNB_4BIT_USE_DOUBLE_QUANT = True
        
        # Evaluation settings
        self.MAX_SAMPLES_PER_SPECIALTY = None  # Set to None for all samples, or a number for testing
        self.OUTPUT_DIR = "./evaluation_results"
        
        # System prompt - Matching your training script exactly
        self.SYSTEM_PROMPT = (
            "You are a helpful medical AI assistant specializing in providing accurate, "
            "evidence-based answers to medical questions. Provide clear, comprehensive, "
            "and scientifically-grounded responses while maintaining appropriate medical "
            "disclaimers when necessary."
        )

class LLMJudge:
    """LLM-as-Judge evaluator using OpenAI GPT models."""
    
    def __init__(self, config: MedicalQAEvaluationConfig):
        self.config = config
        if config.OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.client = None
            print("Warning: No OpenAI API key found. LLM evaluation will be skipped.")
        
    def create_evaluation_prompt(self, question: str, answer: str, specialty: str, criterion: str) -> str:
        """Create evaluation prompt for specific criterion."""
        
        prompts = {
            "factual_accuracy": f"""
You are a medical expert evaluator. Please assess the FACTUAL ACCURACY of the following medical answer.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Evaluate the factual accuracy on a scale of 1-5, where:
1 = Completely inaccurate, contains dangerous misinformation
2 = Mostly inaccurate, significant factual errors
3 = Partially accurate, some factual errors
4 = Mostly accurate, minor factual issues
5 = Completely accurate, all medical facts are correct

Consider:
- Are the medical facts stated correctly?
- Are treatments/recommendations evidence-based?
- Are there any dangerous or harmful recommendations?
- Is the information consistent with current medical knowledge?

Respond with only a number (1-5) and a brief explanation (1-2 sentences).
Format: "Score: X. Explanation: ..."
""",
            
            "fluency": f"""
You are a language quality evaluator. Please assess the FLUENCY of the following medical answer.

Question: {question}
Answer: {answer}

Evaluate the fluency on a scale of 1-5, where:
1 = Very poor grammar, unclear, difficult to understand
2 = Poor grammar, somewhat unclear
3 = Acceptable grammar, mostly clear
4 = Good grammar, clear and professional
5 = Excellent grammar, very clear and well-structured

Consider:
- Grammar and sentence structure
- Clarity and coherence
- Professional tone appropriate for medical context
- Overall readability

Respond with only a number (1-5) and a brief explanation (1-2 sentences).
Format: "Score: X. Explanation: ..."
""",
            
            "specialty_relevance": f"""
You are a medical specialty expert. Please assess how well the answer stays within the domain of {specialty}.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Evaluate the specialty relevance on a scale of 1-5, where:
1 = Completely off-topic, no relevance to {specialty}
2 = Mostly irrelevant, minimal connection to {specialty}
3 = Somewhat relevant, touches on {specialty} but goes off-topic
4 = Mostly relevant, stays focused on {specialty} with minor deviations
5 = Completely relevant, perfectly focused on {specialty}

Consider:
- Does the answer use appropriate terminology for {specialty}?
- Are the recommendations specific to {specialty} practice?
- Does it avoid irrelevant medical information from other specialties?

Respond with only a number (1-5) and a brief explanation (1-2 sentences).
Format: "Score: X. Explanation: ..."
""",
            
            "doctor_recommendation": f"""
You are a medical practice evaluator. Please assess the DOCTOR RECOMMENDATION in the following medical answer.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Evaluate the doctor recommendation on a scale of 1-5, where:
1 = No doctor recommendation when clearly needed, or completely wrong specialist
2 = Poor doctor recommendation, somewhat inappropriate specialist
3 = Basic doctor recommendation, generic but acceptable
4 = Good doctor recommendation, appropriate specialist mentioned
5 = Excellent doctor recommendation, specific and highly appropriate specialist

Consider:
- Is there a recommendation to see a doctor when appropriate?
- Is the recommended doctor type suitable for the condition/question?
- Is the recommendation specific enough (e.g., "cardiologist" vs "doctor")?
- Does it match the medical specialty context?

For {specialty} questions, appropriate recommendations might include specialists in this field.

Respond with only a number (1-5) and a brief explanation (1-2 sentences).
Format: "Score: X. Explanation: ..."
"""
        }
        
        return prompts[criterion]
    
    def evaluate_answer(self, question: str, answer: str, specialty: str) -> Dict[str, float]:
        """Evaluate a single answer across all criteria."""
        if not self.client:
            # Return default scores if no API key
            return {
                "factual_accuracy": 3.0,
                "fluency": 3.0,
                "specialty_relevance": 3.0,
                "doctor_recommendation": 3.0,
                "factual_accuracy_explanation": "No API key provided",
                "fluency_explanation": "No API key provided",
                "specialty_relevance_explanation": "No API key provided",
                "doctor_recommendation_explanation": "No API key provided"
            }
        
        criteria = ["factual_accuracy", "fluency", "specialty_relevance", "doctor_recommendation"]
        scores = {}
        
        for criterion in criteria:
            try:
                prompt = self.create_evaluation_prompt(question, answer, specialty, criterion)
                
                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                
                result = response.choices[0].message.content.strip()
                
                # Extract score
                score_match = re.search(r"Score:\s*(\d+)", result)
                if score_match:
                    score = int(score_match.group(1))
                    scores[criterion] = score
                    scores[f"{criterion}_explanation"] = result
                else:
                    scores[criterion] = 3.0  # Default score if parsing fails
                    scores[f"{criterion}_explanation"] = "Parsing error"
                    
            except Exception as e:
                print(f"Error evaluating {criterion}: {e}")
                scores[criterion] = 3.0  # Default score on error
                scores[f"{criterion}_explanation"] = f"Error: {str(e)}"
        
        return scores

class MedicalQAEvaluator:
    """Main evaluation class for medical QA models."""
    
    def __init__(self, config: MedicalQAEvaluationConfig):
        self.config = config
        self.judge = LLMJudge(config)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
    def setup_quantization_config(self):
        """Setup quantization config matching training script."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.USE_4BIT,
            bnb_4bit_quant_type=self.config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=self.config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=self.config.BNB_4BIT_USE_DOUBLE_QUANT,
        )
        
    def load_models(self):
        """Load base and fine-tuned models exactly as in training script."""
        print("Loading base model...")
        
        # Setup quantization
        bnb_config = self.setup_quantization_config()
        
        # Load tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.config.BASE_MODEL_PATH)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
            
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=self.config.BNB_4BIT_COMPUTE_DTYPE,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="eager",
        )
        
        print("Loading fine-tuned model...")
        # Load fine-tuned model as PEFT model
        self.ft_model = PeftModel.from_pretrained(self.base_model, self.config.FINETUNED_MODEL_PATH)
        self.ft_model = self.ft_model.merge_and_unload()
        self.ft_tokenizer = self.base_tokenizer  # Same tokenizer
        
        # Set models to eval mode
        self.base_model.eval()
        self.ft_model.eval()
        
        print("Models loaded successfully!")
        
    def load_dataset(self):
        """Load the holdout evaluation dataset."""
        print(f"Loading dataset from {self.config.DATASET_PATH}...")
        
        try:
            # Load using HuggingFace datasets
            self.dataset = load_from_disk(self.config.DATASET_PATH)
            print(f"Loaded {len(self.dataset)} samples")
            
            # Print dataset structure
            if len(self.dataset) > 0:
                sample = self.dataset[0]
                print("Dataset fields:", list(sample.keys()))
                print("Sample specialty:", sample.get('specialty', 'N/A'))
                print("Sample question:", sample.get('question', 'N/A')[:100] + "...")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating dummy dataset for testing...")
            self.dataset = self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """Create dummy dataset for testing purposes."""
        dummy_data = [
            {
                "specialty": "addiction medicine",
                "question": "recognizing signs of relapse",
                "response": "Understanding the context of recognizing signs of a relapse within addiction medicine is crucial...",
                "formatted_chat": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
            },
            {
                "specialty": "cardiology",
                "question": "symptoms of heart attack",
                "response": "Heart attack symptoms include chest pain, shortness of breath...",
                "formatted_chat": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
            }
        ]
        return Dataset.from_list(dummy_data)
    
    def generate_response(self, model, tokenizer, question: str) -> str:
        """Generate response for a given question using the exact same format as training."""
        # Create prompt exactly as in training script
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.config.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                do_sample=self.config.DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response exactly as in training
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def evaluate_single_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample with both models."""
        question = sample['question']
        reference = sample['response']
        specialty = sample['specialty']
        
        # Generate responses
        print(f"Generating responses for: {question[:50]}...")
        base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
        ft_response = self.generate_response(self.ft_model, self.ft_tokenizer, question)
        
        # Evaluate with LLM judge
        print("Evaluating with LLM judge...")
        base_scores = self.judge.evaluate_answer(question, base_response, specialty)
        ft_scores = self.judge.evaluate_answer(question, ft_response, specialty)
        
        result = {
            'question': question,
            'reference': reference,
            'specialty': specialty,
            'base_response': base_response,
            'ft_response': ft_response,
            **{f'base_{k}': v for k, v in base_scores.items()},
            **{f'ft_{k}': v for k, v in ft_scores.items()}
        }
        
        return result
    
    def run_evaluation(self):
        """Run complete evaluation."""
        self.load_models()
        self.load_dataset()
        
        results = []
        specialty_counts = defaultdict(int)
        
        print("Starting evaluation...")
        
        # Filter dataset if needed
        evaluation_samples = []
        for sample in self.dataset:
            specialty = sample['specialty']
            
            # Limit samples per specialty if specified
            if (self.config.MAX_SAMPLES_PER_SPECIALTY and 
                specialty_counts[specialty] >= self.config.MAX_SAMPLES_PER_SPECIALTY):
                continue
                
            evaluation_samples.append(sample)
            specialty_counts[specialty] += 1
        
        print(f"Evaluating {len(evaluation_samples)} samples across {len(specialty_counts)} specialties")
        
        for sample in tqdm(evaluation_samples, desc="Evaluating samples"):
            result = self.evaluate_single_sample(sample)
            results.append(result)
        
        self.results = results
        self.save_results()
        self.analyze_results()
        
    def save_results(self):
        """Save detailed results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.config.OUTPUT_DIR, f"evaluation_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.config.OUTPUT_DIR, f"evaluation_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {json_path}")
    
    def analyze_results(self):
        """Analyze and visualize results."""
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        criteria = ["factual_accuracy", "fluency", "specialty_relevance", "doctor_recommendation"]
        
        print("\n" + "="*50)
        print("MEDICAL QA MODEL EVALUATION SUMMARY")
        print("="*50)
        
        # Overall comparison
        print("\nOVERALL PERFORMANCE COMPARISON:")
        print("-" * 40)
        total_improvement = 0
        for criterion in criteria:
            base_mean = df[f'base_{criterion}'].mean()
            ft_mean = df[f'ft_{criterion}'].mean()
            improvement = ft_mean - base_mean
            total_improvement += improvement
            
            print(f"{criterion.replace('_', ' ').title():<25}: "
                  f"Base={base_mean:.3f} | FT={ft_mean:.3f} | "
                  f"Δ={improvement:+.3f}")
        
        avg_improvement = total_improvement / len(criteria)
        print(f"{'Average Improvement':<25}: {avg_improvement:+.3f}")
        
        # Specialty-wise analysis
        print(f"\nSPECIALTY-WISE ANALYSIS:")
        print("-" * 40)
        specialty_analysis = {}
        
        for specialty in sorted(df['specialty'].unique()):
            specialty_df = df[df['specialty'] == specialty]
            specialty_stats = {}
            
            print(f"\n{specialty.replace('_', ' ').title()} (n={len(specialty_df)}):")
            specialty_improvement = 0
            
            for criterion in criteria:
                base_mean = specialty_df[f'base_{criterion}'].mean()
                ft_mean = specialty_df[f'ft_{criterion}'].mean()
                improvement = ft_mean - base_mean
                specialty_improvement += improvement
                
                specialty_stats[criterion] = {
                    'base': base_mean,
                    'fine_tuned': ft_mean,
                    'improvement': improvement
                }
                
                print(f"  {criterion:<20}: {base_mean:.2f} → {ft_mean:.2f} ({improvement:+.2f})")
            
            avg_specialty_improvement = specialty_improvement / len(criteria)
            specialty_stats['average_improvement'] = avg_specialty_improvement
            specialty_analysis[specialty] = specialty_stats
            print(f"  {'Avg Improvement':<20}: {avg_specialty_improvement:+.2f}")
        
        # Statistical significance (basic)
        print(f"\nSTATISTICAL SUMMARY:")
        print("-" * 40)
        for criterion in criteria:
            base_scores = df[f'base_{criterion}']
            ft_scores = df[f'ft_{criterion}']
            
            # Win rate (how often fine-tuned beats base)
            wins = (ft_scores > base_scores).sum()
            ties = (ft_scores == base_scores).sum()
            losses = (ft_scores < base_scores).sum()
            win_rate = wins / len(df) * 100
            
            print(f"{criterion}: Win Rate = {win_rate:.1f}% ({wins}W/{ties}T/{losses}L)")
        
        # Save detailed analysis
        analysis_path = os.path.join(self.config.OUTPUT_DIR, "detailed_specialty_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(specialty_analysis, f, indent=2, default=str)
        print(f"\nDetailed analysis saved to {analysis_path}")
        
        # Create visualizations
        self.create_visualizations(df)
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create evaluation visualizations."""
        criteria = ["factual_accuracy", "fluency", "specialty_relevance", "doctor_recommendation"]
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Medical QA Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Overall performance comparison
        ax = axes[0, 0]
        base_scores = [df[f'base_{c}'].mean() for c in criteria]
        ft_scores = [df[f'ft_{c}'].mean() for c in criteria]
        
        x = np.arange(len(criteria))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, ft_scores, width, label='Fine-tuned Model', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Evaluation Criteria')
        ax.set_ylabel('Average Score (1-5)')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Improvement by specialty
        ax = axes[0, 1]
        specialty_improvements = {}
        for specialty in df['specialty'].unique():
            specialty_df = df[df['specialty'] == specialty]
            avg_improvement = np.mean([
                specialty_df[f'ft_{c}'].mean() - specialty_df[f'base_{c}'].mean() 
                for c in criteria
            ])
            specialty_improvements[specialty] = avg_improvement
        
        # Sort by improvement
        sorted_specialties = sorted(specialty_improvements.items(), key=lambda x: x[1], reverse=True)
        specialties = [s[0].replace('_', ' ').title() for s in sorted_specialties]
        improvements = [s[1] for s in sorted_specialties]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(specialties, improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Average Improvement Score')
        ax.set_title('Average Improvement by Specialty')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            ax.text(val + (0.02 if val >= 0 else -0.02), i, f'{val:+.2f}', 
                   ha='left' if val >= 0 else 'right', va='center', fontsize=8)
        
        # 3. Score distributions
        ax = axes[1, 0]
        all_base_scores = []
        all_ft_scores = []
        for c in criteria:
            all_base_scores.extend(df[f'base_{c}'].tolist())
            all_ft_scores.extend(df[f'ft_{c}'].tolist())
        
        ax.hist(all_base_scores, bins=15, alpha=0.7, label='Base Model', density=True, color='skyblue')
        ax.hist(all_ft_scores, bins=15, alpha=0.7, label='Fine-tuned Model', density=True, color='lightcoral')
        ax.set_xlabel('Score (1-5)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Criteria-specific improvements
        ax = axes[1, 1]
        improvements_by_criteria = []
        for c in criteria:
            improvement = df[f'ft_{c}'].mean() - df[f'base_{c}'].mean()
            improvements_by_criteria.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements_by_criteria]
        bars = ax.bar(range(len(criteria)), improvements_by_criteria, color=colors, alpha=0.7)
        ax.set_xlabel('Evaluation Criteria')
        ax.set_ylabel('Improvement Score')
        ax.set_title('Improvement by Criteria')
        ax.set_xticks(range(len(criteria)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, improvements_by_criteria):
            ax.text(bar.get_x() + bar.get_width()/2., val + (0.01 if val >= 0 else -0.01),
                   f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.OUTPUT_DIR, 'comprehensive_evaluation_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {plot_path}")

def main():
    """Main evaluation function."""
    print("="*60)
    print("MEDICAL QA MODEL EVALUATION FRAMEWORK")
    print("="*60)
    
    # Initialize configuration
    config = MedicalQAEvaluationConfig()
    
    # Print configuration
    print(f"Base Model Path: {config.BASE_MODEL_PATH}")
    print(f"Fine-tuned Model Path: {config.FINETUNED_MODEL_PATH}")
    print(f"Dataset Path: {config.DATASET_PATH}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    
    # Check API key
    if not config.OPENAI_API_KEY:
        print("\nWarning: OPENAI_API_KEY not found in environment variables.")
        print("LLM-based evaluation will use default scores.")
        print("To enable LLM evaluation, set: export OPENAI_API_KEY='your-api-key-here'")
        
        # Ask user if they want to continue
        response = input("\nContinue with basic evaluation? (y/n): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return
    else:
        print(f"Using OpenAI model: {config.OPENAI_MODEL}")
    
    # Create evaluator and run evaluation
    evaluator = MedicalQAEvaluator(config)
    
    try:
        evaluator.run_evaluation()
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
