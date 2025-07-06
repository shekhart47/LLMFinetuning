"""
Medical QA Model Evaluation using GPT-4o as Judge
Based on the 4 evaluation criteria: Factual Accuracy, Fluency, Relevance to Specialty, Doctor Recommendation
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings
import requests
from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from azureml.core.authentication import ServicePrincipalAuthentication

warnings.filterwarnings("ignore")

# Core ML libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from peft import PeftModel

# LangChain imports for Azure OpenAI
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser, CommaSeparatedListOutputParser, StructuredOutputParser

class EvaluationConfig:
    """Configuration for medical QA evaluation."""
    
    def __init__(self):
        # Model paths (pass these from main function)
        self.BASE_MODEL_PATH = None
        self.FINETUNED_MODEL_PATH = None
        self.DATASET_PATH = None
        
        # Azure OpenAI settings
        self.AZURE_DEPLOYMENT_NAME = "gpt-4o"  # Your Azure deployment name
        self.USE_AZURE_OPENAI = True
        
        # Model settings (matching training script)
        self.MAX_NEW_TOKENS = 512
        self.TEMPERATURE = 0.1
        self.TOP_P = 0.9
        self.DO_SAMPLE = True
        self.MAX_SEQ_LENGTH = 2048
        
        # Quantization settings (from training script)
        self.USE_4BIT = True
        self.BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
        self.BNB_4BIT_QUANT_TYPE = "nf4"
        self.BNB_4BIT_USE_DOUBLE_QUANT = True
        
        # Evaluation settings
        self.MAX_SAMPLES_PER_SPECIALTY = None  # None for all samples
        self.OUTPUT_DIR = "./evaluation_results"
        
        # System prompt (exact from training script)
        self.SYSTEM_PROMPT = (
            "You are a helpful medical AI assistant specializing in providing accurate, "
            "evidence-based answers to medical questions. Provide clear, comprehensive, "
            "and scientifically-grounded responses while maintaining appropriate medical "
            "disclaimers when necessary."
        )

class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r

def initialize_llm() -> AzureChatOpenAI:
    """Initialize Azure OpenAI model using workspace credentials."""
    ws = Workspace.from_config()
    keyvault = ws.get_default_keyvault()
    credential = DefaultAzureCredential()
    workspacename = keyvault.get_secret("project-workspace-name")
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    os.environ["AZURE_OPENAI_KEY"] = access_token.token
    openai.api_type = "azure_ad"
    os.environ["AZURE_OPENAI_ENDPOINT"] = f"https://{workspacename}openai.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    subscriptionId = keyvault.get_secret("project-subscription-id")
    
    # Set environment variables for Azure OpenAI credentials
    os.environ["AZURE_OPENAI_API_KEY"] = access_token.token
    os.environ["AZURE_OPENAI_API_BASE"] = f"https://{workspacename}openai.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
    
    subscriptionId = keyvault.get_secret("project-subscription-id")
    apiVersion = "2023-10-01-preview"
    url = f"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{workspacename}-common/providers/Microsoft.CognitiveServices/accounts/{workspacename}openai/deployments?api-version={apiVersion}"
    accessToken = credential.get_token("https://management.azure.com/.default")
    response = requests.get(url, auth=BearerAuth(accessToken.token))
    
    print(f'Initializing Model : {os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]}')
    model = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        max_tokens=4000,
        temperature=0.9,
        model_kwargs={"seed": 1337}
    )
    
    print('Model Initialized')
    return model

def get_query_answer(model: AzureChatOpenAI, medical_query: str, medical_specialty: str, system_prompt: str):
    """Get answer using Azure OpenAI with LangChain."""
    
    class SpecialtiesResponse(BaseModel):
        answer: str = Field(description="answer for the medical query provided by the user in context of the medical specialty")
    
    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = StrOutputParser()
    
    full_prompt = f"""
{system_prompt}

medical_specialty : {medical_specialty}
question : {medical_query}
answer : {{medical_answer}}
"""
    
    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=full_prompt
    )
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty": medical_specialty, "medical_query": medical_query, "answer": "medical_answer"})
    return result

class GPTJudge:
    """Azure OpenAI based evaluator for the 4 medical QA criteria."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        try:
            self.model = initialize_llm()
            print("Azure OpenAI model initialized successfully")
        except Exception as e:
            print(f"Error initializing Azure OpenAI: {e}")
            print("Using dummy scores for evaluation")
            self.model = None
    
    def evaluate_factual_accuracy(self, question: str, answer: str, specialty: str) -> Dict[str, any]:
        """Evaluate factual accuracy (1-5 scale)."""
        system_prompt = f"""You are a medical expert evaluator. Rate the FACTUAL ACCURACY of this medical answer.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Rate on scale 1-5:
1 = Completely inaccurate, dangerous misinformation
2 = Mostly inaccurate, significant errors
3 = Partially accurate, some errors
4 = Mostly accurate, minor issues
5 = Completely accurate, evidence-based

Consider: medical correctness, evidence-based recommendations, safety of advice.

Respond ONLY with: "Score: X. Explanation: [1-2 sentences]"
"""
        return self._get_score_azure(system_prompt, "factual_accuracy")
    
    def evaluate_fluency(self, question: str, answer: str) -> Dict[str, any]:
        """Evaluate fluency and readability (1-5 scale)."""
        system_prompt = f"""You are a language quality evaluator. Rate the FLUENCY of this medical answer.

Question: {question}
Answer: {answer}

Rate on scale 1-5:
1 = Very poor grammar, unclear, hard to understand
2 = Poor grammar, somewhat unclear
3 = Acceptable grammar, mostly clear
4 = Good grammar, clear and professional
5 = Excellent grammar, very clear and well-structured

Consider: grammar, clarity, coherence, professional medical tone, readability.

Respond ONLY with: "Score: X. Explanation: [1-2 sentences]"
"""
        return self._get_score_azure(system_prompt, "fluency")
    
    def evaluate_specialty_relevance(self, question: str, answer: str, specialty: str) -> Dict[str, any]:
        """Evaluate relevance to medical specialty (1-5 scale)."""
        system_prompt = f"""You are a medical specialty expert. Rate how well this answer stays within {specialty}.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Rate on scale 1-5:
1 = Completely off-topic, no relevance to {specialty}
2 = Mostly irrelevant, minimal connection
3 = Somewhat relevant, touches specialty but goes off-topic
4 = Mostly relevant, focused on {specialty} with minor deviations
5 = Completely relevant, perfectly focused on {specialty}

Consider: appropriate terminology, specialty-specific knowledge, domain focus.

Respond ONLY with: "Score: X. Explanation: [1-2 sentences]"
"""
        return self._get_score_azure(system_prompt, "specialty_relevance")
    
    def evaluate_doctor_recommendation(self, question: str, answer: str, specialty: str) -> Dict[str, any]:
        """Evaluate appropriateness of doctor recommendation (1-5 scale)."""
        system_prompt = f"""You are a medical practice evaluator. Rate the DOCTOR RECOMMENDATION in this answer.

Medical Specialty: {specialty}
Question: {question}
Answer: {answer}

Rate on scale 1-5:
1 = No doctor recommendation when needed, or completely wrong specialist
2 = Poor recommendation, somewhat inappropriate specialist
3 = Basic recommendation, generic but acceptable
4 = Good recommendation, appropriate specialist mentioned
5 = Excellent recommendation, specific and highly appropriate

Consider: presence of recommendation when needed, appropriateness of specialist type, specificity.
For {specialty} questions, expect recommendations for relevant specialists.

Respond ONLY with: "Score: X. Explanation: [1-2 sentences]"
"""
        return self._get_score_azure(system_prompt, "doctor_recommendation")
    
    def _get_score_azure(self, system_prompt: str, criterion: str) -> Dict[str, any]:
        """Get score using Azure OpenAI."""
        if not self.model:
            return {criterion: 3.0, f"{criterion}_explanation": "No Azure model available"}
        
        try:
            # Use get_query_answer function pattern but adapted for evaluation
            result = get_query_answer(
                model=self.model,
                medical_query="Evaluate and provide score",
                medical_specialty="Evaluation",
                system_prompt=system_prompt
            )
            
            # Extract score from result
            score_match = re.search(r"Score:\s*(\d+)", result)
            if score_match:
                score = int(score_match.group(1))
                return {criterion: score, f"{criterion}_explanation": result}
            else:
                return {criterion: 3.0, f"{criterion}_explanation": "Parsing error"}
                
        except Exception as e:
            print(f"Error evaluating {criterion}: {e}")
            return {criterion: 3.0, f"{criterion}_explanation": f"Error: {str(e)}"}
    
    def evaluate_all_criteria(self, question: str, answer: str, specialty: str) -> Dict[str, any]:
        """Evaluate all 4 criteria for a single answer."""
        results = {}
        
        # Evaluate each criterion
        results.update(self.evaluate_factual_accuracy(question, answer, specialty))
        results.update(self.evaluate_fluency(question, answer))
        results.update(self.evaluate_specialty_relevance(question, answer, specialty))
        results.update(self.evaluate_doctor_recommendation(question, answer, specialty))
        
        return results

class MedicalQAEvaluator:
    """Main evaluator class."""
    
    def __init__(self, base_model_path: str, finetuned_model_path: str, dataset_path: str, 
                 output_dir: str = "./evaluation_results"):
        # Initialize config with paths
        self.config = EvaluationConfig()
        self.config.BASE_MODEL_PATH = base_model_path
        self.config.FINETUNED_MODEL_PATH = finetuned_model_path
        self.config.DATASET_PATH = dataset_path
        self.config.OUTPUT_DIR = output_dir
        
        self.judge = GPTJudge(self.config)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        print(f"Initialized evaluator:")
        print(f"  Base model: {base_model_path}")
        print(f"  Fine-tuned model: {finetuned_model_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Output: {output_dir}")
    
    def setup_quantization_config(self):
        """Setup quantization config matching training script."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.USE_4BIT,
            bnb_4bit_quant_type=self.config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=self.config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=self.config.BNB_4BIT_USE_DOUBLE_QUANT,
        )
    
    def load_models(self):
        """Load both models exactly as in training script."""
        print("Loading models...")
        
        # Setup quantization
        bnb_config = self.setup_quantization_config()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BASE_MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=self.config.BNB_4BIT_COMPUTE_DTYPE,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="eager",
        )
        
        # Load fine-tuned model as PEFT
        print("Loading fine-tuned model...")
        self.ft_model = PeftModel.from_pretrained(self.base_model, self.config.FINETUNED_MODEL_PATH)
        self.ft_model = self.ft_model.merge_and_unload()
        
        # Set to eval mode
        self.base_model.eval()
        self.ft_model.eval()
        
        print("Models loaded successfully!")
    
    def load_dataset(self):
        """Load holdout dataset."""
        print(f"Loading dataset from {self.config.DATASET_PATH}...")
        
        try:
            self.dataset = load_from_disk(self.config.DATASET_PATH)
            print(f"Loaded {len(self.dataset)} samples")
            
            # Show sample
            if len(self.dataset) > 0:
                sample = self.dataset[0]
                print("Dataset fields:", list(sample.keys()))
                print(f"Sample specialty: {sample.get('specialty', 'N/A')}")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def generate_response(self, model, question: str) -> str:
        """Generate response using exact training format."""
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.config.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(
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
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample."""
        question = sample['question']
        reference = sample['response']
        specialty = sample['specialty']
        
        print(f"Evaluating: {question[:60]}...")
        
        # Generate responses
        base_response = self.generate_response(self.base_model, question)
        ft_response = self.generate_response(self.ft_model, question)
        
        # Evaluate with GPT judge
        base_scores = self.judge.evaluate_all_criteria(question, base_response, specialty)
        ft_scores = self.judge.evaluate_all_criteria(question, ft_response, specialty)
        
        # Combine results
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
        
        print("\nStarting evaluation...")
        
        # Filter samples if needed
        evaluation_samples = []
        for sample in self.dataset:
            specialty = sample['specialty']
            
            if (self.config.MAX_SAMPLES_PER_SPECIALTY and 
                specialty_counts[specialty] >= self.config.MAX_SAMPLES_PER_SPECIALTY):
                continue
                
            evaluation_samples.append(sample)
            specialty_counts[specialty] += 1
        
        print(f"Evaluating {len(evaluation_samples)} samples")
        
        # Evaluate each sample
        for sample in tqdm(evaluation_samples, desc="Evaluating"):
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # Save and analyze results
        self.save_results(results)
        self.analyze_results(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV (structured format as requested)
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        cols = ['specialty', 'question', 'base_response', 'ft_response', 'reference']
        score_cols = [col for col in df.columns if col.endswith(('_accuracy', '_fluency', '_relevance', '_recommendation')) and not col.endswith('_explanation')]
        explanation_cols = [col for col in df.columns if col.endswith('_explanation')]
        
        final_cols = cols + sorted(score_cols) + sorted(explanation_cols)
        df = df[[col for col in final_cols if col in df.columns]]
        
        csv_path = os.path.join(self.config.OUTPUT_DIR, f"medical_qa_evaluation_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.config.OUTPUT_DIR, f"medical_qa_evaluation_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")
    
    def analyze_results(self, results: List[Dict]):
        """Analyze results and create summary."""
        df = pd.DataFrame(results)
        criteria = ["factual_accuracy", "fluency", "specialty_relevance", "doctor_recommendation"]
        
        print("\n" + "="*60)
        print("MEDICAL QA EVALUATION RESULTS")
        print("="*60)
        
        # Overall comparison
        print("\nOVERALL PERFORMANCE:")
        print("-" * 40)
        
        overall_stats = {}
        total_improvement = 0
        
        for criterion in criteria:
            base_mean = df[f'base_{criterion}'].mean()
            ft_mean = df[f'ft_{criterion}'].mean()
            improvement = ft_mean - base_mean
            total_improvement += improvement
            
            # Win/loss analysis
            wins = (df[f'ft_{criterion}'] > df[f'base_{criterion}']).sum()
            ties = (df[f'ft_{criterion}'] == df[f'base_{criterion}']).sum()
            losses = (df[f'ft_{criterion}'] < df[f'base_{criterion}']).sum()
            win_rate = wins / len(df) * 100
            
            overall_stats[criterion] = {
                'base_mean': base_mean,
                'ft_mean': ft_mean,
                'improvement': improvement,
                'win_rate': win_rate,
                'wins': wins,
                'ties': ties,
                'losses': losses
            }
            
            print(f"{criterion.replace('_', ' ').title():<25}:")
            print(f"  Base Model:      {base_mean:.3f}")
            print(f"  Fine-tuned:      {ft_mean:.3f}")
            print(f"  Improvement:     {improvement:+.3f}")
            print(f"  Win Rate:        {win_rate:.1f}% ({wins}W/{ties}T/{losses}L)")
            print()
        
        avg_improvement = total_improvement / len(criteria)
        print(f"AVERAGE IMPROVEMENT: {avg_improvement:+.3f}")
        
        # Specialty-wise macro-averaged analysis
        print(f"\nSPECIALTY-WISE ANALYSIS (Macro-Averaged):")
        print("-" * 50)
        
        specialty_stats = {}
        all_specialty_improvements = []
        
        for specialty in sorted(df['specialty'].unique()):
            specialty_df = df[df['specialty'] == specialty]
            
            specialty_improvements = []
            specialty_data = {'sample_count': len(specialty_df)}
            
            print(f"\n{specialty.replace('_', ' ').title()} (n={len(specialty_df)}):")
            
            for criterion in criteria:
                base_mean = specialty_df[f'base_{criterion}'].mean()
                ft_mean = specialty_df[f'ft_{criterion}'].mean()
                improvement = ft_mean - base_mean
                specialty_improvements.append(improvement)
                
                specialty_data[f'{criterion}_improvement'] = improvement
                print(f"  {criterion:<20}: {base_mean:.2f} → {ft_mean:.2f} ({improvement:+.2f})")
            
            avg_specialty_improvement = np.mean(specialty_improvements)
            specialty_data['avg_improvement'] = avg_specialty_improvement
            specialty_stats[specialty] = specialty_data
            all_specialty_improvements.append(avg_specialty_improvement)
            
            print(f"  {'Average':<20}: {avg_specialty_improvement:+.2f}")
        
        # Macro-averaged improvement across specialties
        macro_avg_improvement = np.mean(all_specialty_improvements)
        print(f"\nMACRO-AVERAGED IMPROVEMENT ACROSS SPECIALTIES: {macro_avg_improvement:+.3f}")
        
        # Save analysis
        analysis = {
            'overall_stats': overall_stats,
            'specialty_stats': specialty_stats,
            'macro_avg_improvement': macro_avg_improvement,
            'evaluation_summary': {
                'total_samples': len(df),
                'num_specialties': len(df['specialty'].unique()),
                'avg_improvement': avg_improvement,
                'macro_avg_improvement': macro_avg_improvement
            }
        }
        
        analysis_path = os.path.join(self.config.OUTPUT_DIR, "evaluation_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nDetailed analysis saved to: {analysis_path}")
        
        # Create visualization
        self.create_visualization(df, criteria)
        
        return analysis
    
    def create_visualization(self, df: pd.DataFrame, criteria: List[str]):
        """Create evaluation visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Medical QA Model Evaluation: Base vs Fine-tuned', fontsize=16, fontweight='bold')
        
        # 1. Overall comparison
        ax = axes[0, 0]
        base_scores = [df[f'base_{c}'].mean() for c in criteria]
        ft_scores = [df[f'ft_{c}'].mean() for c in criteria]
        
        x = np.arange(len(criteria))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x + width/2, ft_scores, width, label='Fine-tuned Model', alpha=0.8, color='orange')
        
        ax.set_ylabel('Average Score (1-5)')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Win rates
        ax = axes[0, 1]
        win_rates = []
        for c in criteria:
            wins = (df[f'ft_{c}'] > df[f'base_{c}']).sum()
            win_rate = wins / len(df) * 100
            win_rates.append(win_rate)
        
        bars = ax.bar(range(len(criteria)), win_rates, color='green', alpha=0.7)
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Fine-tuned Model Win Rate by Criteria')
        ax.set_xticks(range(len(criteria)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], rotation=45, ha='right')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2., val + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Improvement by specialty
        ax = axes[1, 0]
        specialty_improvements = {}
        for specialty in df['specialty'].unique():
            specialty_df = df[df['specialty'] == specialty]
            avg_improvement = np.mean([
                specialty_df[f'ft_{c}'].mean() - specialty_df[f'base_{c}'].mean() 
                for c in criteria
            ])
            specialty_improvements[specialty] = avg_improvement
        
        sorted_specialties = sorted(specialty_improvements.items(), key=lambda x: x[1])
        specialties = [s[0].replace('_', ' ').title() for s in sorted_specialties]
        improvements = [s[1] for s in sorted_specialties]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(range(len(specialties)), improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Average Improvement')
        ax.set_title('Average Improvement by Specialty')
        ax.set_yticks(range(len(specialties)))
        ax.set_yticklabels(specialties)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Score distribution
        ax = axes[1, 1]
        all_base = []
        all_ft = []
        for c in criteria:
            all_base.extend(df[f'base_{c}'].tolist())
            all_ft.extend(df[f'ft_{c}'].tolist())
        
        ax.hist(all_base, bins=15, alpha=0.7, label='Base Model', density=True, color='lightblue')
        ax.hist(all_ft, bins=15, alpha=0.7, label='Fine-tuned Model', density=True, color='orange')
        ax.set_xlabel('Score (1-5)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.OUTPUT_DIR, 'evaluation_results_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")

def main(base_model_path: str, finetuned_model_path: str, dataset_path: str, 
         output_dir: str = "./evaluation_results", max_samples: Optional[int] = None):
    """
    Main evaluation function.
    
    Args:
        base_model_path: Path to base model (e.g., "./llama-3-meerkat-8b-v1.0")
        finetuned_model_path: Path to fine-tuned model (e.g., "./results")
        dataset_path: Path to holdout dataset (e.g., "../../datasets/final_query_set/final_dataset/holdout_dataset")
        output_dir: Output directory for results
        max_samples: Maximum samples per specialty (None for all)
    """
    print("="*60)
    print("MEDICAL QA MODEL EVALUATION WITH AZURE OPENAI JUDGE")
    print("="*60)
    
    # Check Azure workspace configuration
    try:
        ws = Workspace.from_config()
        print(f"Using Azure ML workspace: {ws.name}")
    except Exception as e:
        print(f"Warning: Could not connect to Azure ML workspace: {e}")
        print("Make sure you have a valid config.json file in your working directory")
        print("Continuing with dummy scores...")
    
    # Initialize evaluator
    evaluator = MedicalQAEvaluator(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Set max samples if specified
    if max_samples:
        evaluator.config.MAX_SAMPLES_PER_SPECIALTY = max_samples
        print(f"Limiting to {max_samples} samples per specialty")
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage with your paths
    results = main(
        base_model_path="./llama-3-meerkat-8b-v1.0",
        finetuned_model_path="./results", 
        dataset_path="../../datasets/final_query_set/final_dataset/holdout_dataset",
        output_dir="./evaluation_results",
        max_samples=None  # Set to a number for testing, None for all samples
    )
