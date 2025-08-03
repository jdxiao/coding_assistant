import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dataset import load_leetcode_dataset
from prompt import generation_prompt, explanation_prompt
from llm import LLMInterface

class CodeGeneratorFinetuner: 
    def __init__(self, 
                 model_name: str = "bigcode/starcoderbase-1b",
                 dataset_name_hf: str = "greengerong/leetcode", 
                 output_dir: str = "./lora_adapters",
                 per_device_train_batch_size: int = 1, 
                 gradient_accumulation_steps: int = 8, 
                 warmup_steps: int = 2,
                 max_steps: int = 5, 
                 learning_rate: float = 2e-4,
                 fp16: bool = False, 
                 bf16: bool = True,  
                 num_train_epochs: float = 1.0, 
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 push_to_hub: bool = False,
                 target_languages: list = ["python"] # Only fine-tuning on Python
                ):
        
        self.model_name = model_name
        self.dataset_name_hf = dataset_name_hf 
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.push_to_hub = push_to_hub
        self.target_languages = target_languages

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            bf16 = False 
            fp16 = False 
            per_device_train_batch_size = 1 
            gradient_accumulation_steps = 16 

        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.fp16 = fp16
        self.bf16 = bf16
        self.num_train_epochs = num_train_epochs

        print(f"Loading tokenizer for {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("Tokenizer loaded.")
        
        print(f"Initializing LLMInterface to load base model and parse data...")
        self.llm_interface = LLMInterface(model_name=model_name, device=self.device)
        self.model = self.llm_interface.model
        
        self.model.gradient_checkpointing_enable() 
        self.model = prepare_model_for_kbit_training(self.model)
        print("Model prepared for k-bit training (quantization).")

        self.peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM", 
            target_modules=["c_proj", "c_attn", "q_attn"] 
        )
        self.model = get_peft_model(self.model, self.peft_config)
        print("PEFT (LoRA) model configured.")
        self.model.print_trainable_parameters() 

        print(f"Loading dataset from {self.dataset_name_hf} split 'train'...")
        # Limiting to a small number of problems for a quick demo
        raw_train_dataset = load_leetcode_dataset(split="train", num_problems=10) 
        if raw_train_dataset is None:
            raise ValueError("Failed to load training dataset. Please check src/dataset.py and your internet connection.")
            
        print("Dataset loaded. Applying preprocessing...")
        
        def preprocess_function(examples):
            tokenized_examples = []
            
            for i in range(len(examples["content"])):
                problem_content = examples["content"][i]
                
                if "python" in examples and examples["python"][i]:
                    raw_solution_with_explanation = examples["python"][i]
                    
                    # Split the solution and explanation using parsing function
                    solution_code, explanation = self.llm_interface.parse_llm_output(raw_solution_with_explanation)
                    
                    if solution_code and explanation:
                        # Create a training example for code generation
                        instruction_prompt_code = generation_prompt(problem_content, target_language="python")
                        full_text_code = f"{instruction_prompt_code}\n```python\n{solution_code}\n```{self.tokenizer.eos_token}"
                        tokenized_input_code = self.tokenizer(
                            full_text_code,
                            max_length=1024,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        labels_code = tokenized_input_code["input_ids"].clone()
                        prompt_tokens_code = self.tokenizer(instruction_prompt_code, truncation=True)["input_ids"]
                        labels_code[:, :len(prompt_tokens_code)] = -100
                        tokenized_input_code["labels"] = labels_code
                        tokenized_examples.append(tokenized_input_code)

                        # Create a training example for explanation generation
                        instruction_prompt_explanation = explanation_prompt(problem_content, solution_code)
                        full_text_explanation = f"{instruction_prompt_explanation}\n{explanation}\n{self.tokenizer.eos_token}"
                        tokenized_input_explanation = self.tokenizer(
                            full_text_explanation,
                            max_length=1024,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        labels_explanation = tokenized_input_explanation["input_ids"].clone()
                        prompt_tokens_explanation = self.tokenizer(instruction_prompt_explanation, truncation=True)["input_ids"]
                        labels_explanation[:, :len(prompt_tokens_explanation)] = -100
                        tokenized_input_explanation["labels"] = labels_explanation
                        tokenized_examples.append(tokenized_input_explanation)
            
            # Combine all tokenized examples into a single dictionary
            if not tokenized_examples:
                return {}
            
            combined_dict = {key: [] for key in tokenized_examples[0].keys()}
            for example in tokenized_examples:
                for key in combined_dict:
                    combined_dict[key].append(example[key])
            
            return {
                key: torch.cat(combined_dict[key], dim=0) 
                for key in combined_dict.keys()
            }

        self.tokenized_train_dataset = raw_train_dataset.map( 
            preprocess_function,
            batched=True,
            remove_columns=raw_train_dataset.column_names 
        )
        print("Dataset preprocessed and tokenized.")

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=10, 
            save_steps=100, 
            push_to_hub=self.push_to_hub,
            save_strategy="steps",
            load_best_model_at_end=False, 
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset, 
            tokenizer=self.tokenizer, 
        )

    def train(self):
        """
        Starts the fine-tuning process.
        """
        print("Starting fine-tuning...")
        self.trainer.train()
        print("Fine-tuning complete. Saving LoRA adapters.")
        self.trainer.save_model(self.output_dir) 
        
        if self.push_to_hub:
            print(f"Pushing LoRA adapters to Hugging Face Hub.")
            self.tokenizer.save_pretrained(self.output_dir) 
            self.trainer.push_to_hub(commit_message="Initial LoRA adapters training")
            print("LoRA adapters pushed to Hub.")


if __name__ == "__main__":
    print("Starting Code Generator Fine-tuning Script")

    trainer_config = {
        "output_dir": "./lora_adapters", 
        "per_device_train_batch_size": 1, 
        "gradient_accumulation_steps": 2, 
        "max_steps": 5, 
        "learning_rate": 2e-4,
        "bf16": True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, 
        "fp16": True if torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()) else False, 
        "push_to_hub": False, 
        "target_languages": ["python"] # Only fine-tuning on Python
    }

    try:
        finetuner = CodeGeneratorFinetuner(**trainer_config)
        finetuner.train()
        print("\nFine-tuning process completed successfully")
    except Exception as e:
        print(f"\nAn error occurred during fine-tuning: {e}")
        print("Please check your configuration, dataset, and GPU memory.")
