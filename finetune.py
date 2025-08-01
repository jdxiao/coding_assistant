import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dataset import load_leetcode_dataset
from prompt import generation_prompt
from llm import LLMInterface

class CodeGeneratorFinetuner: 
    def __init__(self, 
                 model_name: str = "bigcode/starcoderbase-1b",
                 dataset_name_hf: str = "greengerong/leetcode", 
                 output_dir: str = "./lora_adapters",
                 per_device_train_batch_size: int = 1, 
                 gradient_accumulation_steps: int = 8, 
                 warmup_steps: int = 2,
                 max_steps: int = 10, 
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
        
        print(f"Initializing LLMInterface to load base model for fine-tuning...")
        llm_interface_instance = LLMInterface(model_name=model_name, device=self.device)
        self.model = llm_interface_instance.model
        
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
        raw_train_dataset = load_leetcode_dataset(split="train", num_problems=10) 
        if raw_train_dataset is None:
            raise ValueError("Failed to load training dataset. Please check src/dataset.py and your internet connection.")
            
        print("Dataset loaded. Applying preprocessing...")
        
        def preprocess_function(examples):
            # Only handles a single language (Python) for now
            tokenized_examples = []
            
            for i in range(len(examples["content"])):
                problem_content = examples["content"][i]
                
                # Directly check if a Python solution exists for this problem
                if "python" in examples and examples["python"][i]:
                    solution_code = examples["python"][i]
                    instruction_prompt = generation_prompt(problem_content, target_language="python")
                    
                    # Format the full text and tokenize it
                    full_text = f"{instruction_prompt}\n```python\n{solution_code}\n```{self.tokenizer.eos_token}"
                    tokenized_input = self.tokenizer(
                        full_text,
                        max_length=1024,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    # Create labels by cloning the input IDs
                    labels = tokenized_input["input_ids"].clone()
                    
                    # Mask the prompt part so the model only learns from the solution
                    prompt_tokens = self.tokenizer(instruction_prompt, truncation=True)["input_ids"]
                    prompt_len = len(prompt_tokens)
                    labels[:, :prompt_len] = -100
                    
                    # Add the labels to the tokenized input dictionary
                    tokenized_input["labels"] = labels
                    tokenized_examples.append(tokenized_input)
            
            # Combine all tokenized examples into a single dictionary
            if not tokenized_examples:
                return {}
            
            return {
                key: torch.cat([ex[key] for ex in tokenized_examples], dim=0) 
                for key in tokenized_examples[0].keys()
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
    print("--- Starting Code Generator Fine-tuning Script ---")

    trainer_config = {
        "output_dir": "./lora_adapters", 
        "per_device_train_batch_size": 1, 
        "gradient_accumulation_steps": 2, 
        "max_steps": 10, 
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
