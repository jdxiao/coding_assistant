import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dataset import load_leetcode_dataset
from prompt import generation_prompt
from llm import LLMInterface

# Class to fine-tune using LoRA adapters
class CodeGeneratorFinetuner: 
    def __init__(self, 
                 model_name: str = "bigcode/starcoderbase-1b",
                 dataset_name_hf: str = "greengerong/leetcode", 
                 output_dir: str = "./lora_adapters",
                 per_device_train_batch_size: int = 1, 
                 gradient_accumulation_steps: int = 8, 
                 warmup_steps: int = 2,
                 max_steps: int = 10,  # small number for testing
                 learning_rate: float = 2e-4,
                 fp16: bool = False, 
                 bf16: bool = True,  
                 num_train_epochs: float = 1.0, 
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 push_to_hub: bool = False,
                 target_languages: list = ["python", "c++", "java"],
                ):
        
        # Initialize parameters for fine-tuning
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

        # Load the base model and tokenizer
        print(f"Loading tokenizer for {model_name}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("Tokenizer loaded.")
        
        # Load the base model and prepare it for LoRA fine-tuning
        print(f"Initializing LLMInterface to load base model for fine-tuning.")
        llm_interface_instance = LLMInterface(model_name=model_name, device=self.device)
        self.model = llm_interface_instance.model
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable() 
        self.model = prepare_model_for_kbit_training(self.model)
        print("Model prepared for k-bit training.")

        # Configure LoRA adapters
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

        # Load and preprocess the dataset
        print(f"Loading dataset from {self.dataset_name_hf} split 'train'...")
        raw_train_dataset = load_leetcode_dataset(split="train", num_problems=None) 
        if raw_train_dataset is None:
            raise ValueError("Failed to load training dataset. Please check src/dataset.py and your internet connection.")
            
        print("Dataset loaded. Applying preprocessing.")
        
        # Preprocess the dataset to create training examples
        # Each example will be a prompt with the problem description and the solution code
        def preprocess_function(examples):
            full_texts = []
            # Iterate through each problem and its solution
            for i in range(len(examples["content"])):
                problem_content = examples["content"][i]
                # For each target language, generate the instruction prompt and solution code
                for lang in examples and examples[lang][i]:
                    solution_code = examples[lang][i]
                    instruction_prompt = generation_prompt(problem_content, target_language=lang)
                
                full_text = f"{instruction_prompt}\n```{lang}\n{solution_code}\n```{self.tokenizer.eos_token}"
                full_texts.append(full_text)
            
            tokenized_inputs = self.tokenizer(
                full_texts,
                max_length=1024, 
                truncation=True,
                padding="max_length", 
                return_tensors="pt"
            )
            
            labels = tokenized_inputs["input_ids"].clone()
            
            # Ignore prompt tokens in the labels
            for i, problem_content in enumerate(examples["content"]):
                for lang in self.target_languages:
                    if lang in examples and examples[lang][i]:
                        instruction_prompt = generation_prompt(problem_content, target_language=lang)
                        prompt_tokens = self.tokenizer(instruction_prompt, truncation=True)["input_ids"]
                        prompt_len = len(prompt_tokens)
                        labels[i, :prompt_len] = -100 

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        self.tokenized_train_dataset = raw_train_dataset.map( 
            preprocess_function,
            batched=True,
            remove_columns=raw_train_dataset.column_names 
        )
        print("Dataset preprocessed and tokenized.")

        # Set up training arguments
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
            hub_model_id=self.hub_model_id if self.push_to_hub else None,
            report_to="tensorboard", 
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
        This method uses the Trainer API to fine-tune the model on the preprocessed dataset.
        It saves the LoRA adapters after training and optionally pushes them to the Hugging Face Hub
        """
        print("Starting fine-tuning...")
        self.trainer.train()
        print("Fine-tuning complete. Saving LoRA adapters.")
        self.trainer.save_model(self.output_dir) 
        
        if self.push_to_hub:
            print(f"Pushing LoRA adapters to Hugging Face Hub: {self.hub_model_id}")
            self.tokenizer.save_pretrained(self.output_dir) 
            self.trainer.push_to_hub(commit_message="Initial LoRA adapters training")
            print("LoRA adapters pushed to Hub.")


# For file testing purposes
if __name__ == "__main__":
    print("Starting Code Generator Fine-tuning.")

    trainer_config = {
        "output_dir": "./lora_adapters", 
        "per_device_train_batch_size": 1, 
        "gradient_accumulation_steps": 2, 
        "max_steps": 10, 
        "learning_rate": 2e-4,
        "bf16": True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False, 
        "fp16": True if torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()) else False, 
        "push_to_hub": False
    }

    try:
        finetuner = CodeGeneratorFinetuner(**trainer_config)
        finetuner.train()
        print("\nFine-tuning process completed successfully")
    except Exception as e:
        print(f"\nAn error occurred during fine-tuning: {e}")
        print("Please check your configuration, dataset, and GPU memory.")