import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LLMInterface:
    def __init__(self, model_name, device, lora_adapters_path):
        """
        Initializes the LLM and tokenizer for code generation.
        
        Args:
            model_name (str): The name of the model to load from Hugging Face Hub.
            device (str): The device to load the model on ("cuda" or "cpu").
            lora_adapters_path (str, optional): Path to the LoRA adapters to load.
        """
        self.device = device
        
        # GPU configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        print(f"Loading base model '{model_name}' to device: {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Base model loaded.")
        
        # Load tokenizer
        print(f"Loading tokenizer for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        print("Tokenizer loaded.")
        
        # Load LoRA adapters if they exist
        if lora_adapters_path:
            print(f"Loading LoRA adapters from '{lora_adapters_path}'...")
            self.model = PeftModel.from_pretrained(self.model, lora_adapters_path)
            print("LoRA adapters loaded and merged.")
        
    def generate_solution(self, prompt, max_new_tokens):
        """
        Generates a code solution based on the given prompt.
        
        Args:
            prompt (str): The instruction prompt for the model.
            max_new_tokens (int): The maximum number of new tokens to generate.
            
        Returns:
            str: The generated code solution.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate the text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, # Use sampling for more creative output
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the code part from the generated text
        # Assumes the code is wrapped in ```python ... ```
        start_code = generated_text.find("```python")
        end_code = generated_text.find("```", start_code + 1)
        
        if start_code != -1 and end_code != -1:
            code_solution = generated_text[start_code + len("```python"):end_code].strip()
        else:
            # If the delimiters are not found, return the full text
            code_solution = generated_text
            
        return code_solution
