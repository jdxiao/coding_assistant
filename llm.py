import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LLMInterface:
    def __init__(self, model_name: str, device: str = None, lora_adapters_path: str = None):
        """
        Initializes the LLM and tokenizer for code generation.
        
        Args:
            model_name (str): The name of the model to load from Hugging Face Hub.
            device (str): The device to load the model on ("cuda" or "cpu").
            lora_adapters_path (str, optional): Path to the LoRA adapters to load.
        """

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Configuration for 4-bit quantization
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
        
        # -Load LoRA adapters if provided
        if lora_adapters_path:
            print(f"Loading LoRA adapters from '{lora_adapters_path}'...")
            self.model = PeftModel.from_pretrained(self.model, lora_adapters_path)
            print("LoRA adapters loaded and merged.")

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generates a text response from the loaded LLM based on the given prompt.

        Args:
            prompt (str): The input text prompt for the model.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Controls the randomness of the generation.

        Returns:
            str: The generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def parse_llm_output(self, raw_output: str, target_language: str = "python"):
        """
        Parses the raw LLM output to extract code and explanation based on a dynamic markdown format.

        Args:
            raw_output (str): The raw text output from the LLM.
            target_language (str): The language of the code to look for. Defaults to 'python'.

        Returns:
            Tuple[str, str]: A tuple containing the extracted code and explanation.
        """
        code_start_tag = f"```{target_language}"
        code_end_tag = "```"

        code = ""
        explanation = ""

        code_start_index = raw_output.find(code_start_tag)
        if code_start_index != -1:
            code_content_start = code_start_index + len(code_start_tag)
            code_end_index = raw_output.find(code_end_tag, code_content_start)
            
            if code_end_index != -1:
                code = raw_output[code_content_start:code_end_index].strip()
                explanation = raw_output[code_end_index + len(code_end_tag):].strip()
            else:
                code = raw_output[code_content_start:].strip()
        
        return code, explanation

