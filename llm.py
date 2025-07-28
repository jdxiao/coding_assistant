from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMInterface:
    def __init__(self, model_name: str = "bigcode/starcoderbase-7b", device: str = None):
        """
        Initializes the LLM interface with a pre-trained model and tokenizer.
        """
        self.model_name = model_name

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading tokenizer for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")

        print(f"Loading model '{model_name}' to device: {self.device}...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        except RuntimeError as e:
            if "device-side assert" in str(e) or "Expected bfloat16" in str(e):
                print("\nWARNING: bfloat16 might not be supported by your GPU or PyTorch setup.")
                print("Attempting to load in float16 (half precision) instead.")
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
            else:
                raise e
        
        print("Model loaded.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Set tokenizer.pad_token_id to tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generates a text response from the loaded LLM based on the given prompt.
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

    def parse_llm_output(self, raw_output: str) -> tuple[str, str]:
        """
        Parses the raw LLM output to extract code and explanation based on expected markdown format.
        """
        code_start_tag = "```python" # Assuming Python as the target language for code blocks
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

if __name__ == "__main__":
    try:
        from prompt import generation_prompt
    except ImportError:
        print("\nERROR: Could not import 'generation_prompt' from 'prompt.py'.")
        exit()

    print("Testing LLMInterface (Base Model Loading and Basic Generation)")

    try:
        llm_agent = LLMInterface() # Uses default StarCoder model (bigcode/starcoderbase-7b)
    except Exception as e:
        print(f"\nFailed to initialize LLMInterface: {e}")
        exit()

    sample_problem_content = """
    Write a Python function `multiply_numbers(a, b)` that takes two numbers as input and returns their product.

    Example:
    Input: a = 5, b = 3
    Output: 15
    """

    code_gen_prompt = generation_prompt(sample_problem_content, target_language="python")
    print("\nGenerated Prompt for LLM")
    print(code_gen_prompt)

    print("\nGenerating response from LLM (this will take a moment)")
    raw_llm_output = llm_agent.generate_response(code_gen_prompt, max_new_tokens=150, temperature=0.2)
    print("\nRaw LLM Output (from Base Model)")
    print(raw_llm_output)

    generated_code, generated_explanation = llm_agent.parse_llm_output(raw_llm_output)
    print("\nParsed Code")
    print(generated_code)
    print("\nParsed Explanation (if any)")
    print(generated_explanation if generated_explanation else "[No explicit explanation parsed]")

    if "def multiply_numbers" in generated_code and "return" in generated_code:
        print("\nBasic check: 'def multiply_numbers' and 'return' found in generated code. Seems reasonable.")
    else:
        print("\nBasic check: Code structure might not be as expected. Parsing or model output needs review.")

    print("\nLLMInterface Test Complete")