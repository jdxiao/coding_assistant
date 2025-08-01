from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# LLMInterface class to load and interact with a pre-trained language model
class LLMInterface:

    # Initializes the LLM interface with a pre-trained model and tokenizer.
    def __init__(self, model_name: str = "bigcode/starcoderbase-1b", device: str = None):
        """
        Initializes the LLM interface with a pre-trained model and tokenizer.
        Args:
            model_name (str): The name of the pre-trained model to load.
            device (str): The device to load the model on. If None, it will use
        """
        self.model_name = model_name

        # Device selection
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Load the model and tokenizer
        print(f"Loading tokenizer for '{model_name}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")

        print(f"Loading model '{model_name}' to device: {self.device}...")
    
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(self.device)
 
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Set tokenizer.pad_token_id to tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")

    # Generates a prompt for code generation based on the problem description and target language.
    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7):
        """
        Generates a text response from the loaded LLM based on the given prompt.
        Args:
            prompt (str): The input prompt to generate a response for.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The temperature for sampling. Higher values mean more randomness.
        Returns:
            str: The generated text response from the model.
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
        
        # Remove the prompt part from the generated text, if it exists
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text


    def parse_llm_output(self, raw_output: str, target_language: str = "python"):
        """
        Parses the raw LLM output to extract code and explanation based on expected markdown format.
        Args:
            raw_output (str): The raw output text from the LLM.
            target_language (str): The programming language of the solution code. Default is "python".
        Returns:
            tuple: A tuple containing the extracted code and explanation.
        """

        code_start_tag = f"```{target_language}"  # Python by default
        code_end_tag = "```"

        code = ""
        explanation = ""

        # Check if the raw output contains the code block
        code_start_index = raw_output.find(code_start_tag)

        # If the code block is found, extract the code and explanation
        if code_start_index != -1:
            code_content_start = code_start_index + len(code_start_tag)
            code_end_index = raw_output.find(code_end_tag, code_content_start)
            
            if code_end_index != -1:
                code = raw_output[code_content_start:code_end_index].strip()
                explanation = raw_output[code_end_index + len(code_end_tag):].strip()
            else:
                code = raw_output[code_content_start:].strip()
        
        return code, explanation

# For file testing purposes
if __name__ == "__main__":
    try:
        from prompt import generation_prompt
    except ImportError:
        print("\nERROR: Could not import 'generation_prompt' from 'prompt.py'.")
        exit()

    print("Testing LLMInterface (Base Model Loading and Basic Generation)")

    try:
        llm_agent = LLMInterface() # Uses default StarCoder model (bigcode/starcoderbase-1b)
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