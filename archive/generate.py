import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from prompt import generation_prompt

def load_finetuned_model(
    base_model_name = "bigcode/starcoderbase-1b",
    adapter_path  = "./lora_adapters",
    device = "cpu"):

    """
    Loads the base model and applies the LoRA adapters.
    Args:
        base_model_name (str): The name of the pre-trained model to load.
        adapter_path (str): Path to the LoRA adapters.
        device (str): Device to load the model on ("cpu" or "cuda").
    Returns:
        model (PeftModel): The model with LoRA adapters applied.
        tokenizer (AutoTokenizer): The tokenizer for the model.
    """

    print(f"Loading base model: {base_model_name}...")
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float32 for CPU
        load_in_8bit=False,
        trust_remote_code=True
    )
    base_model.eval() # Set to evaluation mode

    # Load the LoRA adapters
    print(f"Loading LoRA adapters from: {adapter_path}.")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge the LoRA adapters into the base model for consistent inference
    print("Merging LoRA adapters into the base model.")
    model = model.merge_and_unload()
    
    print("Fine-tuned model loaded and adapters merged.")
    
    # Load the tokenizer
    print(f"Loading tokenizer for {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded.")

    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    problem_description,
    target_language = "python",  # Python by default
    max_new_tokens = 200,
    temperature = 0.1,
    do_sample = True,
    top_p = 0.95
):
    """
    Generates code using the fine-tuned model.
    Args:
        model (PeftModel): The fine-tuned model with LoRA adapters.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        problem_description (str): The description of the coding problem to solve.
        target_language (str): The programming language for the solution. Default is "python".
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generation.
        do_sample (bool): Whether to use sampling or greedy decoding.
        top_p (float): Top-p sampling parameter.
    Returns:
        str: The generated code solution.
    """

    prompt = generation_prompt(problem_description, target_language)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    device = model.device # Get device from model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("\nGenerating Code...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract the generated code from the output
    generated_code_start_marker = f"```python\n"
    
    # Check if the generated text contains the code block start marker
    if generated_code_start_marker in generated_text:
        # Get the part after the prompt
        generated_code = generated_text.split(generated_code_start_marker)[-1]
        # Remove the ending code block marker if it exists
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()
        elif generated_code.endswith(tokenizer.eos_token): # remove EOS token if present
             generated_code = generated_code.replace(tokenizer.eos_token, "").strip()
    else:
        # Fallback if marker not found, show full generation.
        print("Warning: Could not find code block start marker in generated text. Showing full generation.")
        generated_code = generated_text

    print(f"\nProblem:\n{problem_description}\n")
    print(f"\nGenerated Solution:\n{generated_code}")
    
    return generated_code

# For file testing purposes
if __name__ == "__main__":
    BASE_MODEL_NAME = "bigcode/starcoderbase-1b"
    LORA_ADAPTERS_PATH = "./lora_adapters"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the fine-tuned model and tokenizer
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        base_model_name=BASE_MODEL_NAME,
        adapter_path=LORA_ADAPTERS_PATH,
        device=DEVICE
    )

    # Example problem to generate code for
    test_problem = """
    Write a Python function to reverse a string.

    Input: "hello"
    Output: "olleh"
    """

    generated_solution = generate_code(
        finetuned_model,
        finetuned_tokenizer,
        problem_description=test_problem,
        max_new_tokens=150,
        temperature=0.1
    )