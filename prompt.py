def generation_prompt(problem_content: str, target_language: str = "python") -> str:
    """
    Generates a prompt string that structures the coding dataset for fine-tuning the LLM.
    """
    prompt = f"Given the following algorithmic problem:\n\n" \
             f"```problem\n{problem_content}\n```\n\n" \
             f"Provide a complete and correct solution in {target_language}. " \
             f"Your response should only contain the code solution, enclosed within a " \
             f"markdown code block. Do not include any additional explanations, comments " \
             f"outside the code block, or conversational text. Ensure the code is ready to be executed." \
             f"\n\nFor example, if the problem asks for a function 'add(a, b)', " \
             f"your response should start with:\n\n" \
             f"```{target_language}\ndef add(a, b):\n    # your code here\n```"

    return prompt

def explanation_prompt(problem_content: str, solution_code: str, target_language: str = "python") -> str:
    """
    Generates a prompt for an LLM to explain a given coding problem and its solution.
    """
    prompt = f"Given the following algorithmic problem:\n\n" \
             f"```problem\n{problem_content}\n```\n\n" \
             f"And the following solution code in {target_language}:\n\n" \
             f"```{target_language}\n{solution_code}\n```\n\n" \
             f"Provide a clear, concise, and step-by-step explanation of " \
             f"how the problem is solved by the given code. Focus on the algorithm, " \
             f"data structures used, and the logic of the code."
    
    return prompt

if __name__ == "__main__":
    print("Testing src/prompt_engineering.py")

    sample_problem_content = """
    Write a Python function `is_even(number)` that returns True if the given number is even, and False otherwise.

    Example:
    Input: number = 4
    Output: True

    Input: number = 7
    Output: False
    """

    sample_python_solution = """
def is_even(number):
    return number % 2 == 0
"""

    print("\nExample Code Generation Prompt (Python)")
    code_prompt_python = generation_prompt(sample_problem_content, target_language="python")
    print(code_prompt_python)

    print("\n" + "="*50 + "\n")

    print("Example Explanation Prompt (Python)")
    explanation_prompt_python = explanation_prompt(sample_problem_content, sample_python_solution, target_language="python")
    print(explanation_prompt_python)

    print("\nPrompt Engineering Test Complete")