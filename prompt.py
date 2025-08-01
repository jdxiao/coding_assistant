def generation_prompt(problem_content: str, target_language: str = "python"):
    """
    Generates a prompt string to be used for fine-tuning an LLM on code generation.
    The prompt instructs the model to provide a complete solution to a given algorithmic problem
    in the specified programming language.

    Args:
        problem_content (str): The content of the algorithmic problem to be solved.
        target_language (str): The programming language in which the solution should be provided.
            Default is "python".
    Returns:
        str: A formatted prompt string for the LLM.
    """


    prompt = f"""Given the following algorithmic problem:
                ```problem
                {problem_content}
                ```

                Provide a complete and correct solution in {target_language}.
                Your response should only contain the code solution, enclosed within a
                markdown code block. Do not include any additional explanations, comments
                outside the code block, or conversational text. Ensure the code is ready to be executed.

                For example, if the problem asks for a function 'add(a, b)',
                your response should start with:

                ```{target_language}
                def add(a, b):
                    # your code here
                ```"""

    return prompt

def explanation_prompt(problem_content: str, solution_code: str, target_language: str = "python"):
    """
    Generates a prompt for an LLM to explain a given coding problem and its solution.
    This prompt is designed to generate a clear and concise explanation of how the solution works
    for the specified problem.

    Args:
        problem_content (str): The content of the algorithmic problem to be explained.
        solution_code (str): The code solution to the problem that needs to be explained.
        target_language (str): The programming language of the solution code. Default is "python".
    Returns:
        str: A formatted prompt string for the LLM to generate an explanation.
    """

    prompt = f"""Given the following algorithmic problem:
                ```problem
                {problem_content}
                ```

                And the following solution code in {target_language}:

                ```{target_language}
                {solution_code}
                ```

                Provide a clear, concise, and step-by-step explanation of
                how the problem is solved by the given code. Focus on the algorithm,
                data structures used, and the logic of the code."""
    
    return prompt

# For file testing purposes
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