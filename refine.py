import json
import re
import sys
from llm import LLMInterface
from agent import execute_code
from prompt import generation_prompt, explanation_prompt, test_case_prompt
from dataset import load_leetcode_dataset
from testcase import PREDEFINED_PROBLEMS


# Set to True to use the predefined tests and False to use LLM-generated tests.
USE_PREDEFINED_PROBLEMS = True

def extract_function_name(code_string):
    """
    Extracts the function name from a string of Python code.
    
    Args:
        code_string (str): A string containing a Python function definition.
        
    Returns:
        str: The name of the first function found in the string, or None if no function is found.
    """
    # Regex to find the function definition
    # This assumes the function is defined in a standard way, e.g., def function_name
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_string)
    if match:
        return match.group(1)
    return None

# Generates test cases using the LLM based on the problem content and ground truth solution.
def generate_test_cases_with_outputs(llm, problem_content: str, ground_truth_raw: str, function_name: str) -> list:
    """
    Generates test case inputs using the LLM, then uses the ground truth solution code
    to produce the expected outputs.

    Args:
        llm (LLMInterface): The LLM interface object, used for input generation and parsing.
        problem_content (str): The problem description.
        ground_truth_raw (str): The raw string from the dataset containing both code and explanation.
        function_name (str): The name of the function to be tested.

    Returns:
        list: A list of dictionaries representing test cases with inputs and expected outputs.
    """
    print("Generating test case inputs using the LLM...")
    
    prompt = test_case_prompt(problem_content)
    
    try:
        response_json = llm.generate_response(prompt)
        test_case_inputs = json.loads(response_json)
        
        # Parse ground truth solution for executable code
        print("Parsing the ground truth solution from the dataset to get executable code.")
        correct_solution_code, _ = llm.parse_llm_output(ground_truth_raw)
        
        if not correct_solution_code:
            print("Failed to parse correct solution code from the dataset. Cannot generate test cases.")
            return []
            
        print("Executing the correct solution to get expected outputs.")
        
        # Execute the correct solution code with the generated inputs to get expected outputs
        test_cases_with_outputs = []
        for inputs in test_case_inputs:
            inputs_tuple = tuple(inputs) if isinstance(inputs, list) else (inputs,)
            
            execution_results = execute_code(correct_solution_code, function_name, [{"input": inputs_tuple, "expected_output": None}])
            
            # Check if execution was successful and if outputs were generated
            if execution_results["success"] and execution_results["test_results"]:
                test_cases_with_outputs.append({
                    "input": inputs_tuple,
                    "expected_output": execution_results["test_results"][0]["actual_output"]
                })
            else:
                print(f"Error executing correct solution with input: {inputs}. Skipping this test case.")
                continue
        
        return test_cases_with_outputs

    except (json.JSONDecodeError, Exception) as e:
        print(f"Failed to generate or parse test case inputs: {e}")
        return []

# Runs the code generation and refinement loop for a single problem.
def run_refinement_pipeline(llm, problem_content: str, function_name: str, test_cases: list):
    """
    Runs the code generation and self-refinement loop for a single problem.
    
    The loop continues until the solution passes all tests or the user decides to stop.

    Args:
        llm (LLMInterface): The LLM interface object.
        problem_content (str): The description of the programming problem.
        function_name (str): The name of the function to test.
        test_cases (list): A list of dictionaries with 'input' and 'expected_output'.
    """

    print("\nAttempting to solve the problem:")
    print(f"Problem content: {problem_content[:100]}...")
    
    # Initial code generation
    initial_prompt = generation_prompt(problem_content, target_language="python")
    initial_raw_output = llm.generate_response(initial_prompt)
    current_solution, _ = llm.parse_llm_output(initial_raw_output)
    
    print("\nInitial Solution Generated:")
    print(current_solution)

    # Initial explanation generation
    print("\n" + "="*80)
    print("Generating Initial Explanation for the Solution...")
    explanation_prompt_str = explanation_prompt(problem_content, current_solution)
    explanation_raw_output = llm.generate_response(explanation_prompt_str)
    print(explanation_raw_output)
    print("="*80)
    
    # Start the refinement loop
    refinement_attempt = 0
    while True:
        refinement_attempt += 1
        print(f"\nRunning test for refinement attempt {refinement_attempt}")
        
        execution_results = execute_code(current_solution, function_name, test_cases)
        
        # Check if all tests passed
        if execution_results["success"]:
            print("Code executed successfully and all tests passed!")
            print(f"Final solution:\n{current_solution}")
            break
    
        # If tests failed, ask the user if they want to refine the solution
        else:
            print("Tests failed.")
            error_message = json.dumps(execution_results, indent=2)
            print(f"Error Message:\n{error_message}")
            
            user_choice = input("Would you like to refine the solution? (yes/no): ").lower().strip()
            
            if user_choice != 'yes':
                print("Refinement process stopped by user. Giving up on this problem.")
                break
            
            # Generate a refinement prompt based on the error message
            print("Refining solution...")
            refinement_prompt = f"""
                Given the following programming problem:
                {problem_content}

                I previously generated this Python code, but it failed to pass the tests with the following error:
                {error_message}

                Please analyze the error and provide a corrected and fully functional Python solution. The solution should be enclosed in ```python...``` tags.

                ```python
                {current_solution}
                ```

                Corrected Solution:
                ```python
                """

            # Generate a refined solution using the LLM
            refined_raw_output = llm.generate_response(refinement_prompt)
            current_solution, _ = llm.parse_llm_output(refined_raw_output)

            print("\nRefined Solution Generated:")
            print(current_solution)

            # Generate and show explanation for the refined solution
            print("\n" + "="*80)
            print("Generating Refined Explanation for the Solution...")
            explanation_prompt_str = explanation_prompt(problem_content, current_solution)
            explanation_raw_output = llm.generate_response(explanation_prompt_str)
            print(explanation_raw_output)
            print("="*80)


def main():
    """
    Main function for the pipeline.
    """

    model_name = "bigcode/starcoderbase-1b"
    lora_adapters_path = "./lora_adapters"
    
    try:
        llm = LLMInterface(model_name=model_name, lora_adapters_path=lora_adapters_path)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        print("Please ensure your LLM environment is correctly set up.")
        sys.exit(1)

    print("Welcome to the code generation and interactive refinement pipeline.")
    
    # predefined problems
    if USE_PREDEFINED_PROBLEMS:
        print("\nUsing predefined problems and tests.")
        # Iterate through the predefined problems
        for problem_data in PREDEFINED_PROBLEMS.values():
            problem_content = problem_data.get('content')
            function_name = problem_data.get("function_name")
            test_cases = problem_data.get("test_cases")
            
            print(f"\nProcessing predefined problem: '{function_name}'")
            
            if not problem_content or not function_name or not test_cases:
                print("Skipping predefined problem: Missing content, function name, or test cases.")
                continue

            run_refinement_pipeline(llm, problem_content, function_name, test_cases)
            
            continue_choice = input("Continue to the next problem? (yes/no): ").lower().strip()
            if continue_choice != 'yes':
                break

    # leetcode dataset problems
    else:
        print("\nUsing the Leetcode dataset")
        print("Loading a small sample of the dataset...")
        test_dataset = load_leetcode_dataset(num_problems=2)

        if not test_dataset:
            print("Failed to load dataset. Please ensure the dataset loader is working.")
            return

        for i, problem_data in enumerate(test_dataset):
            problem_content = problem_data.get('content')
            correct_solution_raw = problem_data.get('python')

            if not problem_content or not correct_solution_raw:
                print(f"Skipping problem {i+1}: Missing content or correct solution.")
                continue
            
            # Extract the function name from the solution code
            function_name = extract_function_name(correct_solution_raw)
            if not function_name:
                print(f"Skipping problem {i+1}: Failed to extract function name from the solution code.")
                continue
                
            print(f"Generating test cases for '{function_name}' from the LLM.")
            test_cases = generate_test_cases_with_outputs(llm, problem_content, correct_solution_raw, function_name)
            
            if not test_cases:
                print(f"Skipping problem {i+1}: Failed to get valid test cases.")
                continue
            
            run_refinement_pipeline(llm, problem_content, function_name, test_cases)
            
            continue_choice = input("Continue to the next problem? (yes/no): ").lower().strip()
            if continue_choice != 'yes':
                break

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
