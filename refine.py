import json
from llm import LLMInterface
from agent import execute_code
from prompt import generation_prompt
import sys
from dataset import load_leetcode_dataset

def generate_test_cases_with_outputs(llm, problem_content: str, solution_code: str, function_name: str) -> list:
    """
    Generates test case inputs using the LLM, then uses the correct solution
    to produce the expected outputs.

    Args:
        llm (LLMInterface): The LLM interface object.
        problem_content (str): The problem description.
        solution_code (str): The correct solution code from the dataset.
        function_name (str): The name of the function to be tested.

    Returns:
        list: A list of dictionaries representing test cases with inputs and expected outputs.
    """
    print("Generating test case inputs using the LLM...")
    
    prompt = f"""
        Given the following programming problem:
        {problem_content}

        Please generate at least 3 diverse test case inputs for this problem. Provide the output as a
        JSON array of lists, where each list contains the arguments for the function call.
        For example, if the function takes one argument, your output might look like:
        [[1], [2], [3]]
        If it takes two arguments, it might be:
        [[1, 2], [3, 4], [5, 6]]

        JSON Output:
        """
    try:
        response_json = llm.generate_response(prompt)
        test_case_inputs = json.loads(response_json)
        
        print("Generated test case inputs. Now executing the correct solution to get expected outputs.")
        
        test_cases_with_outputs = []
        for inputs in test_case_inputs:
            # Function inputs should be a tuple for execution
            inputs_tuple = tuple(inputs) if isinstance(inputs, list) else (inputs,)
            
            execution_results = execute_code(solution_code, function_name, [{"input": inputs_tuple, "expected_output": None}])
            
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
    print("\n" + "="*80)
    print("Attempting to solve the problem:")
    print(f"Problem content: {problem_content[:100]}...")
    
    # Initial code generation
    initial_prompt = generation_prompt(problem_content, target_language="python")
    initial_raw_output = llm.generate_response(initial_prompt)
    current_solution, _ = llm.parse_llm_output(initial_raw_output)
    
    print("\nInitial Solution Generated:")
    print(current_solution)
    
    # Start the refinement loop
    refinement_attempt = 0
    while True:
        refinement_attempt += 1
        print(f"\n--- Running test for refinement attempt {refinement_attempt} ---")
        
        execution_results = execute_code(current_solution, function_name, test_cases)
        
        # Check if all tests passed
        if execution_results["success"]:
            print("Code executed successfully and all tests passed!")
            print(f"Final solution:\n{current_solution}")
            break
        else:
            print("Tests failed.")
            error_message = json.dumps(execution_results, indent=2)
            print(f"Error Message:\n{error_message}")
            
            user_choice = input("Would you like to refine the solution? (yes/no): ").lower().strip()
            
            if user_choice != 'yes':
                print("Refinement process stopped by user. Giving up on this problem.")
                break
            
            # Generate a refinement prompt
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
            
            # Generate the refined solution
            refined_raw_output = llm.generate_response(refinement_prompt)
            current_solution, _ = llm.parse_llm_output(refined_raw_output)

            print("\nRefined Solution Generated:")
            print(current_solution)


def main():
    """
    Main function for the dataset-driven pipeline.
    """
    # Configuration
    model_name = "bigcode/starcoderbase-1b"
    lora_adapters_path = "./lora_adapters"
    
    try:
        llm = LLMInterface(model_name=model_name, lora_adapters_path=lora_adapters_path)
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        print("Please make sure you have run finetune.py successfully and the adapters exist.")
        sys.exit(1)

    print("Welcome to the code generation and interactive refinement pipeline.")
    print("Loading a small sample of the dataset...")
    test_dataset = load_leetcode_dataset(num_problems=2)

    if not test_dataset:
        print("Failed to load dataset. Please ensure the dataset loader is working.")
        return

    for i, problem_data in enumerate(test_dataset):
        problem_content = problem_data.get('content')
        function_name = problem_data.get("function_name", "solve")
        correct_solution_code = problem_data.get('python')

        if not problem_content or not correct_solution_code:
            print(f"Skipping problem {i+1}: Missing content or correct solution.")
            continue
        
        test_cases = generate_test_cases_with_outputs(llm, problem_content, correct_solution_code, function_name)

        if not test_cases:
            print(f"Skipping problem {i+1}: Failed to generate valid test cases.")
            continue
        
        run_refinement_pipeline(llm, problem_content, function_name, test_cases)
        
        continue_choice = input("Continue to the next problem? (yes/no): ").lower().strip()
        if continue_choice != 'yes':
            break

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
