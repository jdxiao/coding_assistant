import json
from llm import LLMInterface
from agent import execute_code
from prompt import generation_prompt
import sys

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
    
    # Initial solution generation
    initial_prompt = generation_prompt(problem_content, target_language="python")
    current_solution = llm.generate_solution(initial_prompt)
    print("\nInitial Solution Generated:")
    print(current_solution)
    
    # Start the refinement loop
    refinement_attempt = 0
    while True:
        refinement_attempt += 1
        print(f"\nRunning test for refinement attempt {refinement_attempt}")
        
        execution_results = execute_code(current_solution, function_name, test_cases)
        
        if execution_results["success"]:
            print("Code executed successfully and all tests passed!")
            print(f"Final solution:\n{current_solution}")
            break
        else:
            print("Tests failed.")
            error_message = json.dumps(execution_results, indent=2)
            print(f"Error Message:\n{error_message}")
            
            # Ask user if they want to refine the solution
            user_choice = input("Would you like to refine the solution? (yes/no): ").lower().strip()
            
            if user_choice != 'yes':
                print("Refinement process stopped by user. Giving up on this problem.")
                break
            
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
            current_solution = llm.generate_solution(refinement_prompt)
            print("\nRefined Solution Generated:")
            print(current_solution)

def main():
    """
    Main function to run the dynamic code generation and refinement pipeline.
    Using a sample problem and test cases for demonstration.
    """
    # Configuration
    model_name = "bigcode/starcoderbase-1b"
    lora_adapters_path = "./lora_adapters"
    
    # Load the LLM with the fine-tuned adapters
    try:
        llm = LLMInterface(model_name=model_name, lora_adapters_path=lora_adapters_path)
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        print("Please make sure you have run finetune.py successfully and the adapters exist.")
        sys.exit(1)

    # Example usage
    print("Welcome to the code generation assistant.")
    print("Provide a problem and test cases to run the pipeline.")  # User input disabled for now

    # A sample problem for demonstration
    sample_problem = """
        Write a Python function `find_max(numbers)` that takes a list of 
        numbers and returns the largest number in the list. Assume the list is not empty.
        """
    # Test cases for the sample problem
    sample_test_cases = [
        {"input": ([1, 5, 2, 8],), "expected_output": 8},
        {"input": ([100, 20, 30],), "expected_output": 100},
        {"input": ([7],), "expected_output": 7}
    ]

    while True:
        # Get user input
        user_choice = input("Would you like to run the sample problem? (yes/no): ").lower().strip()
        
        if user_choice == 'yes':
            problem_to_solve = sample_problem
            test_cases_to_use = sample_test_cases
            function_name_to_use = "find_max"
            run_refinement_pipeline(llm, problem_to_solve, function_name_to_use, test_cases_to_use)
        elif user_choice == 'no':
            print("You can edit the script to provide your own problem and test cases.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
        
        # Ask if the user wants to continue
        continue_choice = input("Run another problem? (yes/no): ").lower().strip()
        if continue_choice != 'yes':
            break

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
