import json
import io
import sys
import contextlib

# The maximum time a function is allowed to run for a single test case
TIMEOUT = 5

@contextlib.contextmanager
def stdout_redirected(new_stdout):
    """
    Context manager to redirect stdout to a different stream.
    Used to capture print statements from the executed code.
    """
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

def execute_code(code, function_name, test_cases):
    """
    Executes the given Python code with provided test cases.

    This function dynamically imports the provided code string, runs it
    against each test case, and captures the output and any errors.

    Args:
        code (str): The Python code to execute.
        function_name (str): The name of the function to be tested.
        test_cases (List[Dict[str, Any]]): A list of dictionaries, where each
                                           dictionary contains 'input' and 'expected_output' keys.

    Returns:
        Dict[str, Any]: A dictionary containing the execution results, including
                        success status, a message, and detailed results for each test case.
    """
    results = {
        "success": True,
        "message": "All tests passed.",
        "test_results": []
    }
    
    # Namespace to execute the code in
    namespace = {}
    try:
        # Compile and execute the code to make the function available in the namespace
        exec(code, namespace)
        
        # Check if the function exists in the namespace
        if function_name not in namespace:
            results["success"] = False
            results["message"] = f"Function '{function_name}' not found in the generated code."
            return results
        
        target_function = namespace[function_name]

        # Iterate through all the test cases
        for test_case in test_cases:
            test_input = test_case.get("input")
            expected_output = test_case.get("expected_output")
            
            output = None
            error = None
            
            try:
                # Capture print statements and execute the function
                with stdout_redirected(io.StringIO()) as out:
                    # Execute the function with the given input
                    output = target_function(*test_input)
                
                # Check if the output matches the expected output
                if output != expected_output:
                    results["success"] = False
                    error = f"Test case failed. Input: {test_input}, Expected: {expected_output}, Got: {output}"
                    results["message"] = f"Tests failed. See details."
            
            except Exception as e:
                # Capture any exceptions that occur during execution
                results["success"] = False
                error = f"An exception occurred: {type(e).__name__}: {e}"
                results["message"] = f"Tests failed. See details."
            
            # Record the result for this test case
            results["test_results"].append({
                "input": test_input,
                "expected": expected_output,
                "output": output,
                "error": error
            })
            
            # Stop execution if any test case fails
            if not results["success"]:
                return results

    except Exception as e:
        # Error handling for compilation or runtime errors
        results["success"] = False
        results["message"] = f"An error occurred while compiling or running the code: {type(e).__name__}: {e}"
        results["test_results"].append({
            "input": None,
            "expected": None,
            "output": None,
            "error": results["message"]
        })

    return results


# For file testing purposes
if __name__ == '__main__':
    sample_code = """
        def sum_list(nums):
            total = 0
            for num in nums:
                total += num
            return total
        """
    
    test_cases = [
        {"input": ([1, 2, 3],), "expected_output": 6},
        {"input": ([5, 10, 15],), "expected_output": 30}
    ]

    print("Running example 1 (should pass):")
    execution_result = execute_code(sample_code, "sum_list", test_cases)
    print(json.dumps(execution_result, indent=2))
    
    print("\n" + "="*50 + "\n")

    sample_code_with_bug = """
        def sum_list(nums):
            return 0 # This is a bug!
        """
    print("Running example 2 (should fail):")
    execution_result = execute_code(sample_code_with_bug, "sum_list", test_cases)
    print(json.dumps(execution_result, indent=2))
