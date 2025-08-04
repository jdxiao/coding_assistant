"""
This file contains a dictionary of predefined problems, their ground truth solutions,
and associated test cases. This is used to test the refinement pipeline without
relying on a dynamic dataset or LLM generation for test cases.
"""

# Key: Function name
# Value: A dictionary containing:
# - content: The problem description.
# - function_name: The name of the function to be implemented.
# - correct_solution_raw: The ground truth solution code as a string.
# - test_cases: A list of dictionaries, each containing 'input' and 'expected_output' keys.

PREDEFINED_PROBLEMS = {
    "add_numbers": {
        "content": "Given two integers, `a` and `b`, write a function that returns their sum.",
        "function_name": "add_numbers",
        "correct_solution_raw": "def add_numbers(a, b):\n    return a + b",
        "test_cases": [
            {"input": (5, 3), "expected_output": 8},
            {"input": (-1, 1), "expected_output": 0},
            {"input": (0, 0), "expected_output": 0},
            {"input": (100, 200), "expected_output": 300},
        ],
    },
    "reverse_string": {
        "content": "Write a function that takes a string as input and returns the string in reverse order.",
        "function_name": "reverse_string",
        "correct_solution_raw": "def reverse_string(s):\n    return s[::-1]",
        "test_cases": [
            {"input": ("hello",), "expected_output": "olleh"},
            {"input": ("python",), "expected_output": "nohtyp"},
            {"input": ("a",), "expected_output": "a"},
            {"input": ("",), "expected_output": ""},
        ],
    },
}
