#!/usr/bin/env python3
"""
BetterPython: AI-Assisted Python Code Enhancement Tool

This script provides a comprehensive solution for enhancing Python code using AI assistance.
It processes Python source files to add or improve type hints, comments, and docstrings,
resulting in more readable and maintainable code.

Key Features:
1. Automated code enhancement using AI (Anthropic API)
2. Addition of type hints to function parameters and return values
3. Generation of comprehensive docstrings for functions and classes
4. Insertion of inline comments to explain complex logic
5. Preservation of original code structure and functionality
6. Handling of nested functions and classes
7. Custom error handling and logging for robust execution

Main Components:
- parse_python_file: Parses Python source code into an Abstract Syntax Tree (AST)
- extract_functions_and_classes_with_text: Extracts top-level functions and classes from the AST
- generate_ai_response: Interacts with the Anthropic API to enhance code snippets
- reassemble_code: Reconstructs the enhanced code while maintaining original structure
- run_black: Applies the Black code formatter for consistent styling

Usage:
python BetterPython.py -f <input_file> -o <output_file>

Requirements:
- Python 3.7+
- anthropic library
- black library

Note: This script requires an Anthropic API key, which should be securely stored and accessed.

Author: Richard Bunker
Version: 1.0.0
License: MIT
"""

import argparse
import ast
import functools
import logging
import os
import re
import subprocess
import sys
import traceback


import anthropic
from anthropic import Anthropic
from typing import Optional, Union, Callable, Tuple, List, Any, Dict


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="BetterPython.log",
    filemode="a",  # Append to the log file
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def he(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that enhances function execution with logging and error handling.

    This decorator wraps the given function to provide:
    1. Logging of function calls with their arguments
    2. Logging of successful function completion
    3. Error logging in case of exceptions

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: A wrapped version of the input function with added logging and error handling.

    Raises:
        Exception: Re-raises any exception caught during the function execution.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        A wrapper function that adds logging and error handling to the decorated function.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the wrapped function.

        Raises:
            Exception: Re-raises any exception caught during the function execution.
        """
        try:
            # Log the function call with its arguments
            logging.info(f"Calling {func.__name__}")
            logging.debug(
                f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}"
            )

            # Execute the wrapped function and store the result
            result = func(*args, **kwargs)

            # Log successful completion of the function
            logging.info(f"{func.__name__} completed successfully")

            # Return the result of the wrapped function
            return result

        except Exception as e:
            # Log the error details with full stack trace for debugging purposes
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)

            # Re-raise the exception to maintain the original error behavior
            # This allows the calling code to handle the exception if needed
            raise

    # Return the wrapper function, which will replace the original function
    return wrapper


@he
def run_black(file_path: str) -> bool:
    """
    Run the Black code formatter on the specified file.

    This function attempts to format the given Python file using the Black code formatter.
    It first checks if the file exists, then runs Black on the file, capturing any output
    or errors.

    Args:
        file_path (str): The path to the Python file to be formatted.

    Returns:
        bool: True if the formatting was successful, False otherwise.

    Raises:
        No exceptions are raised; all exceptions are caught and handled internally.
    """

    # First, check if the file exists
    try:
        with open(file_path, "r"):
            pass
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return False

    # Attempt to run Black on the file
    try:
        # Run Black as a subprocess, capturing output and errors
        result: Optional[subprocess.CompletedProcess[str]] = subprocess.run(
            ["black", file_path],
            check=True,  # Raise CalledProcessError if the command returns a non-zero exit code
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Return strings instead of bytes
        )

        # If we reach here, Black ran successfully
        print(f"Successfully formatted {file_path} with Black.")
        return True

    except subprocess.CalledProcessError as e:
        # Black encountered an error while formatting
        print(f"Error: Black formatting failed for {file_path}")
        print(f"Error message: {e.stderr}")  # Print the error message from stderr
        return False

    except FileNotFoundError:
        # Black is not installed or not in the system PATH
        print("Error: Black is not installed or not in the system PATH.")
        print("Please install Black using 'pip install black' and try again.")
        return False


@he
def parse_python_file(file_path: str) -> Optional[ast.AST]:
    """
    Parse a Python file and return its Abstract Syntax Tree (AST).

    This function attempts to read and parse a Python file, handling various
    potential errors that may occur during the process. It uses the ast module
    to generate an Abstract Syntax Tree from the source code.

    Args:
        file_path (str): The path to the Python file to be parsed.

    Returns:
        Optional[ast.AST]: The AST of the parsed Python file, or None if parsing fails.

    Raises:
        No exceptions are raised; all exceptions are caught and handled internally.
    """
    try:
        # Attempt to open and read the file with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as file:
            source_code: str = file.read()

        # Parse the source code into an AST
        tree: ast.AST = ast.parse(source_code, filename=file_path)

        # Return the successfully parsed AST
        return tree

    except SyntaxError as e:
        # Handle syntax errors in the Python file
        print(f"Error: Failed to parse {file_path}. Invalid Python syntax.")
        print(f"Details: {e}")
        print("Full traceback:")
        print(traceback.format_exc())

        # Return None to indicate parsing failure
        return None

    except UnicodeDecodeError as e:
        # Handle encoding errors when reading the file
        print(f"Error: Failed to read {file_path}. The file encoding may not be UTF-8.")
        print(f"Details: {e}")

        # Return None to indicate parsing failure
        return None

    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error: An unexpected error occurred while parsing {file_path}.")
        print(f"Details: {e}")
        print("Full traceback:")
        print(traceback.format_exc())

        # Return None to indicate parsing failure
        return None


@he
def extract_functions_and_classes_with_text(
    tree: ast.AST, source_code: str
) -> List[Tuple[Union[ast.FunctionDef, ast.ClassDef], str, int]]:
    """
    Extract functions and classes from an AST, along with their source code and nesting level.

    This function traverses the Abstract Syntax Tree (AST) and extracts all top-level
    function and class definitions, along with their full source code and nesting level.
    It uses a custom AST visitor to perform the traversal and extraction.

    Args:
        tree (ast.AST): The Abstract Syntax Tree to traverse.
        source_code (str): The original source code string.

    Returns:
        List[Tuple[Union[ast.FunctionDef, ast.ClassDef], str, int]]: A list of tuples,
        each containing:
        - The AST node (either FunctionDef or ClassDef)
        - The full source code of the function or class as a string
        - The nesting level (0 for top-level definitions)

    Note:
        This function does not extract nested functions or classes.
    """
    # Initialize an empty list to store the extracted nodes with their text and nesting level
    nodes_with_text: List[Tuple[Union[ast.FunctionDef, ast.ClassDef], str, int]] = []

    class Visitor(ast.NodeVisitor):
        """
        A custom AST visitor to extract function and class definitions.
        """

        def __init__(self) -> None:
            self.parent_stack: List[ast.AST] = []  # Stack to keep track of parent nodes

        def visit(self, node: ast.AST) -> None:
            """
            Visit a node in the AST and extract relevant information if it's a function or class.

            Args:
                node (ast.AST): The node to visit in the AST.
            """
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check if the node is a top-level function or class definition
                if not self.parent_stack or not isinstance(
                    self.parent_stack[-1], (ast.FunctionDef, ast.ClassDef)
                ):
                    # Extract the full source code of the node
                    full_text: str = ast.get_source_segment(source_code, node)

                    # Calculate the nesting level based on the parent stack depth
                    nesting_level: int = len(self.parent_stack)

                    # Append the node, its source code, and nesting level to the result list
                    nodes_with_text.append((node, full_text, nesting_level))

            # Add the current node to the parent stack before visiting its children
            self.parent_stack.append(node)

            # Visit all child nodes recursively
            super().generic_visit(node)

            # Remove the current node from the parent stack after visiting its children
            self.parent_stack.pop()

    # Create an instance of the custom visitor
    visitor = Visitor()

    # Traverse the AST using the custom visitor
    visitor.visit(tree)

    # Return the list of extracted nodes with their text and nesting level
    return nodes_with_text


@he
def setup_anthropic_stuff(
    anthropic_key_file: str | None = None,
) -> tuple[str, Anthropic]:
    """
    Set up Anthropic API key and create an Anthropic client object.

    This function retrieves the Anthropic API key either from a file or an environment variable,
    and initializes an Anthropic client object for API interactions.

    Args:
        anthropic_key_file (str | None): Path to the file containing the Anthropic API key.
            If None, the function will attempt to retrieve the key from an environment variable.

    Returns:
        tuple[str, Anthropic]: A tuple containing:
            - str: The Anthropic API key.
            - Anthropic: An initialized Anthropic client object.

    Raises:
        ValueError: If the API key is not found in the file or environment variable.

    Note:
        The function prioritizes the file-based key if provided. If the file is not accessible
        or not provided, it falls back to the environment variable.
    """

    # Initialize the API key variable
    anthropic_api_key: str | None = None

    # Attempt to read the API key from a file if provided
    if anthropic_key_file:
        try:
            with open(anthropic_key_file, "r") as file:
                anthropic_api_key = file.read().strip()
        except IOError as e:
            # Handle potential file read errors
            print(f"Error: Unable to read API key from file: {anthropic_key_file}")
            print(f"IOError details: {e}")

    # If no file is provided or reading from file failed, attempt to get the key from an environment variable
    if not anthropic_api_key:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Raise an error if no API key is found after both attempts
    if not anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set and no valid API key file is provided."
        )

    # Create an Anthropic client object using the API key
    anthropic_object: Anthropic = anthropic.Anthropic(api_key=anthropic_api_key)

    # Return the API key and the Anthropic client object as a tuple
    return anthropic_api_key, anthropic_object


@he
def generate_ai_response(
    ai_client_object: Any,
    ai_model: str,
    max_tokens: int,
    temperature: float,
    system_message: str,
    prompt_string: str,
) -> Tuple[str, float]:
    """
    Generate an AI response using the provided AI client and parameters.

    This function sends a request to an AI model, processes the response,
    handles token usage, and calculates the cost of the query.

    Args:
        ai_client_object (Any): The AI client object used to make the API call.
        ai_model (str): The name or identifier of the AI model to use.
        max_tokens (int): The maximum number of tokens for the AI response.
        temperature (float): The temperature setting for response generation (0.0 to 1.0).
        system_message (str): The system message to provide context to the AI.
        prompt_string (str): The user's input prompt for the AI.

    Returns:
        Tuple[str, float]: A tuple containing the processed AI response text and the cost of the query.

    Raises:
        Exception: If there's an error during the API call or response processing.
    """

    try:
        # Make the API call to generate the AI response
        message = ai_client_object.messages.create(
            model=ai_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=[{"role": "user", "content": prompt_string}],
        )

        # Process the AI's response
        assistant_response = message.content
        response_text: str = ""

        # Concatenate all content blocks in the response
        for content_block in assistant_response:
            response_text += content_block.text + "\n"

        # Extract token usage information
        input_tokens: int = message.usage.input_tokens
        output_tokens: int = message.usage.output_tokens

    except Exception as e:
        # Log the error and re-raise the exception
        logging.error(f"Error in API call: {str(e)}")
        raise

    # Log a debug message with the start of the response text (first 120 characters)
    logging.debug(
        f"in generate_ai_response, response text starts: {response_text[:120]}..."
    )

    # Calculate the cost of the query based on token usage
    cost: float = calculate_query_cost(ai_model, input_tokens, output_tokens)

    # Log detailed usage information for debugging purposes
    logging.debug(
        f"anthropic token usage: "
        f"input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}, cost={cost:.6f}"
    )

    # Strip any trailing newlines from the response and return it along with the cost
    return response_text.strip(), cost


@he
def calculate_query_cost(
    model: str, input_tokens: int, output_tokens: int
) -> Optional[float]:
    """
    Calculate the cost of a query based on the model, input tokens, and output tokens.

    This function determines the cost of a query for various AI models by considering
    the number of input and output tokens and their respective prices per token.

    Args:
        model (str): The name of the AI model being used.
        input_tokens (int): The number of input tokens in the query.
        output_tokens (int): The number of output tokens generated.

    Returns:
        Optional[float]: The total cost of the query in dollars, or None if the model
            price information is not available.

    Raises:
        None

    Note:
        If the model price information is not found, a warning is logged and None is returned.
    """

    # Dictionary containing price information for various models
    model_prices: Dict[str, Dict[str, float]] = {
        "claude-3-opus-20240229": {"input": 0.000015, "output": 0.000075},
        "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
        "claude-3-5-sonnet-20240620": {"input": 0.000003, "output": 0.000015},
        "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125},
        "gpt-4o": {"input": 0.000005, "output": 0.000015},
        "gpt-4-turbo": {"input": 0.00001, "output": 0.000030},
        "gpt-3.5-turbo-0125": {"input": 0.0000005, "output": 0.0000015},
    }

    # Retrieve the price information for the specified model
    model_price: Optional[Dict[str, float]] = model_prices.get(model)

    # Check if the model price information is available
    if not model_price:
        # Log a warning if price information is not found
        logging.warning(f"Price information not available for model: {model}")
        return None

    # Calculate the cost for input tokens
    input_cost: float = input_tokens * model_price["input"]

    # Calculate the cost for output tokens
    output_cost: float = output_tokens * model_price["output"]

    # Calculate the total cost by summing input and output costs
    total_cost: float = input_cost + output_cost

    # Return the total cost of the query
    return total_cost


@he
def write_modified_code(modified_code: str, output_file: str) -> None:
    """
    Write the modified code to the specified output file.

    This function attempts to write the provided modified code to the given output file.
    If successful, it prints a confirmation message. If an error occurs during the writing
    process, it prints an error message and re-raises the exception.

    Args:
        modified_code (str): The modified code to be written to the file.
        output_file (str): The path to the file where the modified code will be written.

    Raises:
        IOError: If there's an error writing to the output file.

    Returns:
        None
    """
    try:
        # Open the output file in write mode using a context manager
        # This ensures the file is properly closed after writing
        with open(output_file, "w", encoding="utf-8") as file:
            # Write the modified code to the file
            # The entire content of modified_code is written at once
            file.write(modified_code)

        # Print a success message to confirm the operation
        # This provides feedback to the user or calling process
        print(f"Modified code successfully written to {output_file}")

    except IOError as e:
        # If an IOError occurs (e.g., permission denied, disk full),
        # catch it and provide a more informative error message
        print(f"Error occurred while writing to {output_file}: {str(e)}")

        # Re-raise the exception to allow for further handling if needed
        # This allows the calling code to implement additional error handling if desired
        raise

    except Exception as e:
        # Catch any other unexpected exceptions
        # This provides a catch-all for unforeseen errors
        print(f"Unexpected error occurred: {str(e)}")
        raise


@he
def get_valid_output_file(output_file: str) -> str:
    """
    Prompt the user for a valid output file name, handling existing files.

    This function checks if the given output file already exists and prompts
    the user to overwrite, choose a new filename, or cancel the operation.
    It continues to prompt until a valid choice is made or a non-existing
    filename is provided.

    Args:
        output_file (str): The initial output file name to check.

    Returns:
        str: A valid output file name that doesn't exist or is approved for overwriting.

    Raises:
        SystemExit: If the user chooses to cancel the operation.
    """

    while os.path.exists(output_file):
        # Inform the user that the file already exists
        print(f"The file '{output_file}' already exists.")

        # Prompt the user for their choice
        choice: str = input(
            "Do you want to (o)verwrite, choose a (n)ew filename, or (c)ancel? "
        ).lower()

        if choice == "o":
            # User chose to overwrite, return the original filename
            return output_file
        elif choice == "n":
            # User chose to enter a new filename
            output_file = input("Enter a new output filename: ")
        elif choice == "c":
            # User chose to cancel the operation
            print("Operation cancelled.")
            sys.exit(0)
        else:
            # Invalid choice, inform the user and continue the loop
            print("Invalid choice. Please try again.")

    # Return the valid output file name (either new or approved for overwriting)
    return output_file


@he
def split_content(
    content: str, ranges: List[Tuple[int, int, int]]
) -> List[Tuple[str, int]]:
    """
    Split the given content into parts based on specified line ranges and nesting levels.

    This function processes a string of content and a list of ranges, where each range
    defines a start line, end line, and nesting level. It splits the content into parts
    according to these ranges, assigning the appropriate nesting level to each part.

    Args:
        content (str): The input string to be split.
        ranges (List[Tuple[int, int, int]]): A list of tuples, each containing
            (start_line, end_line, nesting_level).

    Returns:
        List[Tuple[str, int]]: A list of tuples, each containing a part of the
        content and its corresponding nesting level.

    Note:
        - Line numbers in the ranges are 1-indexed.
        - The function handles gaps between ranges and content after the last range.
        - Gaps and remaining content are assigned a nesting level of 0.
    """
    # Split the content into lines, preserving line endings
    lines: List[str] = content.splitlines(keepends=True)

    # Initialize the list to store the resulting parts
    parts: List[Tuple[str, int]] = []

    # Keep track of the last processed line (0-indexed)
    last_end: int = 0

    # Sort ranges to ensure they are processed in order
    for start, end, nesting_level in sorted(ranges):
        # Convert to 0-indexed for list operations
        zero_indexed_start: int = start - 1
        zero_indexed_end: int = end

        # Handle gap between last range and current range
        if zero_indexed_start > last_end:
            # Extract and store the gap content with nesting level 0
            gap_content: str = "".join(lines[last_end:zero_indexed_start])
            parts.append((gap_content, 0))

        # Process the current range
        # Extract and store the range content with its specified nesting level
        range_content: str = "".join(lines[zero_indexed_start:zero_indexed_end])
        parts.append((range_content, nesting_level))

        # Update the last processed line
        last_end = zero_indexed_end

    # Handle any remaining content after the last range
    if last_end < len(lines):
        # Extract and store the remaining content with nesting level 0
        remaining_content: str = "".join(lines[last_end:])
        parts.append((remaining_content, 0))

    return parts


@he
def extract_function_or_class_name(part: str) -> Optional[str]:
    """
    Extract the name of a function or class from a given string of Python code.

    This function uses a regular expression to find the name of a function or class
    definition, even if it's preceded by decorators. It supports both function and
    class definitions.

    Args:
        part (str): A string containing Python code, potentially including a function
            or class definition.

    Returns:
        Optional[str]: The name of the function or class if found, None otherwise.

    Note:
        The function assumes that the input string is well-formed Python code.
        It can handle multiple decorators and whitespace variations.
    """

    # search patten to see if there is a def function or class in here
    pattern: str = r"\b(def|class)\s+(\w+)"
    # \b: Adds a word boundary to ensure we match 'def' or 'class' as whole words.
    # (def|class): Matches and captures either 'def' or 'class'.
    # \s+: Matches one or more whitespace characters.
    # (\w+): Captures the name of the function or class.

    # Search for the pattern in the input string
    match: Optional[re.Match] = re.search(pattern, part, re.MULTILINE | re.DOTALL)

    # If a match is found, return the name (second captured group)
    if match:
        return match.group(2)  # group(2) contains the function or class name

    # If no match is found, return None
    return None


@he
def reassemble_code(modified_parts: List[Tuple[str, int]]) -> str:
    """
    Reassemble modified code parts into a single coherent code string.

    This function takes a list of modified code parts, each associated with a nesting level,
    and reassembles them into a single string of code. It handles indentation, function
    scopes, and avoids duplicate function definitions within the same or immediate parent scope.
    This duplicate function stuff is to deal with nested or inner functions.

    Args:
        modified_parts (List[Tuple[str, int]]): A list of tuples, where each tuple contains
            a string of modified code and its base nesting level.

    Returns:
        str: The reassembled code as a single string with proper indentation and structure.
    """
    full_modified_code: str = ""
    indent_stack: List[int] = [0]  # Stack to keep track of indentation levels
    function_stack: List[List[Tuple[str, int]]] = (
        []
    )  # Stack to keep track of function scopes

    for part, base_nesting_level in modified_parts:
        lines: List[str] = part.split("\n")
        part_lines: List[str] = []
        current_function: Optional[Tuple[str, int]] = None

        for line in lines:
            stripped_line: str = line.lstrip()

            if not stripped_line:
                part_lines.append("")  # Preserve empty lines
                continue

            # Check if this is a function definition
            if stripped_line.startswith("def "):
                func_signature: str = stripped_line.split("(")[
                    0
                ]  # Get function name and 'def' keyword
                current_function = (func_signature, base_nesting_level)

                # Check for duplicate in current scope or immediate parent scope
                if function_stack and (
                    current_function in function_stack[-1]
                    or (
                        len(function_stack) > 1
                        and current_function in function_stack[-2]
                    )
                ):
                    break  # Skip this part as it's a duplicate function in the same scope

                # Add the new function to the current scope
                if function_stack and function_stack[-1][0][1] < base_nesting_level:
                    function_stack[-1].append(current_function)
                else:
                    function_stack.append([current_function])

            # Calculate the indent of this line
            line_indent: int = len(line) - len(stripped_line)

            # Adjust indent stack based on the line's indentation
            while (
                len(indent_stack) > base_nesting_level + 1
                and line_indent <= indent_stack[-1]
            ):
                indent_stack.pop()
                if function_stack and function_stack[-1][0][1] >= len(indent_stack):
                    function_stack.pop()

            if line_indent > indent_stack[-1]:
                indent_stack.append(line_indent)

            # Calculate total indentation
            total_indent: str = "    " * (len(indent_stack) - 1)
            part_lines.append(f"{total_indent}{stripped_line}")

        # Only add the part if it wasn't skipped due to being a duplicate function
        if part_lines:
            full_modified_code += "\n".join(part_lines) + "\n"

        # Add an extra newline between parts if it's not top-level code
        if base_nesting_level > 0:
            full_modified_code += "\n"

    # Before returning, clean up newlines between decorators and function definitions
    pattern = r"(@\w+\s*)\n+\s*(@|def\s+|class\s+)"
    full_modified_code = re.sub(
        pattern, r"\1\n\2", full_modified_code, flags=re.MULTILINE
    )

    return full_modified_code


@he
def initial_arguments() -> tuple[argparse.Namespace, str, str]:
    """
    Parse command-line arguments for the script and return input/output file paths.

    This function sets up an argument parser to handle input and output file specifications.
    It defines two optional arguments: input file and output file, both with default values.

    Returns:
        tuple[argparse.Namespace, str, str]: A tuple containing three elements:
            - args (argparse.Namespace): The parsed command-line arguments.
            - input_file (str): The path to the input file (default: "input.py").
            - output_file (str): The path to the output file (default: "output.py").

    Raises:
        SystemExit: If invalid arguments are provided (handled by argparse).
    """
    # Create an ArgumentParser object with a description of the script's purpose
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Add type hints, comments and docstrings to python functions or classes."
    )

    # Add argument for input file with a short and long option
    parser.add_argument(
        "-f",
        "--input-file",
        type=str,
        default="input.py",
        help="The file containing the Python code to modify.",
    )

    # Add argument for output file with a short and long option
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.py",
        help="The file where we will write the suggested, modified code.",
    )

    # Parse the command-line arguments
    args: argparse.Namespace = parser.parse_args()

    # Extract input and output file paths from the parsed arguments
    input_file: str = args.input_file
    output_file: str = args.output_file

    # Return the parsed args and the input/output file paths as a tuple
    return args, input_file, output_file


@he
def main(input_file: str = "input.py", output_file: str = "output.py") -> None:
    """
    Process a Python file to add type hints, comments, and docstrings to functions or classes.

    This function serves as the entry point for the script. It handles command-line arguments,
    sets up logging, processes the input file, and generates an output file with enhanced code.
    The function uses the Anthropic API to generate AI-assisted improvements to the code.

    Args:
        input_file (str): Path to the input Python file to be processed. Defaults to "input.py".
        output_file (str): Path to the output file where modified code will be written. Defaults to "output.py".

    Returns:
        None

    Raises:
        SystemExit: If the input and output files are the same or if parsing fails.
    """
    # Set up logging for the current run
    logging.info("--- New run starting ---")
    logging.info(f"Processing file: {input_file}")

    # Set up command-line argument parsing
    args, input_file, output_file = initial_arguments()

    # Ensure input and output files are different to prevent overwriting
    if input_file == output_file:
        print("Error: Input and output files cannot be the same.")
        sys.exit(1)

    # Get a valid output file name, avoiding overwriting existing files
    output_file = get_valid_output_file(output_file)

    # Run Black formatter on the input file for consistent formatting
    run_black(input_file)

    # Read the source code from the input file
    with open(input_file, "r") as file:
        source_code: str = file.read()

    # Parse the Python file into an Abstract Syntax Tree (AST)
    tree: Optional[ast.AST] = parse_python_file(input_file)
    if not tree:
        print("Error: Parsing failed. Exiting.")
        sys.exit(1)

    # Set up Anthropic API for AI-assisted code improvement
    anthropic_api_key: str
    anthropic_object: Any
    anthropic_api_key, anthropic_object = setup_anthropic_stuff()

    # Get the ranges of functions and classes in the source code with nesting levels
    ranges_with_nesting: List[Tuple[ast.AST, str, int]] = (
        extract_functions_and_classes_with_text(tree, source_code)
    )

    # Convert to the format expected by split_content
    ranges: List[Tuple[int, int, int]] = [
        (node.lineno, node.end_lineno, nesting_level)
        for node, _, nesting_level in ranges_with_nesting
    ]

    # Split the content into parts based on the ranges of functions and classes
    parts: List[Tuple[str, int]] = split_content(source_code, ranges)

    # Initialize total cost for AI processing
    total_cost: float = 0.0

    # Process each part of the source code
    modified_parts: List[Tuple[str, int]] = []
    for i, (part, nesting_level) in enumerate(parts):
        logging.debug(f"Processing part {i}, nesting_level: {nesting_level}:")
        logging.debug(part)
        logging.debug("---")

        # Check if the part is a function or class
        if extract_function_or_class_name(part):
            logging.debug("It seems to be a function or class!")
        else:
            logging.debug(
                "It's not a function or class, so write it to modified_parts as is"
            )
            modified_parts.append((part, nesting_level))
            continue

        # Extract the original name of the function or class
        original_name: Optional[str] = extract_function_or_class_name(part)
        if not original_name:
            logging.warning(
                f"Couldn't read the original name in part {i}. Keeping the original content:\n{part}"
            )
            modified_parts.append((part, nesting_level))
            continue
        else:
            logging.info(f"Found original name: {original_name}")

        # Generate AI response for the current part
        answer: str
        cost: float
        answer, cost = generate_ai_response(
            ai_client_object=anthropic_object,
            ai_model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.5,
            system_message=(
                "You are a computer science professor dedicated to creating the perfect python code, "
                "especially by adding docstrings and type hints and inline comments to code. "
                "Do not add any new functions or classes. Only modify the existing code."
            ),
            prompt_string=(
                f"""
                Here is a python function or class definition, which may include nested functions. 
                Please provide ONLY the modified code with proper type hints, plenty of inline comments 
                and a thorough docstring. Maintain the original structure, including any nested functions. 

                Do not create standalone functions from nested functions. Do not add any new functions or 
                classes.  Do not add blank lines between decorators and function signatures.

                Only modify the existing code. Do not include any explanations or additional text before 
                or after the code. Be liberal with white space in the code, except where it might break
                things -- like between decorators and function definitions. Do not include any import 
                statements, or comment about them, they are there at the top of the file. Start directly 
                with the function or class definition and end with the last line of code. Preserve the 
                original function or class name and overall structure, including nested functions. Examples 
                are usually not needed in the docstrings. 

                Here's the code to modify:\n\n{part}
                """
            ),
        )

        # Update total cost of AI processing
        total_cost += cost

        # Verify and add the modified part
        if original_name not in answer:
            logging.warning(
                f"AI response for {original_name} doesn't contain the original name. Skipping modification."
            )
            modified_parts.append((part, nesting_level))
        else:
            logging.info(f"Successfully modified {original_name}")
            modified_parts.append((answer, nesting_level))

    # Reassemble the code maintaining nested structure
    full_modified_code: str = reassemble_code(modified_parts)

    # Write the modified code to the output file
    write_modified_code(full_modified_code, output_file)

    # Run Black formatter on the output file for consistent formatting
    run_black(output_file)

    # Print the total AI processing cost
    print(f"AI expense: ${total_cost:,.3f}")


if __name__ == "__main__":
    main()
