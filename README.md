# BetterPython
BetterPython is an AI-assisted Python code enhancement tool that automatically improves your Python code by adding type hints, comments and docstrings to functions and classes.
## Features
- Automated code enhancement using AI (Anthropic API)
- Addition of type hints to function parameters and return values
- Generation of comprehensive docstrings for functions and classes
- Insertion of inline comments to explain complex logic
- Preservation of original code structure and functionality
- Handling of nested functions and classes
- Custom error handling and logging for robust execution
- Integration with Black formatter for consistent code styling
## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/BetterPython.git
   cd BetterPython
   ```
2. Install the required dependencies:
   ```
   pip install anthropic black
   ```
3. Set up your Anthropic API key:
   - Create a file named `anthropic_key.txt` in the project root directory
   - Paste your Anthropic API key into this file

   As an alternative, which is possibly better, you can set an environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```
## Usage
Run the script from the command line:
python BetterPython.py -f <input_file> -o <output_file>
 `-f` or `--input-file`: Path to the Python file you want to enhance (default: "input.py")
 `-o` or `--output-file`: Path where the enhanced code will be saved (default: "output.py")
Example:
python BetterPython.py -f my_script.py -o enhanced_script.py
## How It Works
0. IMPORTANT -- the script runs the black formatter on the SOURCE file.  So if you don't want your source file changed, make a copy and run this script on the copy.
1. The script parses the input Python file into an Abstract Syntax Tree (AST).
2. It extracts top-level functions and classes from the AST.
3. Each function or class is sent to the Anthropic API for enhancement.
4. The AI adds type hints, improves docstrings, and inserts helpful comments.
5. The enhanced code snippets are reassembled while maintaining the original structure.
6. The Black formatter is applied to the output, to ensure consistent code style and whitespace, which the AI messes up sometimes.
7. The final enhanced code is written to the output file.
## Configuration
You can modify the following parameters in the script:
- `ai_model`: The Anthropic AI model to use (default: "claude-3-5-sonnet-20240620")
- `max_tokens`: Maximum number of tokens for AI responses (default: 4096)
- `temperature`: AI response randomness (0.0 to 1.0, default: 0.5)
## Limitations
- The AI enhancements are based on the Anthropic model's understanding and may not always be perfect.  Check the resulting code CAREFULLY.  It seems to be especially flaky around nested functions and decorators.
- Very large files may need to be processed in chunks due to API limitations.
- The tool currently focuses on top-level functions and classes.
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.  If I don't notice, reach out to me at rick aat bunker dott us.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Acknowledgments
- This tool uses the Anthropic API for AI-assisted code enhancement.
- The Black formatter is used for consistent code styling.
## Disclaimer
This tool is provided as-is, and while it aims to improve code quality, it's vitally important to back up your file before proceeding, and to review the black and AI-generated changes carefully.
### Custom Error Handling and Logging
BetterPython uses a custom decorator `he` to enhance function execution with logging and error handling. This decorator does some stuff well, but has limitations.  The program will create BetterPython.log with logging and error decorator output.
### Obtaining an Anthropic API Key
To use BetterPython, you will need an Anthropic API key. You can obtain this key by signing up on the Anthropic website.
