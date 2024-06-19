import os
import re
import logging
import click
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import ast


def cleanup(input_string):
    # Split the string into lines
    lines = input_string.split('\n')

    # Remove the first line
    lines = lines[1:]

    # Remove leading whitespace (spaces and tabs) from each line
    lines = [line.lstrip() for line in lines]

    # Join the lines back into a single string
    result_string = '\n'.join(lines)

    return result_string


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_directory', type=click.Path())
def main(input_file, output_directory):

    # Read the content of the input file
    with open(input_file, 'r') as f:
        content = f.read()

    with open(input_file, 'r') as f:
        content = f.read()

    # Parse the content into an AST
    tree = ast.parse(content)

    class FunctionExtractor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            if node.name.startswith("solve_"):
                # Extract the function signature
                function_signature = node.name.split('_')[1]
                # Capture the function body source code
                function_body = ast.get_source_segment(content, node)
                # Rename the function to 'solve'
                function_body = function_body.replace(node.name, 'solve', 1)
                self.functions.append((function_signature, function_body))
            self.generic_visit(node)

    # Extract functions from the AST
    extractor = FunctionExtractor()
    extractor.visit(tree)

    # Write each function to a separate file


    for function_signature, function_body in extractor.functions:
        function_body = cleanup(function_body)
        #print(function_body)
        #exit()
        output_file_path = os.path.join(output_directory, f'{function_signature}.py')
        with open(output_file_path, 'w') as output_file:
            output_file.write(function_body)

    print('Functions have been extracted and saved to separate files.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()