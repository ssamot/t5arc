import inspect
import importlib
from typing import get_type_hints
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path



def get_functions_with_types_from_module(module_name):
    # Import the module
    module = importlib.import_module(module_name)

    # Initialize a dictionary to store function details
    functions_dict = {}

    # Get all functions in the module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Get the function signature
        sig = inspect.signature(func)

        # Get input types (parameter types)
        input_types = {param.name: param.annotation for param in sig.parameters.values()}

        # Get output type (return type)
        output_type = sig.return_annotation

        # Handle cases where type hints are provided using the typing module
        type_hints = get_type_hints(func)
        input_types.update({param: type_hints.get(param, input_types[param]) for param in input_types})
        output_type = type_hints.get('return', output_type)

        # Store the function, input types, and output type in the dictionary
        functions_dict[name] = {
            'function': func,
            'input_types': input_types,
            'output_type': output_type
        }

    return functions_dict


@click.command()
@click.argument('problem', type=click.STRING())
@click.argument('output_filepath', type=click.Path())
def main(problem, output_filepath):
    location = "dsls.our_dsl.functions.dsl_functions"
    functions_info = get_functions_with_types_from_module(location)



    for func_name, details in functions_info.items():
        print(f"Function Name: {func_name}")
        print(f"  Input Types: {details['input_types']}")
        print(f"  Output Type: {details['output_type']}")
        print()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()



# Example usage:
# Assuming you have a module named 'mypackage.mymodule' with some functions

# Print the function names, input types, and output types

