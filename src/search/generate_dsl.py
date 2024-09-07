import inspect
import importlib
from typing import get_type_hints


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


# Example usage:
# Assuming you have a module named 'mypackage.mymodule' with some functions
functions_info = get_functions_with_types_from_module('dsls.our_dsl.functions.dsl_functions')

# Print the function names, input types, and output types
for func_name, details in functions_info.items():
    print(f"Function Name: {func_name}")
    print(f"  Input Types: {details['input_types']}")
    print(f"  Output Type: {details['output_type']}")
    print()
