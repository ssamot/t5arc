import inspect
import importlib
from typing import get_type_hints
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from collections import defaultdict



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
@click.argument('problem', type=click.STRING)
@click.argument('output_filepath', type=click.Path())
def main(problem, output_filepath):
    location = "dsls.our_dsl.functions.dsl_functions"
    functions_info = get_functions_with_types_from_module(location)



    test_functions = [
        "select_object_of_colour",
        "get_distance_touching_between_objects",
        "object_transform_translate_along_direction",
        "make_new_canvas_as",
        "add_object_to_canvas",
        "get_distance_origin_to_origin_between_objects",
        "select_rest_of_the_objects",
    ]


    function_per_output_type = defaultdict(lambda:  [])

    for func_name, details in functions_info.items():
        #print(details)

        if(func_name in test_functions):
            it_l = list(details['input_types'].values())
            #out_l = list(details['output_type'].values())
            #input_types = [t for input_types  ]
            input_types = []
            output_type = details['output_type'].__name__
            if(output_type == "List"):
                obj_type = str(details['output_type']).split(".")[-1]
                output_type = f"{output_type}[{obj_type}"

            for it in it_l:
                #print((it.__name__))
                it = it.__name__
                input_types.append(it)

            function_per_output_type[output_type].append([func_name, input_types])

    function_per_output_type["Canvas"].append(["literal", ["\"canvas\""]])
    function_per_output_type["int"].append(["literal", [f"\"{i}\"" for i in range(11)]])

    def get_init_string(typedata):
        name, params = typedata

        if(name == "literal"):
            params = " | ".join(params)
        else:
            params = " \",\" ".join(params)


        return name, type, params




    for type in function_per_output_type.keys():
        #print(type, function_per_output_type[type])

        name, type, params = get_init_string(function_per_output_type[type][0])
        if(name == "literal"):
            init_string = f"{type} -> {params} "
        else:
            init_string = f"{type} -> \"{name} (\" {params}  \")\""

        #print(init_string)



        for funcs in function_per_output_type[type][1:]:
            name, type, params = get_init_string(funcs)
            if(name == "literal"):
                init_string += f" | {params} "
            else:
                init_string += f" | \" {name} (\" {params}  \")\""







        print(init_string)

   # print(function_per_output_type)
            #print(type(details['input_types']))
            # print(f"Function Name: {func_name}")
            # print(f"  Input Types: {input_types}")
            # print(f"  Output Type: {output_types}")
            # print()


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

