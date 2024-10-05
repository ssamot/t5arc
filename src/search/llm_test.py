from llama_cpp import Llama
import numpy as np
from keras import ops
model_path="/Users/ssamot/Downloads/codegeex4-all-9b-IQ2_M.gguf"

llm = Llama(
      model_path=model_path,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

grammar_str = """
Canvas -> "add_object_to_canvas (" Canvas "," Primitive  ")" | " make_new_canvas_as (" Canvas  ")" | "canvas"
    Vector -> "get_distance_origin_to_origin_between_objects (" Primitive "," Primitive  ")" | " get_distance_touching_between_objects (" Primitive "," Primitive  ")"
    Primitive -> "object_transform_translate_along_direction (" Primitive "," Vector  ")" | " select_object_of_colour (" Canvas "," int  ")"
    int -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10"
"""
output = llm(
      f"Q: You have the following grammar {grammar_str} -- generate a random programme using  A: ", # Prompt
      max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)