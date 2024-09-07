from nltk import CFG
from custom_generate import generate

grammar_str = """
Canvas -> "add_object_to_canvas (" Canvas "," Primitive  ")" | " make_new_canvas_as (" Canvas  ")" | "canvas"
Vector -> "get_distance_origin_to_origin_between_objects (" Primitive "," Primitive  ")" | " get_distance_touching_between_objects (" Primitive "," Primitive  ")"
Primitive -> "object_transform_translate_along_direction (" Primitive "," Vector  ")" | " select_object_of_colour (" Canvas "," int  ")"
int -> "0" | "1" 
"""


#print(grammar_str)
# Define a simple grammar
grammar = CFG.fromstring(
    grammar_str
)
#print(grammar.productions())

depth = 7


# from black import format_str, FileMode
# # Generate all possible sentences
def heuristic(s):
    return (("get_distance_origin_to_origin_between_objects") not in s )

print(len( list(generate(grammar,depth = depth, heuristic=heuristic))))

for sentence in generate(grammar,depth = depth, heuristic=heuristic):
    s = ' '.join(sentence)

    #out = format_str(s, mode=FileMode())

    #print(out)