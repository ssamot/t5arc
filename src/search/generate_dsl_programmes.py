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

depth = 6


from black import format_str, FileMode
# # Generate all possible sentences
def heuristic(s):
    # n = len(s)
    # x = 10
    #
    # for i in range(n):
    #     for j in range(i + x + 1, n - x + 1):
    #         substring = s[i:j]
    #         if len(substring) > x and s.count(substring) >= 3:
    #             #print(substring)
    #             return False

    return True

print(len( list(generate(grammar,depth = depth, heuristic=heuristic))))

for sentence in generate(grammar,depth = depth, heuristic=heuristic):
    s = ' '.join(sentence)
    #print(s)

    out = format_str(s, mode=FileMode())

    #print(out)