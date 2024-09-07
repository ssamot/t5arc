from nltk import CFG
from nltk.parse.generate import generate

grammar_str = """
    S -> function
    function -> "f(" function ")"  | "g(" function ")" | terminal
    terminal -> "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" |"9"

"""

# grammar_str ="""
#   S -> A B
#   A -> B
#   B -> "b" | A
#  """

print(grammar_str)
# Define a simple grammar
grammar = CFG.fromstring(
    grammar_str
)
#print(grammar.productions())

# Generate all possible sentences
for sentence in generate(grammar, depth=4):
    print(' '.join(sentence))