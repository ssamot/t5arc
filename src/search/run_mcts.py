from random import choice
from search.monte_carlo_tree_search import MCTS, Node
from nltk import CFG
from nltk.grammar import Nonterminal
import numpy as np
from black import format_str, FileMode
import itertools
from nltk.parse.generate import generate



grammar_str = """
Canvas -> "add_object_to_canvas (" Canvas "," Primitive  ")" | " make_new_canvas_as (" Canvas  ")" | "canvas"
Vector -> "get_distance_origin_to_origin_between_objects (" Primitive "," Primitive  ")" | " get_distance_touching_between_objects (" Primitive "," Primitive  ")"
Primitive -> "object_transform_translate_along_direction (" Primitive "," Vector  ")" | " select_object_of_colour (" Canvas "," int  ")"
int -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10"
"""

grammar = CFG.fromstring(
    grammar_str
)


class GrammarNode(Node):

    def __init__(self, current_programme, grammar, depth, max_depth):

        self.grammar = grammar
        self.current_programme = current_programme
        self.depth = depth
        self.max_depth = max_depth





        #print(self.to_pretty_string())

        if(self.is_terminal() and depth > 2):
            print("TERMINAL")
            print(self.to_pretty_string())
            print("END TERMINAL")
            exit()




    def find_children(self, random = False):

        if self.is_terminal():
            return set()

        nonterminals = []
        nonterminal_positions = []
        for i,symbol in enumerate(self.current_programme):
            NT = isinstance(symbol, Nonterminal)
            if(NT):
                nonterminals.append(symbol)
                nonterminal_positions.append(i)





        if (self.depth + 1 >= self.max_depth):
            exploded = []
            for nt in nonterminals:
                for max_depth in range(1, 100):
                    full_products_max_depth = list(generate(self.grammar, nt,  depth = max_depth))

                    if(full_products_max_depth!=[]):
                        final = []
                        for element in full_products_max_depth:
                            final.append(" ".join(element))
                        break
                exploded.append(final)
        else:
            exploded = []
            for nt in nonterminals:
                exploded.append([p.rhs() for p in grammar.productions(nt)])

        if(random):
            products = [[choice(t) for t in exploded]]
        else:
            products = itertools.product(*exploded)


        actions = []
        for p in products:
            action = list(self.current_programme[:])
            for i,nt in enumerate(nonterminal_positions):
                action[nt] = p[i]
            new_action = []
            for x in action:
                if(type(x) == tuple):
                    new_action.extend(x)
                else:
                    new_action.append(x)
            actions.append(new_action)

        children = [GrammarNode(s,self.grammar, self.depth+1, self.max_depth) for s in actions]
        print("Len Children", len(children), self.depth)
        return children

    def find_random_child(self):
        return choice(self.find_children(True))

    def reward(self):
        if not self.is_terminal():
            raise RuntimeError(f"reward called on nonterminal board node")

        return np.random.random()

        ###### geoooorge!


    def is_terminal(self):

        to_be_expanded = []
        for symbol in self.current_programme:
                NT = isinstance(symbol, Nonterminal)
                if (NT):
                    to_be_expanded.append(symbol)
        if(to_be_expanded == []):
            return True
        else:
            return False


    def to_pretty_string(self):
        converted = [str(e) for e in self.current_programme]
        out = format_str(" ".join(converted), mode=FileMode())
        return out


    def __hash__(self):
        return hash(" ".join(str(self.current_programme)))

    def __eq__(node1, node2):
        string1 = " ".join(" ".join(str(node1.current_programme)))
        string2 = " ".join(" ".join(str(node2.current_programme)))
        return string1 == string2

def play_game():
    tree = MCTS()

    board = GrammarNode([grammar.start()], grammar, 0, max_depth=10 )

    for _ in range(50):
        tree.do_rollout(board)


if __name__ == "__main__":
    play_game()