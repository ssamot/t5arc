# Natural Language Toolkit: Generating from a CFG
#
# Copyright (C) 2001-2024 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
#         Eric Kafe <kafe.eric@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

import itertools
import sys

from nltk.grammar import Nonterminal


def generate(grammar, start=None, depth=None, n=None, heuristic = None):
    """
    Generates an iterator of all sentences from a CFG.

    :param grammar: The Grammar used to generate sentences.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.
    :return: An iterator of lists of terminal tokens.
    """
    if not start:
        start = grammar.start()
    if depth is None:
        # Safe default, assuming the grammar may be recursive:
        depth = (sys.getrecursionlimit() // 3) - 3



    iter = _generate_all(grammar, [start], depth, heuristic)

    if n:
        iter = itertools.islice(iter, n)

    return iter


def _generate_all(grammar, items, depth, heuristic):
    if items:
        try:
            for frag1 in _generate_one(grammar, items[0], depth, heuristic):
                for frag2 in _generate_all(grammar, items[1:], depth, heuristic):
                    s =  ' '.join(frag1 +  frag2)
                    if(heuristic is not None):
                        if(heuristic(s)):
                            yield frag1 + frag2
                        else:
                            break
                    else:
                        yield frag1 + frag2

        except RecursionError as error:
            # Helpful error message while still showing the recursion stack.
            raise RuntimeError(
                "The grammar has rule(s) that yield infinite recursion!\n\
Eventually use a lower 'depth', or a higher 'sys.setrecursionlimit()'."
            ) from error
    else:
        yield []


def _generate_one(grammar, item, depth, heuristic):
    if depth > 0:
        if isinstance(item, Nonterminal):
            for prod in grammar.productions(lhs=item):
                yield from _generate_all(grammar, prod.rhs(), depth - 1, heuristic)
        else:
            yield [item]
