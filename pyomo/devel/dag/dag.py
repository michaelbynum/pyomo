# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


"""
The purpose of this module is to identify common subexpressions/nodes and make
sure they are shared across constraints.

For efficiency, we only store a hash for each node. If we find two nodes 
with the same hash, then we build the prefix notation to ensure that 
they are indeed the same.

To build the hash, we hash a tuple containing the expression type and 
the hashes for each of the arguments, sorted. If the node is a variable, then
we hash a tuple containing the expression type (var) and the id of the 
variable.
"""

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
import operator
from hashlib import sha512


_var = 'variable'.encode()
_param = 'parameter'.encode()


def compare_expressions(e1, e2) -> bool:
    raise NotImplementedError('This should not be needed.')


def _handle_var(node, data, hash_node_map):
    hasher = sha512()
    hasher.update(_var)
    hasher.update(str(id(node)).encode())
    h = hasher.digest()
    if h not in hash_node_map:
        hash_node_map[h] = (node,)
    elif node is hash_node_map[h][0]:
        pass
    else:
        # hash collision; fail loudly
        raise RuntimeError('unexpected hash collision')
        # hash_node_map[h] += (node,)
    return h, node, [node]


def _handle_node(node, data, hash_node_map, op):
    hasher = sha512()
    hasher.update(str(op).encode())
    for arg_hash, arg, arg_rpn in data:
        hasher.update(arg_hash)
    h = hasher.digest()
    replace = None
    if h in hash_node_map:
        for other in hash_node_map[h]:
            if compare_expressions(node, other):
                replace = other
                break
    if replace is None:
        # we need to re-form the node in case any of the 
        # arguments were replaced
        op_args = tuple(i[0] for i in data)
        replace = op(*op_args)
        if h in hash_node_map:
            hash_node_map[h] = hash_node_map[h] + (replace,)
        else:
            hash_node_map[h] = (replace,)
    return h, replace


def _handle_sum(node)


class _DAGVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.hash_node_map = {}

    def exitNode(self, node, data):
        """
        data should be a tuple of tuples containing the hashes of the arguments 
        and the arguments.
        """
