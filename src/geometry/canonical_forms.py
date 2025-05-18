#!/usr/bin/env python3
"""
canonical_forms.py

Defines the 15 canonical degree-2 quadratic mappings used
for generating and warping image grids. Each mapping is
registered with a unique integer key in the `FORMS` dict.

Functions:
  - get_form(id):    Retrieve the mapping function by ID.
  - random_form():   Sample a random mapping from the set.

Mapping signature: fn(x, y) -> (x', y'). Operates elementwise on coordinates.
"""


import torch
import random

# Dictionary to store transformation functions keyed by their IDs
FORMS = {}

def _register_2(id_, fn):
    """
    Registers a transformation function `fn` with a specific integer ID.

    Args:
        id_ (int): Unique identifier for the transformation.
        fn (callable): A function that takes two arguments (x, y) and returns a tuple of two values.
    """
    FORMS[id_] = fn

# Registering various 2D transformations by assigning each a unique ID.
# Each transformation is a lambda function that maps (x, y) to a new (x', y') based on different mathematical forms.

_register_2(1, lambda x, y: (x**2 + y, y**2 + x))
_register_2(2, lambda x, y: (x**2 - y**2 + x, 2 * x * y - y))
_register_2(3, lambda x, y: (x**2 + y, x * y))
_register_2(4, lambda x, y: (x**2 + y, y**2))
_register_2(5, lambda x, y: (x**2, y**2))
_register_2(6, lambda x, y: (x**2 - y**2, x * y))
_register_2(7, lambda x, y: (x**2 - x, x * y))
_register_2(8, lambda x, y: (x**2, x * y))
_register_2(9, lambda x, y: (x * y, x + y))
_register_2(10, lambda x, y: (x**2 + y**2, x))
_register_2(11, lambda x, y: (x, x * y))
_register_2(12, lambda x, y: (x**2, y))
_register_2(13, lambda x, y: (x**2 + y, x))
_register_2(14, lambda x, y: (x**2, x))
_register_2(15, lambda x, y: (x, y))  # Identity transformation
#_register_2(16, lambda x, y: (x * y, 0 * x))  # Zero second output
#_register_2(17, lambda x, y: (x**2 + y**2, 0 * x))
#_register_2(18, lambda x, y: (x**2 + y, 0 * x))
#_register_2(19, lambda x, y: (x**2, 0 * x))
#_register_2(20, lambda x, y: (x, 0 * x))
#_register_2(21, lambda x, y: (0 * x, 0 * x))  # Constant zero output

def get_form(id_: int):
    """
    Retrieves a transformation function by its ID.

    Args:
        id_ (int): ID of the desired transformation.

    Returns:
        callable: The transformation function associated with the ID.
    """
    return FORMS[id_]

def random_form(rng=None):
    """
    Selects and returns a random transformation function from the registered set.

    Args:
        rng (module, optional): A random number generator module (default is Python's built-in `random`).

    Returns:
        callable: A randomly selected transformation function.
    """
    rng = rng or random
    id_ = rng.choice(list(FORMS))
    return get_form(id_)
