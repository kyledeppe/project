
"""
STAT 636
Chunk 8B: Inspect null-model implementation.
"""

import inspect
import null_models


functions_to_inspect = [
    "generate_er",
    "generate_configuration",
    "generate_dcsbm",
    "generate_all_null_models",
    "validate_adjacency",
    "save_null_network",
    "assign_weights_to_edges",
    "extract_empirical_edges"
]

for name in functions_to_inspect:

    function = getattr(null_models, name)

    print("\n" + "=" * 65)
    print("FUNCTION:", name)
    print("=" * 65)

    print("SIGNATURE:")
    print(inspect.signature(function))

    print("\nSOURCE CODE:")

    try:
        print(inspect.getsource(function))

    except (OSError, TypeError):

        print("Source unavailable.")
