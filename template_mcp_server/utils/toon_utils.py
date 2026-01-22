"""TOON (Token Oriented Object Notation) utility functions.

This module provides utilities for converting Python objects to TOON format,
which reduces token usage by 30-60% compared to JSON while maintaining
full data fidelity and human readability.
"""

import toon_format


def to_toon(data: dict | list | str | int | float | bool | None) -> str:
    r"""Convert Python data structure to TOON format string.

    Args:
        data: Python object to convert (dict, list, or primitive types)

    Returns:
        str: TOON-formatted string representation

    Examples:
        >>> to_toon({"success": True, "count": 42})
        'success: true\ncount: 42'

        >>> to_toon([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        '[2]{id,name}:\n  1,Alice\n  2,Bob'

        >>> to_toon({"sql": "SELECT *\nFROM users"})
        'sql: "SELECT *\\nFROM users"'
    """
    return toon_format.encode(data)


def from_toon(toon_str: str) -> dict | list | str | int | float | bool | None:
    r"""Parse TOON format string back to Python data structure.

    Args:
        toon_str: TOON-formatted string

    Returns:
        Python object (dict, list, or primitive type)

    Examples:
        >>> from_toon('success: true\ncount: 42')
        {'success': True, 'count': 42}

    """
    return toon_format.decode(toon_str)
