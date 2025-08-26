"""Secure random number generation utilities."""

import secrets
from typing import List, Union


def secure_random() -> float:
    """Generate cryptographically secure random float [0.0, 1.0).

    Returns:
        Secure random float
    """
    return secrets.randbits(32) / (2**32)


def secure_randint(a: int, b: int) -> int:
    """Generate cryptographically secure random integer in [a, b].

    Args:
        a: Lower bound (inclusive)
        b: Upper bound (inclusive)

    Returns:
        Secure random integer
    """
    return secrets.randbelow(b - a + 1) + a


def secure_choice(sequence: List[Union[int, float, str]]) -> Union[int, float, str]:
    """Securely choose random element from sequence.

    Args:
        sequence: Sequence to choose from

    Returns:
        Random element from sequence
    """
    if not sequence:
        raise ValueError("Cannot choose from empty sequence")

    index = secrets.randbelow(len(sequence))
    return sequence[index]


def secure_shuffle(sequence: List) -> List:
    """Securely shuffle sequence in place.

    Args:
        sequence: List to shuffle

    Returns:
        Shuffled list (same object)
    """
    for i in range(len(sequence) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence


# For non-cryptographic use cases where performance matters more than security
import random as _stdlib_random


def fast_random() -> float:
    """Generate fast (non-secure) random float for performance-critical code.

    Note: Use only where cryptographic security is not required.

    Returns:
        Random float
    """
    return _stdlib_random.random()


def fast_randint(a: int, b: int) -> int:
    """Generate fast (non-secure) random integer for performance-critical code.

    Note: Use only where cryptographic security is not required.

    Args:
        a: Lower bound (inclusive)
        b: Upper bound (inclusive)

    Returns:
        Random integer
    """
    return _stdlib_random.randint(a, b)


def fast_choice(sequence: List[Union[int, float, str]]) -> Union[int, float, str]:
    """Fast (non-secure) choice for performance-critical code.

    Note: Use only where cryptographic security is not required.

    Args:
        sequence: Sequence to choose from

    Returns:
        Random element from sequence
    """
    return _stdlib_random.choice(sequence)
