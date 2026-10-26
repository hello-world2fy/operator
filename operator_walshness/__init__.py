"""Operator Walshness, round-1 convention.

The fixed-basis quantity is the Walsh free energy of the 2n-bit vectorization
|O>> = sum_{s,t} O(s,t) |s>⊗|t|, with degree |r|+|q|. The intrinsic quantity
minimizes that value over physical local basis changes O -> U† O U, U=⊗U_i.
Numerical minima are upper bounds on the intrinsic value, except at α→0 where
the minimum is the closed form sum_i (1-|m_i(ρ²)|).
"""

from .walsh import (
    degree_histogram,
    fixed_basis_spectrum,
    identity_walshness,
    walshness_from_spectrum,
)

__all__ = [
    "degree_histogram",
    "fixed_basis_spectrum",
    "identity_walshness",
    "walshness_from_spectrum",
]
