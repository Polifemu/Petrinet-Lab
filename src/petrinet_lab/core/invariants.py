"""Structural analysis: incidence matrix, P/T invariants and structural properties.

Invariants are computed exactly over the rationals with a pure-Python
Gauss-Jordan elimination, so no external linear-algebra dependency is needed.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import gcd

from .model import PetriNet

Vector = list[Fraction]
Matrix = list[list[Fraction]]


def _rref(matrix: Matrix) -> tuple[Matrix, list[int]]:
    rows = [row[:] for row in matrix]
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    pivots: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (i for i in range(pivot_row, row_count) if rows[i][column] != 0), None
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for i in range(row_count):
            if i != pivot_row and rows[i][column] != 0:
                factor = rows[i][column]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[pivot_row], strict=False)]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return rows, pivots


def null_space(matrix: Matrix) -> list[Vector]:
    if not matrix or not matrix[0]:
        return []
    reduced, pivots = _rref(matrix)
    free_columns = [c for c in range(len(matrix[0])) if c not in pivots]
    basis: list[Vector] = []
    for free in free_columns:
        vector = [Fraction(0)] * len(matrix[0])
        vector[free] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free]
        basis.append(vector)
    return basis


def _to_integer_vector(vector: Vector) -> list[int]:
    denominator = 1
    for value in vector:
        denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    integers = [int(value * denominator) for value in vector]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    return [value // divisor for value in integers] if divisor else integers


def _normalised_positive(vector: Vector) -> tuple[int, ...] | None:
    integers = _to_integer_vector(vector)
    if all(value <= 0 for value in integers):
        integers = [-value for value in integers]
    if any(value < 0 for value in integers):
        return None
    if all(value == 0 for value in integers):
        return None
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, value)
    return tuple(value // divisor for value in integers)


def _canonical_invariants(basis: list[Vector], limit_coefficient: int = 3) -> list[tuple[int, ...]]:
    candidates: set[tuple[int, ...]] = set()
    for vector in basis:
        normalised = _normalised_positive(vector)
        if normalised is not None:
            candidates.add(normalised)
    coefficient_range = [c for c in range(-limit_coefficient, limit_coefficient + 1) if c != 0]
    for first, second in combinations(basis, 2):
        for a, b in product(coefficient_range, repeat=2):
            normalised = _normalised_positive([a * x + b * y for x, y in zip(first, second, strict=False)])
            if normalised is not None:
                candidates.add(normalised)
    minimal: list[tuple[int, ...]] = []
    for candidate in sorted(candidates, key=lambda v: (sum(1 for x in v if x), sum(v), v)):
        support = {i for i, value in enumerate(candidate) if value}
        if any(support < {i for i, value in enumerate(other) if value} for other in minimal):
            continue
        if any(support == {i for i, value in enumerate(other) if value} for other in minimal):
            continue
        minimal.append(candidate)
    return minimal


def incidence_matrix(net: PetriNet) -> Matrix:
    return [[Fraction(value) for value in row] for row in net.incidence_matrix()]


def p_invariants(net: PetriNet) -> list[dict[str, int]]:
    matrix = incidence_matrix(net)
    if not matrix:
        return []
    transposed = [[row[j] for row in matrix] for j in range(len(matrix[0]))]
    basis = null_space(transposed)
    return [
        {pid: value for pid, value in zip(net.place_ids, vector, strict=False) if value}
        for vector in _canonical_invariants(basis)
    ]


def t_invariants(net: PetriNet) -> list[dict[str, int]]:
    matrix = incidence_matrix(net)
    basis = null_space(matrix)
    return [
        {tid: value for tid, value in zip(net.transition_ids, vector, strict=False) if value}
        for vector in _canonical_invariants(basis)
    ]


def is_conservative(net: PetriNet) -> bool:
    return any(all(value > 0 for value in invariant.values()) for invariant in p_invariants(net))


def is_repetitive(net: PetriNet) -> bool:
    return any(all(value > 0 for value in invariant.values()) for invariant in t_invariants(net))


def structural_properties(net: PetriNet) -> dict[str, object]:
    p_inv = p_invariants(net)
    t_inv = t_invariants(net)
    return {
        "p_invariants": p_inv,
        "t_invariants": t_inv,
        "conservative": any(all(v > 0 for v in inv.values()) for inv in p_inv),
        "repetitive": any(all(v > 0 for v in inv.values()) for inv in t_inv),
        "structurally_bounded": any(all(v > 0 for v in inv.values()) for inv in p_inv),
    }
