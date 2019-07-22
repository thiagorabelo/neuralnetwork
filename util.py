from __future__ import annotations

from typing import Union, List, TypeVar, Type, Callable, Text, Iterable, Tuple, Any


Number = Union[int, float]
MBase = TypeVar('MBase', bound='matrix.MatrixBase')
MatType = TypeVar('matrix.Matrix', bound='matrix.Matrix')


def match(mat1: MatType, mat2: MatType) -> bool:
    return mat2.rows == mat1.rows and mat2.cols == mat1.cols


def doest_match(mat1: MatType, mat2: MatType) -> ValueError:
    return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
        mat1.rows, mat1.cols,
        mat2.rows, mat2.cols,
    ))


def unexpected(other: Any) -> ValueError:
    return ValueError('Unexpected parameter of type %s' % type(other).__name__)


def matrix_op(left: MBase, right: MBase, operation: Callable[[Number, Number], Number], cls: MatType) -> MBase:
    if not match(left, right):
        raise doest_match(left, right)

    new_matrix = cls(right.rows, right.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return new_matrix


def scalar_op(left: MBase, scalar: Number, operation: Callable[[Number, Number], Number], cls: MatType) -> MBase:
    new_matrix = cls(left.rows, left.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), scalar))
    return new_matrix


def imatrix_op(left: MBase, right: MBase, operation: Callable[[Number, Number], Number]) -> MBase:
    if not match(left, right):
        raise doest_match(left, right)

    left.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return left


def iscalar_op(left: MBase, scalar: Number, operation: Callable[[Number, Number], Number]) -> MBase:
    left.imap(lambda val, row, col: operation(val, scalar))
    return left


del Number, MBase, MatType
