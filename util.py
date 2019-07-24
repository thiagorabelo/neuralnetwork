from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Union, Iterable, Iterator, List, Tuple, TypeVar


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


def matrix_op(left: MBase,
              right: MBase,
              operation: Callable[[Number, Number], Number],
              cls: MatType) -> MBase:

    if not match(left, right):
        raise doest_match(left, right)

    new_matrix = cls(right.rows, right.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return new_matrix


def scalar_op(left: MBase,
              scalar: Number,
              operation: Callable[[Number, Number], Number],
              cls: MatType) -> MBase:

    new_matrix = cls(left.rows, left.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), scalar))
    return new_matrix


def imatrix_op(left: MBase,
               right: MBase,
               operation: Callable[[Number, Number], Number]) -> MBase:

    if not match(left, right):
        raise doest_match(left, right)

    left.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return left


def iscalar_op(left: MBase,
               scalar: Number,
               operation: Callable[[Number, Number], Number]) -> MBase:

    left.imap(lambda val, row, col: operation(val, scalar))
    return left


def arange(start: Number, stop: Number = None, step: Number = 1.0) -> Iterable[Number]:
    if not step:
        raise ValueError('step can not be None or 0')

    if stop is None and start > 0.0:
        start, stop = 0.0, start
    elif stop is None and start < 0.0 < step:
        step *= -1
        start, stop = 0.0, start
    elif start > stop and step > 0.0:
        step *= -1

    if start < stop:
        while start < stop:
            yield start
            start += step
    else:
        while start > stop:
            yield start
            start += step


def divide_arange(start: Number, stop: Number, slices: Number) -> Iterable[Number]:
    step = (stop - start) / abs(slices)
    return arange(start, stop, step)


def enumerate_reversed(a_list: List[Number]) -> Union[Iterable[Tuple[int, Number]],
                                                      Iterator[Tuple[int, Number]]]:
    return zip(range(len(a_list) - 1, -1, -1), reversed(a_list))


def clip(min_val: Number, max_val: Number) -> \
        Callable[[Number, Number], Callable[[Number], Number]]:
    def decorator(func: Callable[[Number], Number]) -> Callable[[Number], Number]:
        @wraps(func)
        def wrapper(value: Number) -> Number:
            return func(min_val if value < min_val else
                        max_val if value > max_val else
                        value)
        return wrapper
    return decorator


del Number, MBase, MatType
