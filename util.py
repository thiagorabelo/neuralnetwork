from functools import wraps
from typing import Any, Callable, Union, Iterable, Iterator, List, Tuple, TypeVar


Number = TypeVar('Number', bound='numbers.Number')
MatBaseType = TypeVar('MatBaseType', bound='matrix.MatrixBase')
MatType = TypeVar('MatType', bound='matrix.Matrix')
MatProxyType = TypeVar('MatProxyType', bound='matrix.Matrix')
ColType = TypeVar('ColType', bound='matrix.Col')
RowType = TypeVar('RowType', bound='matrix.Row')


def match(mat1: MatType, mat2: MatType) -> bool:
    return mat2.rows == mat1.rows and mat2.cols == mat1.cols


def doesnt_match(mat1: MatType, mat2: MatType) -> ValueError:
    return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
        mat1.rows, mat1.cols,
        mat2.rows, mat2.cols,
    ))


def unexpected(other: Any) -> ValueError:
    return ValueError('Unexpected parameter of type %s' % type(other).__name__)


def matrix_op(left: MatBaseType,
              right: MatBaseType,
              operation: Callable[[Number, Number], Number],
              cls: MatType) -> MatBaseType:

    if not match(left, right):
        raise doesnt_match(left, right)

    new_matrix = cls(right.rows, right.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return new_matrix


def scalar_op(left: MatBaseType,
              scalar: Number,
              operation: Callable[[Number, Number], Number],
              cls: MatType) -> MatBaseType:

    new_matrix = cls(left.rows, left.cols)
    # new_matrix.data[:] = map(lambda val: operation(val, scalar), left.data)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), scalar))
    return new_matrix


def imatrix_op(left: MatBaseType,
               right: MatBaseType,
               operation: Callable[[Number, Number], Number]) -> MatBaseType:

    if not match(left, right):
        raise doesnt_match(left, right)

    left.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return left


def iscalar_op(left: MatBaseType,
               scalar: Number,
               operation: Callable[[Number, Number], Number]) -> MatBaseType:

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
