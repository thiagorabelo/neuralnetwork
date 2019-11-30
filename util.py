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


class Scaler:
    def __init__(self,
                 values: Iterable[Number],
                 scalar_min: Number = 0.0,
                 scalar_max: Number = 1.0):
        self.min: Number = min(values)
        self.max: Number = max(values)
        self.smin: Number = scalar_min
        self.smax: Number = scalar_max
        self.range: Number = self.max - self.min
        self.srange: Number = scalar_max - scalar_min

    def scale(self, value: Number) -> Number:
        return (((value - self.min) * self.srange) / self.range) + self.smin

    def reverse(self, value: Number) -> Number:
        return (((value - self.smin) * self.range) / self.srange) + self.min

    def scale_all(self, values: Iterable[Number]) -> Iterable[Number]:
        return [self.scale(val) for val in values]

    def reverse_all(self, values: Iterable[Number]) -> Iterable[Number]:
        return [self.reverse(val) for val in values]


class TrainSetScaler:
    def __init__(self,
                 train_set: Iterable[Union[List[List[Number]],
                                     Tuple[List[Number], List[Number]]]],
                 scalar_min: Number = 0.0,
                 scalar_max: Number = 1.0):
        train_set_len = len(train_set)
        input_len = len(train_set[0][0])
        target_len = len(train_set[0][1])

        input_layer = [[0] * train_set_len for _ in range(input_len)]
        target_layer = [[0] * train_set_len for _ in range(target_len)]

        for train_set_idx, (input_data, target_output) in enumerate(train_set):
            for idx in range(input_len):
                input_layer[idx][train_set_idx] = input_data[idx]

            for idx in range(target_len):
                target_layer[idx][train_set_idx] = target_output[idx]

        self.input_scalers: List[Scaler] = [Scaler(input_data, scalar_min, scalar_max)
                                            for input_data in input_layer]
        self.target_scalers: List[Scaler] = [Scaler(target_data, scalar_min, scalar_max)
                                             for target_data in target_layer]

    @staticmethod
    def _scale(scalers, values):
        return [scaler.scale(val) for scaler, val in zip(scalers, values)]

    @staticmethod
    def _reverse(scalers, values):
        return [scaler.reverse(val) for scaler, val in zip(scalers, values)]

    def scale_input(self, input_values):
        return self._scale(self.input_scalers, input_values)

    def scale_target(self, target_values):
        return self._scale(self.target_scalers, target_values)

    def reverse_input(self, input_values):
        return self._reverse(self.input_scalers, input_values)

    def reverse_target(self, target_values):
        return self._reverse(self.target_scalers, target_values)

    def scale_train_set(self, train_set):
        return tuple(
            (self.scale_input(i), self.scale_target(t))
            for i, t in train_set
        )
