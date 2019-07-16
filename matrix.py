from __future__ import annotations

import numbers
import random
from typing import Union, List, TypeVar, Type, Callable, Text, Iterable, Tuple


Number = Union[int, float]
MBase = TypeVar('MBase', bound='MatrixBase')


class Row:
    def __init__(self, matrix: Matrix, row: int) -> None:
        self.matrix = matrix
        self.row = row

    def __getitem__(self, col: int) -> Number:
        return self.matrix.data[self.matrix.dt_idx(self.row, col)]

    def __setitem__(self, col: int, value: Number) -> None:
        self.matrix.data[self.matrix.dt_idx(self.row, col)] = value


class Col:
    def __init__(self, matrix: Matrix, col: int) -> None:
        self.matrix = matrix
        self.col = col

    def __getitem__(self, row: int) -> Number:
        return self.matrix.data[self.matrix.dt_idx(row, self.col)]

    def __setitem__(self, row: int, value: Number) -> None:
        self.matrix.data[self.matrix.dt_idx(row, self.col)] = value


def _match(mat1: Matrix, mat2: Matrix) -> bool:
    return mat2.rows == mat1.rows and mat2.cols == mat1.cols


def _doest_match(mat1: Matrix, mat2: Matrix) -> ValueError:
    return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
        mat1.rows, mat1.cols,
        mat2.rows, mat2.cols,
    ))


def _unexpected(other: Matrix) -> ValueError:
    return ValueError('Unexpected parameter of type %s' % type(other).__name__)


def _get_type(obj: MBase) -> Type[Matrix]:
    if isinstance(obj, ProxyMatrix):
        return type(obj.matrix)
    if isinstance(obj, MatrixBase):
        return type(obj)

    raise _unexpected(obj)


class MatrixBase:

    # You should define on derived class
    data: List[Number] = None
    rows: int = None
    cols: int = None
    fmt: Callable[[Number], Text] = None

    def __init__(self, fmt: Callable[[Number], Text] = None) -> None:
        self.fmt = fmt or str

    @property
    def array_length(self) -> int:
        return self.rows * self.cols

    @property
    def indexes(self) -> Iterable[Tuple[int, int]]:
        return ((row, col)
                for row in range(self.rows)
                for col in range(self.cols))

    def __len__(self) -> int:
        return self.array_length

    def dt_idx(self, row: int, col: int) -> int:
        return col + row * self.cols

    def imap(self, func: Callable[[Number, int, int], Number]) -> None:
        for i, j in self.indexes:
            idx = self.dt_idx(i, j)
            val = self.data[idx]
            self.data[idx] = func(val, i, j)

    def map(self, func: Callable[[Number, int, int], Number]) -> MBase:
        new_copy = 1 * self
        new_copy.imap(func)
        return new_copy

    def print(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                print(self.fmt(self.data[self.dt_idx(i, j)]), end=" ")
            print()

    def _operate_new(self,
                     other: Union[MBase, Number],
                     fn_matrix: Callable[[Number, int, int], Number],
                     fn_scalar: Callable[[Number, int, int], Number]) -> MBase:

        if isinstance(other, MatrixBase):
            if not _match(self, other):
                raise _doest_match(self, other)

            cls = _get_type(other)
            new_matrix = cls(other.rows, other.cols)
            new_matrix.imap(fn_matrix)
            return new_matrix

        if isinstance(other, numbers.Number):
            cls = _get_type(self)
            new_matrix = cls(self.rows, self.cols)
            new_matrix.imap(fn_scalar)
            return new_matrix

        raise _unexpected(other)

    def _operate_inplace(self,
                         other: Union[MBase, Number],
                         fn_matrix: Callable[[Number, int, int], Number],
                         fn_scalar: Callable[[Number, int, int], Number]) -> MBase:

        if isinstance(other, MatrixBase):
            if not _match(self, other):
                raise _doest_match(self, other)

            self.imap(fn_matrix)
            return self

        if isinstance(other, numbers.Number):
            self.imap(fn_scalar)
            return self

        raise _unexpected(other)

    def randomize(self, rand: Callable[[], Number] = lambda: random.uniform(-1, 1)) -> None:
        for i, _ in enumerate(self.data):
            self.data[i] = rand()

    def __add__(self, other: Union[MBase, Number]) -> MBase:
        return self._operate_new(
            other,
            lambda val, i, j: self.get(i, j) + other.get(i, j),
            lambda val, i, j: self.get(i, j) + other,
        )

    def __iadd__(self, other: Union[MBase, Number]) -> MBase:
        return self._operate_inplace(
            other,
            lambda val, i, j: val + other.get(i, j),
            lambda val, i, j: val + other,
        )

    def __radd__(self, other: Union[MBase, Number]) -> MBase:
        return self + other

    def __sub__(self, other: Union[MBase, Number]) -> MBase:
        return self._operate_new(
            other,
            lambda val, i, j: self.get(i, j) - other.get(i, j),
            lambda val, i, j: self.get(i, j) - other,
        )

    def __isub__(self, other: Union[MBase, Number]) -> MBase:
        return self._operate_inplace(
            other,
            lambda val, i, j: val - other.get(i, j),
            lambda val, i, j: val - other,
        )

    def __rsub__(self, other: Union[MBase, Number]) -> MBase:
        return self._operate_new(
            other,
            lambda val, i, j: other.get(i, j) - self.get(i, j),
            lambda val, i, j: other - self.get(i, j)
        )

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return 1 * self

    def __abs__(self):
        copy = 1 * self
        copy.imap(lambda val, row, col: abs(val))
        return copy

    def __invert__(self):
        copy = 1 * self
        copy.imap(lambda val, row, col: ~val)
        return copy

    def __mul__(self, other: Union[MBase, Number]) -> MBase:
        if isinstance(other, MatrixBase):
            if not self.cols == other.rows:
                raise ValueError(
                    'Matrix parameters a.cols and b.rows must match. '  # pylint: disable=bad-string-format-type
                    'Given: (c=%d, r=%d), (c=%d, r=%d)' % (
                        self.rows, self.cols,
                        other.rows, other.cols,
                    )
                )

            cls = _get_type(self)
            new_matrix = cls(self.rows, other.cols)

            for row, col in new_matrix.indexes:
                index = new_matrix.dt_idx(row, col)
                new_matrix.data[index] = sum(
                    map(lambda c: self.get(row, c) * other.get(c, col),  # pylint: disable=cell-var-from-loop
                        range(self.cols)),
                    0
                )

            return new_matrix

        if isinstance(other, numbers.Number):
            return self._operate_new(
                other,
                None,
                lambda val, i, j: self.get(i, j) * other
            )

        raise _unexpected(other)

    def __imul__(self, other: Union[MBase, Number]) -> MBase:
        if isinstance(other, numbers.Number):
            return self._operate_inplace(
                other,
                None,
                lambda val, i, j: val * other,
            )
        raise ValueError('Can not do inplace Matrix multiplication')

    def __rmul__(self, other: Union[MBase, Number]) -> MBase:
        return self * other

    def __getitem__(self, row: int) -> Row:
        return Row(self, row)

    def __setitem__(self, row: int, value: Number) -> Row:
        return Row(self, row)

    def __iter__(self) -> Iterable[Number]:
        return (
            self.data[self.dt_idx(row, col)]
            for row, col in self.indexes
        )

    def get(self, row: int, col: int) -> Number:
        return self.data[self.dt_idx(row, col)]

    def set(self, row: int, col: int, val: Number) -> None:
        self.data[self.dt_idx(row, col)] = val


class Matrix(MatrixBase):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.data = [0] * self.array_length

    @classmethod
    def from_array(cls, rows: int, cols: int, array: List[Number]) -> Matrix:
        if not isinstance(array, (list, tuple)):
            array = list(array)

        if rows * cols == len(array):
            matrix = cls(rows, cols)
            matrix.imap(lambda val, i, j: array[matrix.dt_idx(i, j)])
            return matrix

        raise ValueError("Total of array elements must be %d (%d * %d) but given %d" % (
            rows * cols, rows, cols, len(array)
        ))

    @property
    def t(self) -> ProxyTransposed:  # pylint: disable=invalid-name
        return ProxyTransposed(self)

    def transpose(self) -> Matrix:
        return self.t * 1


class ProxyMatrix(MatrixBase):

    def __init__(self, matrix: MBase) -> None:
        super().__init__()
        self.matrix = matrix

    @property
    def rows(self) -> int:
        return self.matrix.cols

    @property
    def cols(self) -> int:
        return self.matrix.cols

    @property
    def data(self) -> List[Number]:
        return self.matrix.data


class ProxyTransposed(ProxyMatrix):

    def __init__(self, matrix: MBase) -> None:
        super().__init__(matrix)
        self.fmt = matrix.fmt

    @property
    def rows(self) -> int:
        return self.matrix.cols

    @property
    def cols(self) -> int:
        return self.matrix.rows

    def dt_idx(self, row: int, col: int) -> int:
        return row + col * self.rows

    def __iadd__(self, other: Union[MBase, Number]) -> MBase:
        raise TypeError('Unsuported operation on %s' % type(self).__name__)

    def __isub__(self, other: Union[MBase, Number]) -> MBase:
        raise TypeError('Unsuported operation on %s' % type(self).__name__)

    def __imul__(self, other: Union[MBase, Number]) -> MBase:
        raise TypeError('Unsuported operation on %s' % type(self).__name__)

    @property
    def t(self) -> MBase:  # pylint: disable=invalid-name
        return self.matrix

    def transpose(self) -> MBase:
        return self.t * 1


del Number, MBase
