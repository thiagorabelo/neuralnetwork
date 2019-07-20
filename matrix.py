from __future__ import annotations

import abc
import itertools
import numbers
import operator
import random

from typing import Union, List, TypeVar, Type, Callable, Text, Iterable, Tuple, Any


Number = Union[int, float]
MBase = TypeVar('MBase', bound='MatrixBase')


def _match(mat1: Matrix, mat2: Matrix) -> bool:
    return mat2.rows == mat1.rows and mat2.cols == mat1.cols


def _doest_match(mat1: Matrix, mat2: Matrix) -> ValueError:
    return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
        mat1.rows, mat1.cols,
        mat2.rows, mat2.cols,
    ))


def _unexpected(other: Any) -> ValueError:
    return ValueError('Unexpected parameter of type %s' % type(other).__name__)


def _get_type(obj: MBase) -> Type[Matrix]:
    if isinstance(obj, ProxyMatrix):
        return type(obj.matrix)
    if isinstance(obj, MatrixBase):
        return type(obj)

    raise _unexpected(obj)


def matrix_op(left: MBase, right: MBase, operation: Callable[[Number, Number], Number]) -> MBase:
    if not _match(left, right):
        raise _doest_match(left, right)

    cls = _get_type(right)
    new_matrix = cls(right.rows, right.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return new_matrix


def scalar_op(left: MBase, scalar: Number, operation: Callable[[Number, Number], Number]) -> MBase:
    cls = _get_type(left)
    new_matrix = cls(left.rows, left.cols)
    new_matrix.imap(lambda val, row, col: operation(left.get(row, col), scalar))
    return new_matrix


def imatrix_op(left: MBase, right: MBase, operation: Callable[[Number, Number], Number]) -> MBase:
    if not _match(left, right):
        raise _doest_match(left, right)

    left.imap(lambda val, row, col: operation(left.get(row, col), right.get(row, col)))
    return left


def iscalar_op(left: MBase, scalar: Number, operation: Callable[[Number, Number], Number]) -> MBase:
    left.imap(lambda val, row, col: operation(val, scalar))
    return left


class MatrixBase(abc.ABC):

    # Should be defined in derived class
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
        return itertools.product(range(self.rows),
                                 range(self.cols))

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
        new_copy = self.copy()
        new_copy.imap(func)
        return new_copy

    def print(self) -> None:
        buff = [None] * self.rows
        sep = ',\n '

        for i in range(self.rows):
            row_buff = [self.fmt(self.data[self.dt_idx(i, j)]) for j in range(self.cols)]
            buff[i] = f'[{", ".join(row_buff)}]'

        print(f"[{sep.join(buff)}]")

    def _apply_op(self,
                  other: Union[MBase, Number],
                  operation: Callable[[Number, Number], Number]) -> MBase:

        if isinstance(other, MatrixBase):
            return matrix_op(self, other, operation)

        if isinstance(other, numbers.Number):
            return scalar_op(self, other, operation)

        raise _unexpected(other)

    def _iapply_op(self,
                   other: Union[MBase, Number],
                   operation: Callable[[Number, Number], Number]) -> MBase:

        if isinstance(other, MatrixBase):
            return imatrix_op(self, other, operation)

        if isinstance(other, numbers.Number):
            return iscalar_op(self, other, operation)

    def randomize(self, rand: Callable[[], Number] = lambda: random.uniform(-1.0, 1.0)) -> None:
        for i, _ in enumerate(self.data):
            self.data[i] = rand()

    def __add__(self, other: Union[MBase, Number]) -> MBase:
        return self._apply_op(other, operator.add)

    def __iadd__(self, other: Union[MBase, Number]) -> MBase:
        return self._iapply_op(other, operator.iadd)

    def __radd__(self, other: Number) -> MBase:
        return self._apply_op(other, operator.add)

    def __sub__(self, other: Union[MBase, Number]) -> MBase:
        return self._apply_op(other, operator.sub)

    def __isub__(self, other: Union[MBase, Number]) -> MBase:
        return self._iapply_op(other, operator.isub)

    def __rsub__(self, other: Number) -> MBase:
        return self._apply_op(other, lambda val_b, val_a: operator.sub(val_a, val_b))

    def __mul__(self, other: Number) -> MBase:
        return self._apply_op(other, operator.mul)

    def __imul__(self, other: Union[MBase, Number]) -> MBase:
        if isinstance(other, numbers.Number):
            return iscalar_op(self, other, operator.imul)

        raise ValueError('Can not do inplace matrix multiplication. Only scalar inplace '
                         'multiplications are allowed.')

    def __rmul__(self, other: Number) -> MBase:
        return self._apply_op(other, operator.mul)

    def __floordiv__(self, other: Number) -> MBase:
        return self._apply_op(other, operator.floordiv)

    def __ifloordiv__(self, other: Union[MBase, Number]) -> MBase:
        return self._iapply_op(other, operator.ifloordiv)

    def __rfloordiv__(self, other: Number) -> MBase:
        return self._apply_op(other, lambda val_b, val_a: operator.floordiv(val_a, val_b))

    def __truediv__(self, other: Union[MBase, Number]) -> MBase:
        return self._apply_op(other, operator.truediv)

    def __itruediv__(self, other: Union[MBase, Number]) -> MBase:
        return self._iapply_op(other, operator.itruediv)

    def __rtruediv__(self, other: Number) -> MBase:
        return self._apply_op(other, lambda val_b, val_a: operator.truediv(val_a, val_b))

    def __pow__(self, power: Number, modulo: Number = None):
        new_matrix = self._apply_op(power, operator.powe)

        if modulo is not None:
            new_matrix._iapply_op(modulo, operator.mod)

        return new_matrix

    def __rpow__(self, other: Number):
        return self._apply_op(other, lambda val_b, val_a: operator.pow(val_a, val_b))

    def __ipow__(self, other):
        return self._iapply_op(other, operator.pow)

    def __matmul__(self, other: MBase) -> MBase:
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

        raise _unexpected(other)

    def __rmatmul__(self, other: List[Number]):
        return Matrix.from_array_cols(self.rows, other, copy=False) @ self

    def __imatmul__(self, other):
        raise ValueError('Implement this?')

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __abs__(self):
        copy = self.copy()
        copy.imap(lambda val, row, col: abs(val))
        return copy

    def __invert__(self):
        copy = self.copy()
        copy.imap(lambda val, row, col: ~val)
        return copy

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

    @abc.abstractmethod
    def copy(self) -> MBase: ...


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


class Matrix(MatrixBase):
    def __init__(self, rows: int, cols: int, data: List[Number] = None, copy: bool = True) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols

        if not data:
            self.data = [0] * self.array_length
        elif copy:
            self.data = data.copy()
        else:
            self.data = data

    @classmethod
    def from_array(cls, rows: int, cols: int, array: List[Number], copy: bool = True) -> Matrix:
        if not isinstance(array, (list, tuple)):
            array = list(array)

        if rows * cols == len(array):
            matrix = cls(rows, cols, data=array, copy=copy)
            return matrix

        raise ValueError("Total of array elements must be %d (%d * %d) but given %d" % (
            rows * cols, rows, cols, len(array)
        ))

    @classmethod
    def from_array_rows(cls, rows: int, array: List[Number], copy: bool = True) -> Matrix:
        cols: int = len(array) // rows
        return Matrix.from_array(rows, cols, array, copy)

    @classmethod
    def from_array_cols(cls, cols: int, array: List[Number], copy: bool = True) -> Matrix:
        rows: int = len(array) // cols
        return Matrix.from_array(rows, cols, array, copy)

    @property
    def t(self) -> ProxyTransposed:  # pylint: disable=invalid-name
        return ProxyTransposed(self)

    def transpose(self) -> Matrix:
        return self.t * 1  # Make an actual Matrix as copy from ProxyTransposed

    def copy(self) -> MBase:
        cls = type(self)
        return cls(self.rows, self.cols, self.data)


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

    @abc.abstractmethod
    def copy(self) -> MBase: ...


class ProxyTransposed(ProxyMatrix):

    def __init__(self, matrix: Matrix) -> None:
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

    @property
    def t(self) -> MBase:  # pylint: disable=invalid-name
        return self.matrix

    def transpose(self) -> MBase:
        return self.t.copy()

    def copy(self) -> MBase:
        cls = type(self)
        return cls(self.matrix.copy())


del Number, MBase
