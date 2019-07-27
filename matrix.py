import abc
import itertools
import numbers
import operator
import random

import util

from typing import Union, List, TypeVar, Type, Callable, Text, Iterable, Tuple

from util import MatBaseType, MatType, MatProxyType, RowType, ColType, Number


class MatrixBase(abc.ABC):

    # Should be defined in derived class
    data: List[Number] = None
    rows: int = None
    cols: int = None
    fmt: Callable[[Number], Text] = None

    def __init__(self, fmt: Callable[[Number], Text] = None) -> None:
        self.fmt = fmt or str

    def __str__(self):
        return self._to_str()

    def __repr__(self):
        return self._to_str(f'Matrix({self.rows}, {self.cols}, ', ')')

    def print(self) -> None:
        print(str(self))

    def _to_str(self, prefix='', suffix=''):
        cols = [list(Col(self, col)) for col in range(self.cols)]
        cols_fmt = [[self.fmt(elem) for elem in col] for col in cols]
        max_cols = [max(len(c) for c in col) for col in cols_fmt]
        buff = [None] * self.rows
        sep = f',\n{" " * len(prefix)} '
        del cols

        for i in range(self.rows):
            row_buff = [cols_fmt[j][i].rjust(max_cols[j], ' ')
                        for j in range(self.cols)]
            buff[i] = ", ".join(row_buff)

        return f"{prefix}[{sep.join(buff)}]{suffix}"

    def _apply_op(self,
                  other: Union[MatBaseType, Number],
                  operation: Callable[[Number, Number], Number]) -> MatBaseType:

        if isinstance(other, MatrixBase):
            return util.matrix_op(self, other, operation, self.get_class())

        if isinstance(other, numbers.Number):
            return util.scalar_op(self, other, operation, self.get_class())

        raise util.unexpected(other)

    def _iapply_op(self,
                   other: Union[MatBaseType, Number],
                   operation: Callable[[Number, Number], Number]) -> MatBaseType:

        if isinstance(other, MatrixBase):
            return util.imatrix_op(self, other, operation)

        if isinstance(other, numbers.Number):
            # self.data[:] = map(lambda val: operation(val, other), self.data)
            self.imap(lambda val, row, col: operation(val, other))
            return self

    def dt_idx(self, row: int, col: int) -> int:
        return col + row * self.cols

    def imap(self, func: Callable[[Number, int, int], Number]) -> None:
        # 7.77 µs ± 100 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         idx = self.dt_idx(row, col)
        #         self.data[idx] = func(self.data[idx], row, col)

        # 8.3 µs ± 28.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        for i, j in self.indexes:
            idx = self.dt_idx(i, j)
            self.data[idx] = func(self.data[idx], i, j)

    def map(self, func: Callable[[Number, int, int], Number]) -> MatBaseType:
        new_copy = self.copy()
        new_copy.imap(func)
        return new_copy

    def randomize(self, rand: Callable[[], Number] = lambda: random.uniform(-1.0, 1.0)) -> None:
        for i in range(len(self.data)):
            self.data[i] = rand()

    @property
    def indexes(self) -> Iterable[Tuple[int, int]]:
        return itertools.product(range(self.rows),
                                 range(self.cols))

    @property
    def array_length(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self.rows * self.cols

    def __add__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._apply_op(other, operator.add)

    def __iadd__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._iapply_op(other, operator.iadd)

    def __radd__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, operator.add)

    def __sub__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._apply_op(other, operator.sub)

    def __isub__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._iapply_op(other, operator.isub)

    def __rsub__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, lambda val_b, val_a: operator.sub(val_a, val_b))

    def __mul__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, operator.mul)

    def __imul__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._iapply_op(other, operator.imul)

    def __rmul__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, operator.mul)

    def __floordiv__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, operator.floordiv)

    def __ifloordiv__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._iapply_op(other, operator.ifloordiv)

    def __rfloordiv__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, lambda val_b, val_a: operator.floordiv(val_a, val_b))

    def __truediv__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._apply_op(other, operator.truediv)

    def __itruediv__(self, other: Union[MatBaseType, Number]) -> MatBaseType:
        return self._iapply_op(other, operator.itruediv)

    def __rtruediv__(self, other: Number) -> MatBaseType:
        return self._apply_op(other, lambda val_b, val_a: operator.truediv(val_a, val_b))

    def __pow__(self, power: Number, modulo: Number = None):
        new_matrix = self._apply_op(power, operator.pow)

        if modulo is not None:
            new_matrix._iapply_op(modulo, operator.mod)

        return new_matrix

    def __rpow__(self, other: Number):
        return self._apply_op(other, lambda val_b, val_a: operator.pow(val_a, val_b))

    def __ipow__(self, other):
        return self._iapply_op(other, operator.pow)

    def __matmul__(self, other: MatBaseType) -> MatBaseType:
        if isinstance(other, MatrixBase):
            if not self.cols == other.rows:
                raise ValueError(
                    'Matrix parameters a.cols and b.rows must match. '  # pylint: disable=bad-string-format-type
                    'Given: (c=%d, r=%d), (c=%d, r=%d)' % (
                        self.rows, self.cols,
                        other.rows, other.cols,
                    )
                )

            new_matrix = self.get_class()(self.rows, other.cols)

            # 7.28 µs ± 116 ns per loop (mean ± std. dev. of 7 runs,  100000 loops each) m1 @ m1.t
            # 6.26 µs ± 93.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each) m1 @ m2
            # row = 0
            # while row < new_matrix.rows:
            #     col = 0
            #     while col < new_matrix.cols:
            #         i = 0
            #         sum_ = 0.0
            #         while i < self.cols:
            #             sum_ += self.get(row, i) * other.get(i, col)
            #             i += 1
            #         new_matrix.data[new_matrix.dt_idx(row, col)] = sum_
            #         col += 1
            #     row += 1

            # 7.69 µs ± 62.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each) m1 @ m1.t
            # 6.74 µs ± 121 ns per loop (mean ± std. dev. of 7 runs,  100000 loops each) m2 @ m2
            for row in range(new_matrix.rows):
                for col in range(new_matrix.cols):
                    sum_ = 0.0
                    for i in range(self.cols):
                        sum_ += self.get(row, i) * other.get(i, col)
                    new_matrix.data[new_matrix.dt_idx(row, col)] = sum_

            # 9.39 µs ± 17.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each) m1 @ m1.t
            # 8.46 µs ± 68.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each) m1 @ m2
            # for row, col in new_matrix.indexes:
            #     index = new_matrix.dt_idx(row, col)
            #     new_matrix.data[index] = sum(
            #         map(lambda c: self.get(row, c) * other.get(c, col),  # pylint: disable=cell-var-from-loop
            #             range(self.cols)),
            #         0
            #     )

            return new_matrix

        raise util.unexpected(other)

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

    def __getitem__(self, row: int) -> RowType:
        return Row(self, row)

    def __setitem__(self, row: int, value: Number) -> RowType:
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
    def copy(self) -> MatBaseType: ...

    @abc.abstractmethod
    def get_class(self) -> Type[MatBaseType]: ...


class Row:
    def __init__(self, matrix: MatType, row: int) -> None:
        self.matrix = matrix
        self.row = row

    def __getitem__(self, col: int) -> Number:
        return self.matrix.get(self.row, col)

    def __setitem__(self, col: int, value: Number) -> None:
        self.matrix.set(self.row, col, value)

    def __iter__(self):
        for col in range(self.matrix.cols):
            yield self.matrix.get(self.row, col)


class Col:
    def __init__(self, matrix: MatType, col: int) -> None:
        self.matrix = matrix
        self.col = col

    def __getitem__(self, row: int) -> Number:
        return self.matrix.data[self.matrix.dt_idx(row, self.col)]

    def __setitem__(self, row: int, value: Number) -> None:
        self.matrix.data[self.matrix.dt_idx(row, self.col)] = value

    def __iter__(self):
        for row in range(self.matrix.rows):
            yield self.matrix.get(row, self.col)


class Matrix(MatrixBase):
    def __init__(self, rows: int, cols: int, data: List[Number] = None, copy: bool = True) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols
        self._t = None

        if not data:
            self.data = [0] * self.array_length
        elif copy:
            self.data = data.copy()
        else:
            self.data = data

    @classmethod
    def from_array(cls, rows: int, cols: int, array: List[Number], copy: bool = True) -> MatType:
        if not isinstance(array, (list, tuple)):
            array = list(array)

        if rows * cols == len(array):
            matrix = cls(rows, cols, data=array, copy=copy)
            return matrix

        raise ValueError("Total of array elements must be %d (%d * %d) but given %d" % (
            rows * cols, rows, cols, len(array)
        ))

    @classmethod
    def from_array_rows(cls, rows: int, array: List[Number], copy: bool = True) -> MatType:
        cols: int = len(array) // rows
        return Matrix.from_array(rows, cols, array, copy)

    @classmethod
    def from_array_cols(cls, cols: int, array: List[Number], copy: bool = True) -> MatType:
        rows: int = len(array) // cols
        return Matrix.from_array(rows, cols, array, copy)

    @property
    def t(self) -> MatProxyType:  # pylint: disable=invalid-name
        if self._t is None:
            self._t = ProxyTransposed(self)
        return self._t

    def transpose(self) -> MatType:
        return self.t * 1  # Make an actual Matrix as copy from ProxyTransposed

    def copy(self) -> MatBaseType:
        cls = type(self)
        return cls(self.rows, self.cols, self.data)

    def get_class(self) -> Type[MatBaseType]:
        return type(self)


class ProxyMatrix(MatrixBase):

    def __init__(self, matrix: MatBaseType) -> None:
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
    def copy(self) -> MatBaseType: ...


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
    def t(self) -> MatBaseType:  # pylint: disable=invalid-name
        return self.matrix

    def transpose(self) -> MatBaseType:
        return self.t.copy()

    def copy(self) -> MatBaseType:
        cls = type(self)
        return cls(self.matrix.copy())

    def get_class(self) -> Type[MatBaseType]:
        return type(self.matrix)
