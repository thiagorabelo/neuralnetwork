import numbers
import random


class Row:
    def __init__(self, matrix, row):
        self.matrix = matrix
        self.row = row

    def __getitem__(self, col):
        return self.matrix.data[self.matrix.dt_idx(self.row, col)]

    def __setitem__(self, col, value):
        self.matrix.data[self.matrix.dt_idx(self.row, col)] = value


class Col:
    def __init__(self, matrix, col):
        self.matrix = matrix
        self.col = col

    def __getitem__(self, row):
        return self.matrix.data[self.matrix.dt_idx(row, self.com)]

    def __setitem__(self, row, value):
        self.matrix.data[self.matrix.dt_idx(row, self.com)] = value


def _match(m1, m2):
    return m2.rows == m1.rows and m2.cols == m1.cols


def _doest_match(mat1, mat2):
    return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
        mat1.rows, mat1.cols,
        mat2.rows, mat2.cols,
    ))


def _unexpected(other):
    return ValueError('Unexpected parameter of type %s' % type(other).__name__)


def _get_type(obj):
    if isinstance(obj, ProxyMatrix):
        return type(obj.matrix)
    elif isinstance(obj, MatrixBase):
        return type(obj)

    raise _unexpected(obj)


class MatrixBase:

    # You should define on derived class
    data = None
    rows = None
    cols = None
    fmt = None

    def __init__(self, fmt=None):
        self.fmt = fmt or str

    @property
    def array_length(self):
        return self.rows * self.cols

    @property
    def indexes(self):
        return ((row, col)
                for row in range(self.rows)
                for col in range(self.cols))

    def __len__(self):
        return self.array_length

    def dt_idx(self, row, col):
        return col + row * self.cols

    def _map(self, fn):
        for i, j in self.indexes:
            idx = self.dt_idx(i, j)
            val = self.data[idx]
            self.data[idx] = fn(val, i, j)

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print(self.fmt(self.data[self.dt_idx(i, j)]), end=" ")
            print()

    def _operate_new(self, other, fn_matrix, fn_scalar):
        if isinstance(other, MatrixBase):
            if not _match(self, other):
                raise _doest_match(self, other)

            cls = _get_type(other)
            new_matrix = cls(other.rows, other.cols)
            new_matrix._map(fn_matrix)
            return new_matrix

        elif isinstance(other, numbers.Number):
            cls = _get_type(self)
            new_matrix = cls(self.rows, self.cols)
            new_matrix._map(fn_scalar)
            return new_matrix

        raise _unexpected(other)

    def _operate_inplace(self, other, fn_matrix, fn_scalar):
        if isinstance(other, MatrixBase):
            if not _match(self, other):
                raise _doest_match(self, other)

            self._map(fn_matrix)
            return self

        elif isinstance(other, numbers.Number):
            self._map(fn_scalar)
            return self

        raise _unexpected(other)

    def randomize(self, rand=lambda: random.uniform(-1, 1)):
        for i, _ in enumerate(self.data):
            self.data[i] = rand()

    def __add__(self, other):
        return self._operate_new(
            other,
            lambda val, i, j: self.get(i, j) + other.get(i, j),
            lambda val, i, j: self.get(i, j) + other,
        )

    def __iadd__(self, other):
        return self._operate_inplace(
            other,
            lambda val, i, j: val + other.get(i, j),
            lambda val, i, j: val + other,
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._operate_new(
            other,
            lambda val, i, j: self.get(i, j) - other.get(i, j),
            lambda val, i, j: self.get(i, j) - other,
        )

    def __isub__(self, other):
        return self._operate_inplace(
            other,
            lambda val, i, j: val - other.get(i, j),
            lambda val, i, j: val - other,
        )

    def __rsub__(self, other):
        return self._operate_new(
            other,
            lambda val, i, j: other.get(i, j) - self.get(i, j),
            lambda val, i, j: other - self.get(i, j)
        )

    def __mul__(self, other):
        if isinstance(other, MatrixBase):
            if not self.cols == other.rows:
                raise ValueError(
                    'Matrix parameters a.cols and b.rows must match. '
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
                    map(lambda c: self.get(row, c) * other.get(c, col), range(self.cols)),
                    0
                )

            return new_matrix

        elif isinstance(other, numbers.Number):
            return self._operate_new(
                other,
                None,
                lambda val, i, j: self.get(i, j) * other
            )

        raise _unexpected(other)

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            return self._operate_inplace(
                other,
                None,
                lambda val, i, j: val * other,
            )
        raise ValueError('Can not do inplace Matrix multiplication')

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, row):
        return Row(self, row)

    def __setitem__(self, row, value):
        return Row(self, row)

    def get(self, row, col):
        return self.data[self.dt_idx(row, col)]

    def set(self, row, col, val):
        self.data[self.dt_idx(row, col)] = val

    def __iter__(self):
        return (
            self.data[self.dt_idx(row, col)]
            for row, col in self.indexes
        )


class Matrix(MatrixBase):
    def __init__(self, rows, cols):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.data = [0] * self.array_length

    @classmethod
    def from_array(cls, rows, cols, array):
        if not isinstance(array, (list, tuple)):
            array = list(array)

        if rows * cols == len(array):
            matrix = cls(rows, cols)
            matrix._map(lambda val, i, j: array[matrix.dt_idx(i, j)])
            return matrix

        raise ValueError("Total of array elements must be %d (%d * %d) but given %d" % (
            rows * cols, rows, cols, len(array)
        ))

    @property
    def t(self):
        return ProxyTransposed(self)

    def transpose(self):
        return self.t * 1


class ProxyMatrix(MatrixBase):

    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    @property
    def rows(self):
        return self.matrix.cols

    @property
    def cols(self):
        return self.matrix.cols

    @property
    def data(self):
        return self.matrix.data


class ProxyTransposed(ProxyMatrix):

    def __init__(self, matrix):
        super().__init__(matrix)
        self.fmt = matrix.fmt

    @property
    def rows(self):
        return self.matrix.cols

    @property
    def cols(self):
        return self.matrix.rows

    def dt_idx(self, row, col):
        return row + col * self.rows

    def __iadd__(self, other):
        raise TypeError('Unsuported operation on %s', type(self).__name__)

    def __isub__(self, other):
        raise TypeError('Unsuported operation on %s', type(self).__name__)

    def __imul__(self, other):
        raise TypeError('Unsuported operation on %s', type(self).__name__)

    @property
    def t(self):
        return self.matrix

    def transpose(self):
        return self.t * 1
