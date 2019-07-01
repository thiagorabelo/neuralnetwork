import numbers
import random


class Row:
    def __init__(self, matrix, row):
        self.matrix = matrix
        self.row = row

    def __getitem__(self, col):
        return self.matrix.data[self.matrix._idx(self.row, col)]


class Matrix:

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.array_length = rows * cols
        self.data = list(map(lambda _: 0, range(self.array_length)))

    @classmethod
    def _match(cls, m1, m2):
        return m2.rows == m1.rows and m2.cols == m1.cols

    @classmethod
    def _doest_match(cls, m1, m2):
        return ValueError('Matrix dimensions does not match: (%d, %d), (%d, %d)' % (
            m1.rows, m1.cols,
            m2.rows, m2.cols,
        ))

    @classmethod
    def _unexpected(cls, other):
        return ValueError('Unexpected parameter of type %s' % other.__class__.__name__)

    def _idx(self, r, c):
        return c + self.cols * r

    def _map(self, fn):
        for i, j, idx in ((r, c, self._idx(r, c)) for r in range(self.rows) for c in range(self.cols)):
            val = self.data[idx]
            self.data[idx] = fn(val, i, j, idx)

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print('%.2f' % self.data[self._idx(i, j)], end=" ")
            print()

    def randomize(self):
        for i, _ in enumerate(self.data):
            self.data[i] = random.uniform(-1, 1)

    def _operate_new(self, other, fn_matrix, fn_scalar):
        if isinstance(other, Matrix):
            if not Matrix._match(self, other):
                raise Matrix._doest_match(self, other)

            new_matrix = Matrix(other.rows, other.cols)
            new_matrix._map(fn_matrix)
            return new_matrix

        elif isinstance(other, numbers.Number):
            new_matrix = Matrix(self.rows, self.cols)
            new_matrix._map(fn_scalar)
            return new_matrix

        raise Matrix._unexpected(other)

    def _operate_inplace(self, other, fn_matrix, fn_scalar):
        if isinstance(other, Matrix):
            if not Matrix._match(self, other):
                raise Matrix._doest_match(self, other)

            self._map(fn_matrix)
            return self

        elif isinstance(other, numbers.Number):
            self._map(fn_scalar)
            return self

        raise Matrix._unexpected(other)

    def __add__(self, other):
        return self._operate_new(
            other,
            lambda val, i, j, idx: self.data[idx] + other.data[idx],
            lambda val, i, j, idx: self.data[idx] + other,
        )

    def __iadd__(self, other):
        return self._operate_inplace(
            other,
            lambda val, i, j, idx: val + other.data[idx],
            lambda val, i, j, idx: val + other,
        )

    def __sub__(self, other):
        return self._operate_new(
            other,
            lambda val, i, j, idx: self.data[idx] - other.data[idx],
            lambda val, i, j, idx: self.data[idx] - other,
        )

    def __isub__(self, other):
        return self._operate_inplace(
            other,
            lambda val, i, j, idx: val - other.data[idx],
            lambda val, i, j, idx: val - other,
        )

    def __getitem__(self, row):
        return Row(self, row)

    def get(self, row, col):
        return self.data[self._idx(row, col)]
