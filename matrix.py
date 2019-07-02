import numbers
import random


class Row:
    def __init__(self, matrix, row):
        self.matrix = matrix
        self.row = row

    def __getitem__(self, col):
        return self.matrix.data[self.matrix._idx(self.row, col)]

    def __setitem__(self, col, value):
        self.matrix.data[self.matrix._idx(self.row, col)] = value


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

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if not self.cols == other.rows:
                raise ValueError(
                    'Matrix parameters a.cols and b.rows must match. '
                    'Given: (c=%d, r=%d), (c=%d, r=%d)' % (
                        self.rows, self.cols,
                        other.rows, other.cols,
                    )
                )

            new_matrix = Matrix(self.rows, other.cols)

            for row, col, index in ((r, c, new_matrix._idx(r, c))
                                    for r in range(new_matrix.rows)
                                    for c in range(new_matrix.cols)):
                new_matrix.data[index] = sum(
                    map(lambda c: self.get(row, c) * other.get(c, col), range(self.cols)),
                    0
                )

            return new_matrix

        elif isinstance(other, numbers.Number):
            return self._operate_new(
                other,
                None,
                lambda val, i, j, idx: self.data[idx] * other
            )

        raise Matrix._unexpected(other)

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            return self._operate_inplace(
                other,
                None,
                lambda val, i, j, idx: val * other,
            )
        raise NotImplementedError('Can not do inplace Matrix multiplication')

    def __getitem__(self, row):
        return Row(self, row)

    def __setitem__(self, row, value):
        return Row(self, row)

    def get(self, row, col):
        return self.data[self._idx(row, col)]

    def set(self, row, col, val):
        self.data[self._idx(row, col)] = val
