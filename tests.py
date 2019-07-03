import unittest

from matrix import Matrix


class TestMatrix(unittest.TestCase):

    def setUp(self) -> None:
        self.arr_1 = [1, 2, 1,
                      0, 1, 0,
                      1, 3, 4]

        self.arr_2 = [2, 5,
                      6, 7,
                      1, 8]

        self.arr_3 = [9, 8, 7,
                      6, 5, 4,
                      3, 2, 1]

        self.transp_1 = [1, 0, 1,
                         2, 1, 3,
                         1, 0, 4]

        self.transp_2 = [2, 6, 1,
                         5, 7, 8]

        self.transp_3 = [9, 6, 3,
                         8, 5, 2,
                         7, 4, 1]

    def test_add(self):
        m1 = Matrix.from_array(3, 3, self.arr_1)
        m2 = Matrix.from_array(3, 2, self.arr_2)
        m3 = Matrix.from_array(3, 3, self.arr_3)

        self.assertListEqual((m1 + m3).data, list(map(sum, zip(self.arr_1, self.arr_3))))

        self.assertListEqual((m2 + 2).data, list(map(lambda i: i + 2, self.arr_2)))
        self.assertListEqual((2 + m2).data, list(map(lambda i: i + 2, self.arr_2)))
        self.assertListEqual((m3 + 2).data, list(map(lambda i: i + 2, self.arr_3)))
        self.assertListEqual((2 + m3).data, list(map(lambda i: i + 2, self.arr_3)))

        m2 += 2
        self.assertListEqual(m2.data, list(map(lambda i: i + 2, self.arr_2)))

        m1 += m3
        self.assertListEqual(m1.data, list(map(sum, zip(self.arr_1, self.arr_3))))

        def r1():
            m1 + m2

        def r2():
            m = Matrix.from_array(3, 3, self.arr_1)
            m += m2

        self.assertRaises(ValueError, r1)
        self.assertRaises(ValueError, r2)
