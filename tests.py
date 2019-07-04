import unittest

from matrix import Matrix


class TestMatrix(unittest.TestCase):  # pylint: disable=too-many-instance-attributes

    def setUp(self) -> None:
        self.arr_1 = [1, 2, 1,
                      0, 1, 0,
                      2, 3, 4]

        self.arr_2 = [2, 5,
                      6, 7,
                      1, 8]

        self.arr_3 = [9, 8, 7,
                      6, 5, 4,
                      3, 2, 1]

        self.transp_1 = [1, 0, 2,
                         2, 1, 3,
                         1, 0, 4]

        self.transp_2 = [2, 6, 1,
                         5, 7, 8]

        self.transp_3 = [9, 6, 3,
                         8, 5, 2,
                         7, 4, 1]

        self.mul_m1_m2 = [15, 27,
                          6,  7,
                          26, 63]

        self.m1 = Matrix.from_array(3, 3, self.arr_1)  # pylint: disable=invalid-name
        self.m2 = Matrix.from_array(3, 2, self.arr_2)  # pylint: disable=invalid-name
        self.m3 = Matrix.from_array(3, 3, self.arr_3)  # pylint: disable=invalid-name

    def test_add(self) -> None:
        self.assertListEqual(list(self.m1 + self.m3),
                             list(map(sum, zip(self.arr_1, self.arr_3))))

        self.assertListEqual(list(self.m2 + 2), list(map(lambda i: i + 2, self.arr_2)))
        self.assertListEqual(list(2 + self.m2), list(map(lambda i: 2 + i, self.arr_2)))
        self.assertListEqual(list(self.m3 + 2), list(map(lambda i: i + 2, self.arr_3)))
        self.assertListEqual(list(2 + self.m3), list(map(lambda i: 2 + i, self.arr_3)))

        self.m2 += 2
        self.assertListEqual(list(self.m2), list(map(lambda i: i + 2, self.arr_2)))

        self.m1 += self.m3
        self.assertListEqual(list(self.m1), list(map(sum, zip(self.arr_1, self.arr_3))))

        def raise1():
            return self.m1 + self.m2

        def raise2():
            # pylint: disable=invalid-name
            m = Matrix.from_array(3, 3, self.arr_1)
            m += self.m2
            return m

        self.assertRaises(ValueError, raise1)
        self.assertRaises(ValueError, raise2)

    def test_sub(self) -> None:
        def sub(args):
            # pylint: disable=invalid-name
            it = iter(args)
            result = next(it)
            for i in it:
                result -= i
            return result

        self.assertListEqual(list(self.m1 - self.m3),
                             list(map(sub, zip(self.arr_1, self.arr_3))))

        self.assertListEqual(list(self.m2 - 2), list(map(lambda i: i - 2, self.arr_2)))
        self.assertListEqual(list(2 - self.m2), list(map(lambda i: 2 - i, self.arr_2)))
        self.assertListEqual(list(self.m3 - 2), list(map(lambda i: i - 2, self.arr_3)))
        self.assertListEqual(list(2 - self.m3), list(map(lambda i: 2 - i, self.arr_3)))

        self.m2 -= 2
        self.assertListEqual(list(self.m2), list(map(lambda i: i - 2, self.arr_2)))

        self.m1 -= self.m3
        self.assertListEqual(list(self.m1), list(map(sub, zip(self.arr_1, self.arr_3))))

        def raise1():
            return self.m1 - self.m2

        def raise2():
            # pylint: disable=invalid-name
            m = Matrix.from_array(3, 3, self.arr_1)
            m -= self.m2
            return m

        self.assertRaises(ValueError, raise1)
        self.assertRaises(ValueError, raise2)

    def test_mul(self) -> None:
        self.assertListEqual(list(self.m1 * self.m2), self.mul_m1_m2)
        self.assertListEqual(list(self.m1 * 2), list(map(lambda i: i * 2, self.arr_1)))
        self.assertListEqual(list(2 * self.m1), list(map(lambda i: i * 2, self.arr_1)))

        def raise1():
            return self.m1 * Matrix.from_array(2, 3, self.transp_2)

        def raise2():
            m = Matrix.from_array(3, 3, self.m1)
            m *= self.m2
            return m

        self.assertRaises(ValueError, raise1)
        self.assertRaises(ValueError, raise2)

    def test_transpose(self) -> None:
        self.assertListEqual(list(self.m1.t), self.transp_1)
        self.assertListEqual(list(self.m1.t.t), self.arr_1)
        self.assertListEqual(list(self.m1.transpose()), self.transp_1)
        self.assertListEqual(list(self.m1 * Matrix.from_array(2, 3, self.transp_2).t), self.mul_m1_m2)

        t1 = Matrix.from_array(3, 3, self.transp_1)
        t2 = Matrix.from_array(2, 3, self.transp_2)

        self.assertListEqual(list(t1.t * self.m2), self.mul_m1_m2)
        self.assertListEqual(list(t1.t * t2.t), self.mul_m1_m2)


if __name__ == '__main__':
    # python -m tests
    # python -m unittest -v tests
    # python -m unittest discover -v -s tests
    unittest.main()
