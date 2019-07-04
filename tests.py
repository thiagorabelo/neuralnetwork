import unittest

from matrix import Matrix


class TestMatrix(unittest.TestCase):  #pylint: disable=too-many-instance-attributes

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

        self.m1 = Matrix.from_array(3, 3, self.arr_1)  # pylint: disable=invalid-name
        self.m2 = Matrix.from_array(3, 2, self.arr_2)  # pylint: disable=invalid-name
        self.m3 = Matrix.from_array(3, 3, self.arr_3)  # pylint: disable=invalid-name

    def test_add(self) -> None:
        self.assertListEqual((self.m1 + self.m3).data,
                             list(map(sum, zip(self.arr_1, self.arr_3))))

        self.assertListEqual((self.m2 + 2).data, list(map(lambda i: i + 2, self.arr_2)))
        self.assertListEqual((2 + self.m2).data, list(map(lambda i: 2 + i, self.arr_2)))
        self.assertListEqual((self.m3 + 2).data, list(map(lambda i: i + 2, self.arr_3)))
        self.assertListEqual((2 + self.m3).data, list(map(lambda i: 2 + i, self.arr_3)))

        self.m2 += 2
        self.assertListEqual(self.m2.data, list(map(lambda i: i + 2, self.arr_2)))

        self.m1 += self.m3
        self.assertListEqual(self.m1.data, list(map(sum, zip(self.arr_1, self.arr_3))))

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

        self.assertListEqual((self.m1 - self.m3).data,
                             list(map(sub, zip(self.arr_1, self.arr_3))))

        self.assertListEqual((self.m2 - 2).data, list(map(lambda i: i - 2, self.arr_2)))
        self.assertListEqual((2 - self.m2).data, list(map(lambda i: 2 - i, self.arr_2)))
        self.assertListEqual((self.m3 - 2).data, list(map(lambda i: i - 2, self.arr_3)))
        self.assertListEqual((2 - self.m3).data, list(map(lambda i: 2 - i, self.arr_3)))

        self.m2 -= 2
        self.assertListEqual(self.m2.data, list(map(lambda i: i - 2, self.arr_2)))

        self.m1 -= self.m3
        self.assertListEqual(self.m1.data, list(map(sub, zip(self.arr_1, self.arr_3))))

        def raise1():
            return self.m1 - self.m2

        def raise2():
            # pylint: disable=invalid-name
            m = Matrix.from_array(3, 3, self.arr_1)
            m -= self.m2
            return m

        self.assertRaises(ValueError, raise1)
        self.assertRaises(ValueError, raise2)

if __name__ == '__main__':
    # python -m tests
    # python -m unittest -v tests
    # python -m unittest discover -v -s tests
    unittest.main()
