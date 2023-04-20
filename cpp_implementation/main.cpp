#include <iostream>
#include "matrix.hpp"


Matrix<int> create_matrix_3x2()
{
    return Matrix<int>{
        2, 3,
        new int[6]{
            9, 8, 7,
            6, 5, 4
        },
        true
    };
}


Matrix<int> create_matrix_2x2()
{
    return Matrix<int>{
        2, 2,
        new int[4]{
            1, 2,
            3, 4
        },
        true
    };
}


int main()
{
    int v[6] = {1, 2, 3,
                4, 5, 6};
    Matrix<int> mat{2, 3, v};

    std::cout << "mat[0][0] = " << mat[0][0] << "\n"
              << "mat[1][1] = " << mat[1][1] << "\n\n";

    std::cout << "mat[1][2] = " << mat(1, 2) << "\n"
              << "mat[0][1] = " << mat(0, 1) << "\n" << std::endl;

    mat.print();

    (mat + 2).print();

    (mat - 2).print();

    (mat * 2).print();

    (mat / 2).print();


    (2 + mat).print();

    (2 - mat).print();

    (2 * mat).print();

    (2 / mat).print();


    Matrix<int> outra = create_matrix_3x2();

    std::cout << std::endl;

    Matrix<int> copy = outra;
    copy.print();

    mat = copy;
    mat.print();

    outra = create_matrix_2x2();
    mat.print();

    return 0;
}
