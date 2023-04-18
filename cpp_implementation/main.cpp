#include <iostream>

#include "matrix.hpp"

Matrix<int> create_matrix_2x2()
{
    return Matrix<int>{
        2, 2,
        new int[4]{
            0, 1,
            2, 3
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

    Matrix<int> adicacao = mat + 2;
    adicacao.print();

    Matrix<int> subtracao = mat - 2;
    subtracao.print();

    Matrix<int> multiplicacao = mat * 2;
    multiplicacao.print();

    Matrix<int> divisao = mat / 2;
    divisao.print();

    Matrix<int> copy = divisao;
    copy.print();

    mat = copy;
    mat.print();

    mat = create_matrix_2x2();
    mat.print();

    return 0;
}
