#include <iostream>
// #include "matrix.hpp"
#include "multilayer_perceptron.hpp"


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
    // int v[6] = {1, 2, 3,
    //             4, 5, 6};
    // Matrix<int> mat{2, 3, v};

    // std::cout << "mat[0][0] = " << mat[0][0] << "\n"
    //           << "mat[1][1] = " << mat[1][1] << "\n\n";

    // std::cout << "mat[1][2] = " << mat(1, 2) << "\n"
    //           << "mat[0][1] = " << mat(0, 1) << "\n" << std::endl;

    // std::cout << "mat:\n";
    // mat.print();

    // std::cout << "\nmat::transposed:\n";
    // mat.transpose().print();

    // std::cout << std::endl;

    // (mat + 2).print();

    // (mat - 2).print();

    // (mat * 2).print();

    // (mat / 2).print();

    // (2 + mat).print();

    // (2 - mat).print();

    // (2 * mat).print();

    // (2 / mat).print();


    // Matrix<int> outra = create_matrix_3x2();

    // std::cout << std::endl;

    // (mat + outra).print();

    // (outra - mat).print();

    // Matrix<int> copy = outra;
    // copy.print();

    // mat = copy;
    // mat.print();

    // outra = create_matrix_2x2();
    // mat.print();

    // // try {
    // //     (create_matrix_2x2() + create_matrix_3x2()).print();
    // // } catch (MatrixOperationError& ex) {
    // //     std::cout << ex.what_str() << std::endl;
    // // }

    // Matrix<int> left_mult{
    //     3, 2,
    //     new int[6]{
    //         1, 2,
    //         3, 4,
    //         5, 6
    //     },
    //     true
    // };
    // Matrix<int> right_mult{
    //     2, 1,
    //     new int[2]{
    //         1,
    //         2
    //     },
    //     true
    // };

    // (left_mult * right_mult).print();

    MLP mlp{
        1,
        {10, 30, 20, 10, 1},
        TANH,
        LINEAR,
        -1.0, 1.0
    };

    return 0;
}
