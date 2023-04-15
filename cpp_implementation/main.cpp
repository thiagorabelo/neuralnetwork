#include <iostream>

#include "matrix.hpp"

int main()
{
    int v[6] = {1, 2, 3,
                4, 5, 6};
    Matrix<int> mat{2, 3, v};

    //mat.print();

    std::cout << "mat[0][0] = " << mat[0][0] << "\n"
              << "mat[1][1] = " << mat[1][1] << "\n\n";
    
    std::cout << "mat[1][2] = " << mat(1, 2) << "\n"
              << "mat[0][1] = " << mat(0, 1) << std::endl;

    return 0;
}