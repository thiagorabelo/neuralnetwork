#include <iostream>

#include "matrix.hpp"

int main()
{
    int v[6] = {1, 2, 3,
                4, 5, 6};
    Matrix<int> mat{2, 3, v};

    mat.print();

    return 0;
}