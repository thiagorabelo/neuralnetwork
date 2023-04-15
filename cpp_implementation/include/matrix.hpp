#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>
// #include <sstream>


template<typename T>
class Row;


template<typename T>
class Matrix
{
    public:
        Matrix(size_t rows, size_t cols)
        : m_rows{rows}, m_cols{cols}, m_data{std::make_unique<T[]>(rows * cols)}
        {
        }

        Matrix(size_t rows, size_t cols, T data[], bool take_ownership = false)
        :
            m_rows{rows},
            m_cols{cols},
            m_data{take_ownership ? std::unique_ptr<T[]>{data} : std::make_unique<T[]>(rows * cols)}
        {
            if (!take_ownership) {
                std::memcpy(m_data.get(), data, sizeof(T) * m_rows * m_cols);
            }
        }

        virtual ~Matrix()
        {
        }

        inline T get(size_t row, size_t col)
        {
            return m_data.get()[row * m_cols + col];
        }

        inline T operator()(size_t row, size_t col)
        {
            return get(row, col);
        }

        inline Row<T> operator[](size_t col)
        {
            return Row<T>(*this, col);
        }

        void print() const
        {
            for (size_t row = 0; row < m_rows; row++) {
                for (size_t col = 0; col < m_cols; col++) {
                    std::cout << m_data.get()[row * m_cols + col] << ", ";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }

        size_t size() const
        {
            return m_rows * m_cols;
        }

    private:
        size_t m_rows;
        size_t m_cols;
        std::unique_ptr<T[]> m_data;

        friend class Row<T>;
};


template<typename T>
class Row
{
    public:
        Row(Matrix<T>& mat, size_t row)
        : m_row{row}, m_mat{mat}
        {
        }

        virtual ~Row()
        {
            // std::stringstream stream;
            // stream << "~Row(mat, " << m_row << ")";
            // std::cout << stream.str() << std::endl;
        }

        inline T operator[](size_t col)
        {
            return m_mat.get(m_row, col);
        }

    private:
        size_t m_row;
        Matrix<T>& m_mat;
};


#endif // MATRIX_HPP