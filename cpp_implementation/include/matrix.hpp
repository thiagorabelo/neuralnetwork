#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>


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
};


#endif // MATRIX_HPP