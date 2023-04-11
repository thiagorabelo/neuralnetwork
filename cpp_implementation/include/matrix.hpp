#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>


template<typename number_t>
class Matrix
{
    public:
        using DataUniquePtr = std::unique_ptr<number_t>;

    public:
        Matrix(size_t rows, size_t cols, number_t* data = nullptr, bool copy = true)
        : m_rows(rows), m_cols(cols)
        {
            if (!data) {
                m_data = std::make_unique<number_t>(rows * cols);
            } else if (copy) {
                number_t* tmp_ptr = new number_t[rows * cols];
                std::memcpy(tmp_ptr, data, sizeof(number_t) * rows * cols);
                m_data = std::unique_ptr<number_t>(tmp_ptr);
            } else {
                m_data = std::unique_ptr<number_t>(data);
            }
        }

        virtual ~Matrix()
        {
        }

        void print() const
        {
            for (size_t i = 0; i < size(); i++) {
                std:: cout << (*m_data)[i] << std::endl;
            }
        }

        size_t size() const
        {
            return m_rows * m_cols;
        }

    private:
        DataUniquePtr m_data;
        size_t m_rows;
        size_t m_cols;
};


#endif // MATRIX_HPP