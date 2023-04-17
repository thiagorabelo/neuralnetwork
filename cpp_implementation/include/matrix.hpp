#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>


template<typename T>
class Matrix;

template<typename T>
class Row;

template<typename Tp, typename Op>
Matrix<Tp> apply_op(Matrix<Tp>& left, Op operation);

template<typename T>
Matrix<T> operator+ (Matrix<T>&, T);

template<typename T>
Matrix<T> operator- (Matrix<T>&, T);

template<typename T>
Matrix<T> operator* (Matrix<T>&, T);

template<typename T>
Matrix<T> operator/ (Matrix<T>&, T);


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

        void print(std::stringstream& stream)
        {
            for (size_t row = 0; row < m_rows; row++) {
                for (size_t col = 0; col < m_cols; col++) {
                    stream << m_data.get()[row * m_cols + col] << ", ";
                }
                stream << "\n";
            }
            stream << "\n";
        }

        void print()
        {
            std::stringstream stream;
            print(stream);
            std::cout << stream.str() << std::flush;
        }

        size_t size() const
        {
            return m_rows * m_cols;
        }


        /* FRIENDS */
        friend class Row<T>;

        template<typename Tp, typename Op>
        friend Matrix<Tp> apply_op(Matrix<Tp>& left, Op operation);

        friend Matrix<T> operator+ <> (Matrix<T>& left, T right);
        friend Matrix<T> operator- <> (Matrix<T>& left, T right);
        friend Matrix<T> operator* <> (Matrix<T>& left, T right);
        friend Matrix<T> operator/ <> (Matrix<T>& left, T right);

    private:
        size_t m_rows;
        size_t m_cols;
        std::unique_ptr<T[]> m_data;
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


template<typename Tp, typename Op>
Matrix<Tp> apply_op(Matrix<Tp>& left, Op operation)
{
    Tp* result = new Tp[left.size()];
    std::transform(
        left.m_data.get(),
        left.m_data.get()+left.size(),
        result,
        operation
    );

    return Matrix<Tp>{left.m_rows, left.m_cols, result, true};
}


template<typename T>
Matrix<T> operator+(Matrix<T>& left, T right)
{
    return apply_op(
        left,
        [&right](T& val){
            return val + right;
        }
    );
}


template<typename T>
Matrix<T> operator- (Matrix<T>& left, T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val - right;
        }
    );
}

template<typename T>
Matrix<T> operator* (Matrix<T>& left, T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val * right;
        }
    );
}

template<typename T>
Matrix<T> operator/ (Matrix<T>& left, T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val / right;
        }
    );
}


#endif // MATRIX_HPP