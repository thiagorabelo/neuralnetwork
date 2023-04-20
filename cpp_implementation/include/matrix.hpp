#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <utility>


template<typename T>
class Matrix;

template<typename T>
class Row;

template<typename Tp, typename Op>
Matrix<Tp> apply_op(const Matrix<Tp>& left, Op operation);

template<typename T>
Matrix<T> operator+ (const Matrix<T>&, const T);

template<typename T>
Matrix<T> operator- (const Matrix<T>&, const T);

template<typename T>
Matrix<T> operator* (const Matrix<T>&, const T);

template<typename T>
Matrix<T> operator/ (const Matrix<T>&, const T);

template<typename T>
Matrix<T> operator+ (const T, const Matrix<T>&);

template<typename T>
Matrix<T> operator- (const T, const Matrix<T>&);

template<typename T>
Matrix<T> operator* (const T, const Matrix<T>&);

template<typename T>
Matrix<T> operator/ (const T, const Matrix<T>&);


template<typename T>
class Matrix
{
    public:
        Matrix(size_t rows, size_t cols)
        : m_rows{rows}, m_cols{cols}, m_data{std::make_unique<T[]>(rows * cols)}
        {
            // std::cout << "Matrix(size_t, size_t)" << std::endl;
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
            // std::cout << "Matrix(size_t, size_t, T[], bool)" << std::endl;
        }

        Matrix(Matrix<T>& other)
        : m_rows{other.m_rows}, m_cols{other.m_cols}, m_data{std::make_unique<T[]>(other.m_rows * other.m_cols)}
        {
            std::memcpy(m_data.get(), other.m_data.get(), sizeof(T) * m_rows * m_cols);
            // std::cout << "Matrix(Matrix<T>&)" << std::endl;
        }

        Matrix(Matrix<T>&& other)
        : m_rows{other.m_rows}, m_cols{other.m_cols}
        {
            m_data = std::move(other.m_data);
            // std::cout << "Matrix(Matrix<T>&&)" << std::endl;
        }

        virtual ~Matrix()
        {
        }

        Matrix<T>& operator=(Matrix<T>& other)
        {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data.reset(new T[m_rows * m_cols]);
            std::memcpy(m_data.get(), other.m_data.get(), sizeof(T) * m_rows * m_cols);
            // std::cout << "operator=(Matrix<T>&)" << std::endl;
            return *this;
        }

        Matrix<T>& operator=(Matrix<T>&& other)
        {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = std::move(other.m_data);
            // std::cout << "operator=(Matrix<T>&&)" << std::endl;
            return *this;
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
        friend Matrix<Tp> apply_op(const Matrix<Tp>& left, Op operation);

        friend Matrix<T> operator+ <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator- <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator* <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator/ <> (const Matrix<T>& left, const T right);

        friend Matrix<T> operator+ <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator- <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator* <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator/ <> (const T left, const Matrix<T>& right);

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
Matrix<Tp> apply_op(const Matrix<Tp>& matrix, Op operation)
{
    Matrix<Tp> result{matrix.m_rows, matrix.m_cols};
    std::transform(
        matrix.m_data.get(),
        matrix.m_data.get()+matrix.size(),
        result.m_data.get(),
        operation
    );

    return result;
}


template<typename T>
Matrix<T> operator+(const Matrix<T>& left, const T right)
{
    return apply_op(
        left,
        [&right](T& val){
            return val + right;
        }
    );
}


template<typename T>
Matrix<T> operator- (const Matrix<T>& left, const T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val - right;
        }
    );
}


template<typename T>
Matrix<T> operator* (const Matrix<T>& left, const T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val * right;
        }
    );
}


template<typename T>
Matrix<T> operator/ (const Matrix<T>& left, const T right)
{
    return apply_op(
        left,
        [&right](T& val) {
            return val / right;
        }
    );
}


template<typename T>
Matrix<T> operator+(const T left, const Matrix<T>& right)
{
    return apply_op(
        right,
        [&left](T& val){
            return left + val;
        }
    );
}


template<typename T>
Matrix<T> operator- (const T left, const Matrix<T>& right)
{
    return apply_op(
        right,
        [&left](T& val) {
            return left - val;
        }
    );
}


template<typename T>
Matrix<T> operator* (const T left, const Matrix<T>& right)
{
    return apply_op(
        right,
        [&left](T& val) {
            return left * val;
        }
    );
}


template<typename T>
Matrix<T> operator/ (const T left, const Matrix<T>& right)
{
    return apply_op(
        right,
        [&left](T& val) {
            return left / val;
        }
    );
}


#endif // MATRIX_HPP