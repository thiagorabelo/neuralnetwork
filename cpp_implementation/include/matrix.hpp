#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>


template<typename T>
class Matrix;

template<typename T>
class Row;

template<typename Tp, typename Op>
Matrix<Tp> apply_op(const Matrix<Tp>& left, Op operation);

template<typename Tp, typename Op>
Matrix<Tp> apply_op(const Matrix<Tp>& left, const Matrix<Tp>& right, Op operation);

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
Matrix<T> operator+ (const Matrix<T>& left, const Matrix<T>& right);

template<typename T>
Matrix<T> operator- (const Matrix<T>& left, const Matrix<T>& right);

template<typename T>
Matrix<T> operator* (const Matrix<T>& left, const Matrix<T>& right);


class MatrixOperationError : public std::exception
{
    public:
        MatrixOperationError(std::string cause)
        : std::exception(), m_what{cause}
        {
        }

        MatrixOperationError(const MatrixOperationError& ex)
        : std::exception(ex), m_what{ex.m_what}
        {
        }

        MatrixOperationError(MatrixOperationError&& ex)
        : std::exception(std::move(ex))
        {
            m_what = std::move(ex.m_what);
        }

        MatrixOperationError& operator=(const MatrixOperationError& ex)
        {
            m_what = ex.m_what;
            std::exception::operator=(ex);
            return *this;
        }

        MatrixOperationError& operator=(MatrixOperationError&& ex)
        {
            m_what = std::move(ex.m_what);
            std::exception::operator=(std::move(ex));
            return *this;
        }

        virtual ~MatrixOperationError()
        {
        }

        virtual const char* what() const noexcept
        {
            return m_what.c_str();
        }

        virtual std::string what_str() const noexcept
        {
            return m_what;
        }
    private:
        std::string m_what;
};


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

        inline T get(size_t row, size_t col) const
        {
            return m_data.get()[row * m_cols + col];
        }

        inline T& set(size_t row, size_t col)
        {
            return m_data.get()[row * m_cols + col];
        }

        inline T operator()(size_t row, size_t col) const
        {
            return get(row, col);
        }

        inline T& operator()(size_t row, size_t col)
        {
            return set(row, col);
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

        size_t rows() const
        {
            return m_rows;
        }

        size_t cols() const
        {
            return m_cols;
        }

        size_t size() const
        {
            return m_rows * m_cols;
        }

        /* FRIENDS */
        friend class Row<T>;

        template<typename Tp, typename Op>
        friend Matrix<Tp> apply_op(const Matrix<Tp>& left, Op operation);

        template<typename Tp, typename Op>
        friend Matrix<Tp> apply_op(const Matrix<Tp>& left, const Matrix<Tp>& right, Op operation);

        friend Matrix<T> operator+ <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator- <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator* <> (const Matrix<T>& left, const T right);
        friend Matrix<T> operator/ <> (const Matrix<T>& left, const T right);

        friend Matrix<T> operator+ <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator- <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator* <> (const T left, const Matrix<T>& right);
        friend Matrix<T> operator/ <> (const T left, const Matrix<T>& right);

        friend Matrix<T> operator+ <> (const Matrix<T>& left, const Matrix<T>& right);
        friend Matrix<T> operator- <> (const Matrix<T>& left, const Matrix<T>& right);
        friend Matrix<T> operator* <> (const Matrix<T>& left, const Matrix<T>& right);

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


template<typename Tp, typename Op>
Matrix<Tp> apply_op(const Matrix<Tp>& left, const Matrix<Tp>& right, Op operation)
{
    if (!(left.m_rows == right.m_rows && left.m_cols == right.m_cols)) {
        std::stringstream ss;
        ss << "Matrices "
           << "(" << left.m_rows << ", " << left.m_cols << ") and "
           << "(" << right.m_rows << ", " << right.m_cols << ") "
           << "does not match dimensions";
        throw MatrixOperationError(ss.str());
    }

    Matrix<Tp> result{right.m_rows, right.m_cols};
    for (size_t row = 0; row < left.m_rows; row++) {
        for (size_t col = 0; col < left.m_cols; col++) {
            result.set(row, col) = operation(left.get(row, col), right.get(row, col));
        }
    }

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


template<typename T>
Matrix<T> operator+ (const Matrix<T>& left, const Matrix<T>& right)
{
    return apply_op(
        left,
        right,
        [](T a, T b) {
            return a + b;
        }
    );
}


template<typename T>
Matrix<T> operator- (const Matrix<T>& left, const Matrix<T>& right)
{
    return apply_op(
        left,
        right,
        [](T a, T b) {
            return a - b;
        }
    );
}


template<typename T>
Matrix<T> operator* (const Matrix<T>& left, const Matrix<T>& right)
{
    if (left.m_cols != right.m_rows) {
        std::stringstream ss;
        ss << "Matrices "
           << "M1(rows=" << left.m_rows << ", cols=" << left.m_cols << ") and "
           << "M2(rows=" << right.m_rows << ", cols=" << right.m_cols << ") must "
           << "match M1.cols and M2.rows";
        throw MatrixOperationError(ss.str());
    }

    Matrix<T> result{left.m_rows, right.m_cols};

    for (size_t row = 0; row < result.m_rows; row++) {
        for (size_t col = 0; col < result.m_cols; col++) {
            T sum = T(0);
            for (size_t i = 0; i < left.m_cols; i++) {
                sum += left.get(row, i) * right.get(i, col);
            }
            result.set(row, col) = sum;
        }
    }

    return result;
}

#endif // MATRIX_HPP