#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <cstddef>      // For size_t
#include <iosfwd>       // For std::ostream (forward declaration)
#include <iomanip>      // For std::setw
#include <string>       // For std::string, std::to_string
#include <tuple>        // For std::tuple
#include <vector>       // For std::vector
#include <type_traits>  // For std::is_arithmetic_v, std::is_same_v
#include <stdexcept>    // For std::out_of_range, std::invalid_argument, std::runtime_error

template<typename T>
class Matrix {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                  "Matrix can only be instantiated with numeric (arithmetic, non-bool) types.");

private:
    int rows;           // Number of rows in the matrix
    int cols;           // Number of columns in the matrix
    std::vector<T> mat; // 1D vector storing matrix elements in row-major order
    
private:
    // Check if indices i, j are within bounds; throw if not
    void checkBounds(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::out_of_range("Matrix Index Out of Bounds: i=" + std::to_string(i) +
                                    ", j=" + std::to_string(j));
        }
    }

    // Check if dimensions match for addition/subtraction; throw if not
    void checkAdditionCompatibility(int row, int col) const {
        if (row != rows || col != cols) {
            throw std::invalid_argument("Both Matrix's Dimensions must be same (For Operations +, -): " +
                                        std::to_string(rows) + "x" + std::to_string(cols) + " vs " +
                                        std::to_string(row) + "x" + std::to_string(col));
        }
    }

    // Check if dimensions are compatible for multiplication; throw if not
    void checkMultiplicationCompatibility(int row, int col) const {
        if (cols != row) {
            throw std::invalid_argument("Matrix Dimensions are not Compatible For Multiplication: " +
                                        std::to_string(rows) + "x" + std::to_string(cols) + " vs " +
                                        std::to_string(row) + "x" + std::to_string(col));
        }
    }

    // Check if the matrix is square (rows == cols); throw if not
    void checkSquare() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix should be Square (For Determinant/Inverse): " +
                                        std::to_string(rows) + "x" + std::to_string(cols));
        }
    }

    // Check if the matrix is invertible (square and non-singular); throw if not
    void checkInverse() const {
        // checkSquare();       // Square is being checked in determinant()
        if (determinant() == 0) {
            throw std::invalid_argument("Matrix cannot be singular (For Inverse).");
        }
    }
    
    void checkDimensions(int row, int col) {
        if (row <= 0 || col <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
    }

public:
    // Get the number of rows
    int getRows() const {
        return rows;
    }

    // Get the number of columns
    int getCols() const {
        return cols;
    }

    // Get ith row in Vector Form
    std::vector<T> getRow(int i) const {
        checkBounds(i, 0);
        std::vector<T> vec;
        
        for(int j = 0; j < cols; ++j) {
            vec.push_back(mat[i * cols + j]);
        }
        
        return vec;
    }
    
    // Get jth column in Vector Form
    std::vector<T> getCol(int j) const {
        checkBounds(0, j);
        std::vector<T> vec;
        
        for(int i = 0; i < rows; ++i) {
            vec.push_back(mat[i * cols + j]);
        }

        return vec;
    }

    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), mat(other.mat) {}

    // Construct a matrix of size row x col, initialized to 0
    Matrix(int row, int col) {
        checkDimensions(row, col);
        rows = row;
        cols = col;
        mat.resize(rows * cols, 0);
    }

    // Construct a matrix of size row x col, initialized to val
    Matrix(int row, int col, T val) {
        checkDimensions(row, col);
        rows = row;
        cols = col;
        mat.resize(rows * cols, val);
    }

    // Construct a matrix from a 2D vector
    Matrix(std::vector<std::vector<T>> val) {
        if (val.empty()) {
            throw std::invalid_argument("Input Vector should not be empty.");
        }

        size_t expectedCols = val[0].size();
        for (size_t i = 1; i < val.size(); ++i) {
            if (val[i].size() != expectedCols) {
                throw std::invalid_argument("Input Vector should have consistent number of cols.");
            }
        }

        rows = static_cast<int>(val.size());
        cols = static_cast<int>(val[0].size());
        mat.resize(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat[i * cols + j] = val[i][j];
            }
        }
    }

    // Get a reference to element at (i, j); throws if out of bounds
    T& getElement(int i, int j) {
        checkBounds(i, j);
        return mat[i * cols + j];
    }

    // Get a const reference to element at (i, j); throws if out of bounds
    const T& getElement(int i, int j) const {
        checkBounds(i, j);
        return mat[i * cols + j];
    }

    // Assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            mat = other.mat;
        }
        return *this;
    }

    // Addition operator: returns a new matrix (this + other)
    Matrix operator+(const Matrix& other) const {
        checkAdditionCompatibility(other.rows, other.cols);
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.mat[i * cols + j] = mat[i * cols + j] + other.mat[i * cols + j];
            }
        }
        return newMatrix;
    }

    // Addition assignment operator: modifies this matrix
    Matrix& operator+=(const Matrix& other) {
        checkAdditionCompatibility(other.rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat[i * cols + j] += other.mat[i * cols + j];
            }
        }
        return *this;
    }

    // Subtraction operator: returns a new matrix (this - other)
    Matrix operator-(const Matrix& other) const {
        checkAdditionCompatibility(other.rows, other.cols);
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.mat[i * cols + j] = mat[i * cols + j] - other.mat[i * cols + j];
            }
        }
        return newMatrix;
    }

    // Subtraction assignment operator: modifies this matrix
    Matrix& operator-=(const Matrix& other) {
        checkAdditionCompatibility(other.rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat[i * cols + j] -= other.mat[i * cols + j];
            }
        }
        return *this;
    }

    // Scalar multiplication operator: returns a new matrix (this * scale)
    Matrix operator*(const T& scale) const {
        std::vector<T> newMat(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMat[i * cols + j] = mat[i * cols + j] * scale;
            }
        }
        Matrix<T> newMatrix(rows, cols);
        newMatrix.mat = newMat;
        return newMatrix;
    }

    // Scalar multiplication assignment operator: modifies this matrix
    Matrix& operator*=(const T& scale) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat[i * cols + j] *= scale;
            }
        }
        return *this;
    }

    // Matrix multiplication operator: returns a new matrix (this * other)
    Matrix operator*(const Matrix& other) const {
        checkMultiplicationCompatibility(other.rows, other.cols);
        std::vector<T> newMat(rows * other.cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += mat[i * cols + k] * other.mat[k * other.cols + j];
                }
                newMat[i * other.cols + j] = sum;
            }
        }

        Matrix<T> newMatrix(rows, other.cols);
        newMatrix.mat = newMat;
        return newMatrix;
    }

    // Matrix multiplication assignment operator: modifies this matrix
    Matrix& operator*=(const Matrix& other) {
        checkMultiplicationCompatibility(other.rows, other.cols);
        std::vector<T> newMat(rows * other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += mat[i * cols + k] * other.mat[k * other.cols + j];
                }
                newMat[i * other.cols + j] = sum;
            }
        }
        cols = other.cols;
        mat = newMat;
        return *this;
    }

    // Transpose the matrix in place: swaps rows and columns
    Matrix& transpose() {
        std::vector<T> newMat(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMat[j * rows + i] = mat[i * cols + j];
            }
        }
        std::swap(rows, cols);
        mat = std::move(newMat);
        return *this;
    }

    // Perform LU decomposition: returns (L, U, P, swapCount) where PA = LU
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, int> LU() const {
        // checkSquare();

        Matrix<T> lower(rows, rows), upper(rows, cols);
        std::vector<int> perm(rows);
        std::vector<T> tempMat = mat;
        int swapCount = 0;

        // Initialize permutation vector and set L diagonal to 1
        for (int i = 0; i < rows; ++i) {
            perm[i] = i;
            lower.mat[i * rows + i] = 1;
        }

        for (int i = 0; i < rows; ++i) {
            // Partial pivoting: find row with largest element in column i
            int pivotRow = i;
            T pivotVal = std::abs(tempMat[perm[i] * cols + i]);
            for (int k = i + 1; k < rows; ++k) {
                T val = std::abs(tempMat[perm[k] * cols + i]);
                if (val > pivotVal) {
                    pivotVal = val;
                    pivotRow = k;
                }
            }

            // Swap rows if necessary
            if (pivotRow != i) {
                std::swap(perm[i], perm[pivotRow]);
                ++swapCount;
            }

            // Check for singular or near-singular matrix
            T pivot = tempMat[perm[i] * cols + i];
            if (std::abs(pivot) < 1e-10) {
                throw std::runtime_error("LU decomposition failed: Matrix is singular or near-singular (pivot near zero at position "
                                        + std::to_string(i) + ")");
            }

            // Compute U's row i
            for (int j = i; j < cols; ++j) {
                T sum = 0;
                for (int k = 0; k < i; ++k) {
                    sum += lower.mat[i * rows + k] * upper.mat[k * cols + j];
                }
                upper.mat[i * cols + j] = tempMat[perm[i] * cols + j] - sum;
            }

            // Compute L's column i below the diagonal
            for (int j = i + 1; j < rows; ++j) {
                T sum = 0;
                for (int k = 0; k < i; ++k) {
                    sum += lower.mat[j * rows + k] * upper.mat[k * cols + i];
                }
                lower.mat[j * rows + i] = (tempMat[perm[j] * cols + i] - sum) / upper.mat[i * cols + i];
            }
        }

        // Create permutation matrix P
        Matrix<T> permMat(rows, rows);
        for (int i = 0; i < rows; ++i) {
            permMat.mat[i * rows + perm[i]] = 1.0;
        }

        return std::make_tuple(lower, upper, permMat, swapCount);
    }

    // Compute the determinant using LU decomposition
    T determinant() const {
        checkSquare();

        auto [L, U, perm, swapCount] = LU();

        T determ = 1;
        // Adjust sign based on number of swaps
        if (swapCount % 2 == 1) {
            determ = -1;
        }
        // Multiply diagonal elements of U
        for (int i = 0; i < rows; ++i) {
            determ *= U.mat[i * rows + i];
        }

        return determ;
    }
    
    // Compute the inverse using LU decomposition with forward/back substitution
    Matrix inverse() const {
        checkInverse();

        auto [L, U, P, swapCount] = LU();

        // Compute P^T (transpose of permutation matrix)
        Matrix<T> PT = P;
        PT.transpose();

        // Solve L Y = P^T for Y (forward substitution)
        Matrix<T> Y(rows, rows);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < rows; ++i) {
                T sum = 0;
                for (int k = 0; k < i; ++k) {
                    sum += L.mat[i * cols + k] * Y.mat[k * cols + j];
                }
                Y.mat[i * cols + j] = PT.mat[i * cols + j] - sum;
            }
        }

        // Solve U X = Y for X (back substitution), where X is the inverse
        Matrix<T> X(rows, rows);
        for (int j = 0; j < rows; ++j) {
            for (int i = rows - 1; i >= 0; --i) {
                T sum = 0;
                for (int k = i + 1; k < rows; ++k) {
                    sum += U.mat[i * cols + k] * X.mat[k * cols + j];
                }
                X.mat[i * cols + j] = (Y.mat[i * cols + j] - sum) / U.mat[i * cols + i];
            }
        }
        
        return X;
    }
    
    // Computes Inner Product of i and j column of this Matrix
    T innerProduct(int i, int j) const {
        checkBounds(0, i);
        checkBounds(0, j);

        T result = 0;

        for(int k = 0; k < rows; ++k) {
            result += mat[k * cols + i] * mat[k * cols + j];
        }
        return result;
    }

    // Computes Inner Product of i Column of this and j Column of other Matrix
    T innerProduct(const Matrix& other, int i, int j) const {
        if(rows != other.rows) {
            throw std::invalid_argument("Number of Elements in both vectors are different (For Inner Product): " 
                                        + std::to_string(rows) + " vs. " + std::to_string(other.rows));
        }
        checkBounds(0, i);
        other.checkBounds(0, j);

        T result = 0;

        for(int k = 0; k < rows; ++k) {
            result += mat[k * cols + i] * other.mat[k * other.cols + j];
        }
        return result;
    }
    
    // Computes Inner Product of two 1D vector
    T innerProduct(const std::vector<T>& vec1, const std::vector<T>& vec2) {
        if(vec1.size() != vec2.size()) {
            throw std::invalid_argument("Number of Elements in both vectors are different (For Inner Product): " 
                                        + std::to_string(vec1.size()) + " vs. " + std::to_string(vec2.size()));
        }
        
        T result = 0;
        
        for(size_t i = 0; i < vec1.size(); ++i) {
            result += vec1[i] * vec2[i];
        }
        
        return result;
    }
    
    // Computes QR Decomposition of Matrix
    std::tuple<Matrix<T>, Matrix<T>> QR() {
        checkSquare();
        
        std::vector<std::vector<T>> Rvec(cols, std::vector<T>(cols, 0));
        std::vector<std::vector<T>> v;
        std::vector<std::vector<T>> QT(rows, std::vector<T>(cols));
        
        for(int i = 0; i < cols; ++i) {
            v.push_back(getCol(i));
        }
        
        for(int i = 0; i < cols; ++i) {
            Rvec[i][i] = sqrt(innerProduct(v[i], v[i]));
            
            for(int j = 0; j < rows; ++j) {
                QT[i][j] = v[i][j] / Rvec[i][i];
            }
            
            for(int k = i + 1; k < cols; ++k) {
                Rvec[i][k] = innerProduct(QT[i], v[k]);
                
                for(int j = 0; j < rows; ++j) {
                    v[k][j] -= Rvec[i][k] * QT[i][j];
                }
            }
        }

        Matrix<T> R(Rvec);
        Matrix<T> Q(QT);
        Q.transpose();

        return std::make_tuple(Q, R);
    }
    
    // Computes Frobenius Norm of Matrix
    T frobeniusNorm() const {
        T result = 0;
        for(int i = 0; i < rows * cols; ++i) {
            result += mat[i] * mat[i];
        }
        result = std::sqrt(result);
        return result;
    }
    
    // Computes largest Absolute Eigenvalue of the matrix
    T eigenvalue() const {
        checkSquare();
        
        Matrix<T> X = Matrix(rows, 1);
        for(int i = 0; i < rows; ++i) X.mat[i] = (double(rand() % 1000) / 1000);
        // X.getElement(0, 0) = 1; // Initialize X to [1, 0, ..., 0]
        float tolerance = 1e-10;

        int iter = 1000;
        while(iter--) {
            Matrix<T> X2 = *(this) * X;
            X2 *= 1 / X2.frobeniusNorm();       // Normalize to prevent Overflow
            if((X2 - X).frobeniusNorm() < tolerance) 
                break;
            X = X2;
        }
        
        if (iter <= 0)
            throw std::runtime_error("Eigenvalue computation did not converge within 1000 iterations.");

        Matrix<T> XT = X;
        XT.transpose();
        T eigenvalue = (XT * *(this) * X).mat[0] / (XT * X).mat[0];
        return eigenvalue;
    }

};

// Non-member scalar multiplication: scale * matrix
template<typename T>
Matrix<T> operator*(const T& scale, const Matrix<T>& mat) {
    Matrix<T> newMatrix(mat.getRows(), mat.cols);
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            newMatrix.getElement(i, j) = scale * mat.getElement(i, j);
        }
    }
    return newMatrix;
}

// Output operator: prints the matrix to an output stream
template<typename T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T>& mat) {
    
    std::vector<size_t> colWidths(mat.getCols(), 0);
    
    for(int j = 0; j < mat.getCols(); ++j) {
        for(int i = 0; i < mat.getRows(); ++i) {
            
            std::ostringstream oss;
            if constexpr (std::is_floating_point_v<T>) {
                oss << std::fixed << std::setprecision(6) << mat.getElement(i, j);
            } else {
                oss << mat.getElement(i, j);
            }
            std::string str = oss.str();
            
            if constexpr (std::is_floating_point_v<T>) {
                str.erase(str.find_last_not_of('0') + 1, std::string::npos);
                if(str.back() == '.') {
                    str.pop_back();
                }
            }
            colWidths[j] = std::max(colWidths[j], str.length());
        }
    }
    
    for(int i = 0; i < mat.getRows(); ++i) {
        for(int j = 0; j < mat.getCols(); ++j) {
            
            std::ostringstream oss;
            if constexpr (std::is_floating_point_v<T>) {
                oss << std::fixed << std::setprecision(6) << mat.getElement(i, j);
            } else {
                oss << mat.getElement(i, j);
            }
            std::string str = oss.str();
            
            if constexpr (std::is_floating_point_v<T>) {
                str.erase(str.find_last_not_of('0') + 1, std::string::npos);
                if(str.back() == '.') {
                    str.pop_back();
                }
            }
            stream << std::left << std::setw(colWidths[j]) << str << " ";
        }
        stream << '\n';
    }
    
    return stream;
}

#endif