#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>      // For size_t
#include <iosfwd>       // For std::ostream (forward declaration)
#include <iomanip>      // For std::setw
#include <string>       // For std::string, std::to_string
#include <tuple>        // For std::tuple
#include <vector>       // For std::vector
#include <type_traits>  // For std::is_arithmetic_v, std::is_same_v
#include <stdexcept>    // For std::out_of_range, std::invalid_argument, std::runtime_error
using namespace std;

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
    void checkCompatibility(int row, int col) const {
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
        checkSquare();
        if (determinant() == 0) {
            throw std::invalid_argument("Matrix cannot be singular (For Inverse).");
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

    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), mat(other.mat) {}

    // Construct a matrix of size row x col, initialized to 0
    Matrix(int row, int col) {
        if (row <= 0 || col <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        rows = row;
        cols = col;
        mat.resize(rows * cols, 0);
    }

    // Construct a matrix of size row x col, initialized to val
    Matrix(int row, int col, T val) {
        if (row <= 0 || col <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
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
        checkCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.getElement(i, j) = getElement(i, j) + other.getElement(i, j);
            }
        }
        return newMatrix;
    }

    // Addition assignment operator: modifies this matrix
    Matrix& operator+=(const Matrix& other) {
        checkCompatibility(other.getRows(), other.getCols());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                getElement(i, j) += other.getElement(i, j);
            }
        }
        return *this;
    }

    // Subtraction operator: returns a new matrix (this - other)
    Matrix operator-(const Matrix& other) const {
        checkCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.getElement(i, j) = getElement(i, j) - other.getElement(i, j);
            }
        }
        return newMatrix;
    }

    // Subtraction assignment operator: modifies this matrix
    Matrix& operator-=(const Matrix& other) {
        checkCompatibility(other.getRows(), other.getCols());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                getElement(i, j) -= other.getElement(i, j);
            }
        }
        return *this;
    }

    // Scalar multiplication operator: returns a new matrix (this * scale)
    Matrix operator*(const T& scale) const {
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.getElement(i, j) = getElement(i, j) * scale;
            }
        }
        return newMatrix;
    }

    // Scalar multiplication assignment operator: modifies this matrix
    Matrix& operator*=(const T& scale) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                getElement(i, j) *= scale;
            }
        }
        return *this;
    }

    // Matrix multiplication operator: returns a new matrix (this * other)
    Matrix operator*(const Matrix& other) const {
        checkMultiplicationCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, other.getCols());

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.getCols(); ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += getElement(i, k) * other.getElement(k, j);
                }
                newMatrix.getElement(i, j) = sum;
            }
        }
        return newMatrix;
    }

    // Matrix multiplication assignment operator: modifies this matrix
    Matrix& operator*=(const Matrix& other) {
        checkMultiplicationCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, other.getCols());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.getCols(); ++j) {
                T sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += getElement(i, k) * other.getElement(k, j);
                }
                newMatrix.getElement(i, j) = sum;
            }
        }
        rows = newMatrix.rows;
        cols = newMatrix.cols;
        mat = newMatrix.mat;
        return *this;
    }
    
    // Transpose the matrix in place: swaps rows and columns
    Matrix& transpose() {
        std::vector<T> newMat(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMat[j * rows + i] = getElement(i, j);
            }
        }
        std::swap(rows, cols);
        mat = std::move(newMat);
        return *this;
    }

    // Perform LU decomposition: returns (L, U, P, swapCount) where PA = LU
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, int> LU() const {
        checkSquare();

        Matrix<T> lower(rows, cols), upper(rows, cols);
        std::vector<int> perm(rows);
        std::vector<T> tempMat = mat;
        int swapCount = 0;

        // Initialize permutation vector and set L diagonal to 1
        for (int i = 0; i < rows; ++i) {
            perm[i] = i;
            lower.mat[i * cols + i] = 1;
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
                throw std::runtime_error("LU decomposition failed: Matrix is singular or near-singular (pivot near zero at position " +
                                            std::to_string(i) + ")");
            }

            // Compute U's row i
            for (int j = i; j < cols; ++j) {
                T sum = 0;
                for (int k = 0; k < i; ++k) {
                    sum += lower.mat[i * cols + k] * upper.mat[k * cols + j];
                }
                upper.mat[i * cols + j] = tempMat[perm[i] * cols + j] - sum;
            }

            // Compute L's column i below the diagonal
            for (int j = i + 1; j < rows; ++j) {
                T sum = 0;
                for (int k = 0; k < i; ++k) {
                    sum += lower.mat[j * cols + k] * upper.mat[k * cols + i];
                }
                lower.mat[j * cols + i] = (tempMat[perm[j] * cols + i] - sum) / upper.mat[i * cols + i];
            }
        }

        // Create permutation matrix P
        Matrix<T> permMat(rows, cols);
        for (int i = 0; i < rows; ++i) {
            permMat.getElement(i, perm[i]) = 1.0;
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
};

// Non-member scalar multiplication: scale * matrix
template<typename T>
Matrix<T> operator*(const T& scale, const Matrix<T>& mat) {
    Matrix<T> newMatrix(mat.getRows(), mat.getCols());
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.getCols(); ++j) {
            newMatrix.getElement(i, j) = scale * mat.getElement(i, j);
        }
    }
    return newMatrix;
}

// Output operator: prints the matrix to an output stream
template<typename T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T>& mat) {
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.getCols(); ++j) {
            stream << std::setw(10) << mat.getElement(i, j) << ' ';
        }
        stream << '\n';
    }
    return stream;
}

#endif