#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <iomanip>
using namespace std;

template<typename T> class Matrix {
    static_assert(is_arithmetic_v<T> && !is_same_v<T, bool>, "Matrix can only be instantiated with numeric (arithmetic, non bool) Types.");
private:
    int rows;
    int cols;
    vector<T> mat;
    
private:
    void checkBounds(int i, int j) const {
        if(i < 0 || i >= rows || j < 0 || j >= cols) {
            throw out_of_range("Matrix Index Out of Bounds: i=" + to_string(i) + ", j=" + to_string(j));
        }
    }
    
    void checkCompatibility(int row, int col) const {
        if(row != rows || col != cols) {
            throw invalid_argument("Both Matrix's Dimensions must be same (For Operations +, -): " +
                                    std::to_string(rows) + "x" + std::to_string(cols) + " vs " +
                                    std::to_string(row) + "x" + std::to_string(col));
        }
    }
    
    void checkMultiplicationCompatibility(int row, int col) const {
        if(cols != row) throw invalid_argument("Matrix Dimensions are not Compatible For Multiplication: " +
                                                std::to_string(rows) + "x" + std::to_string(cols) + " vs " +
                                                std::to_string(row) + "x" + std::to_string(col));
    }
    
public:
    int getRows() const {
        return rows;
    }
    
    int getCols() const {
        return cols;
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), mat(other.mat) {}
    
    Matrix(int row, int col) {
        if(row <= 0 || col <= 0) {
            throw invalid_argument("Matrix dimensions must be positive");
        }
        rows = row;
        cols = col;
        mat.resize(rows*cols, 0);
    }

    Matrix(int row, int col, T val) {
        if(row <= 0 || col <= 0) {
            throw invalid_argument("Matrix dimensions must be positive");
        }
        rows = row;
        cols = col;
        mat.resize(rows*cols, val);
    }
    
    Matrix(vector<vector<T>> val) {
        if(val.empty()) {
            throw invalid_argument("Input Vector should not be empty.");
        }
        
        size_t expectedCols = val[0].size();
        for(size_t i = 1; i < val.size(); ++i) {
            if(val[i].size() != expectedCols) {
                throw invalid_argument("Input Vector should have consistent number of cols.");
            }
        }
        
        rows = val.size();
        cols = val[0].size();
        mat.resize(rows * cols);
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                mat[i * cols + j] = val[i][j];
            }
        }
    }
    
    T& getElement(int i, int j) {
        checkBounds(i, j);
        return mat[i * cols + j];
    }

    const T& getElement(int i, int j) const {
        checkBounds(i, j);
        return mat[i * cols + j];
    }

    // Assignment = Operator
    Matrix& operator=(const Matrix& other) {
        if(this != &other) {
            rows = other.rows;
            cols = other.cols;
            mat = other.mat;
        }
        return *this;
    }
    
    // Addition + Operator
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
    
    // Addition += Operator
    Matrix& operator+=(const Matrix& other) {
        checkCompatibility(other.getRows(), other.getCols());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                getElement(i, j) += other.getElement(i, j);
            }
        }
        return (*this);
    }
    
    // Subtraction - Operator
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
    
    // Subtraction -= Operator
    Matrix& operator-=(const Matrix& other) {
        checkCompatibility(other.getRows(), other.getCols());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                getElement(i, j) -= other.getElement(i, j);
            }
        }
        return (*this);
    }
    
    // Scalar Multiplication * Operator
    Matrix operator*(const T& scale) const {
        Matrix<T> newMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                newMatrix.getElement(i, j) = getElement(i, j) * scale;
            }
        }
        return newMatrix;
    }

    // Scalar Multiplication *= Operator
    Matrix& operator*=(const T& scale) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                (*this).getElement(i, j) *= scale;
            }
        }
        return (*this);
    }
    
    // Matrix Multiplication * Operator
    Matrix operator*(const Matrix& other) const {
        checkMultiplicationCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, other.getCols());

        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < other.getCols(); ++j) {
                float sum = 0;
                for(int k = 0; k < cols; ++k) {
                    sum += (*this).getElement(i, k) * other.getElement(k, j);
                }
                newMatrix.getElement(i, j) = sum;
            }
        }
        return newMatrix;
    }
    
    // Matrix Multiplication *= Operator
    Matrix& operator*=(const Matrix& other) {
        checkMultiplicationCompatibility(other.getRows(), other.getCols());
        Matrix<T> newMatrix(rows, other.getCols());
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < other.getCols(); ++j) {
                float sum = 0;
                for(int k = 0; k < cols; ++k) {
                    sum += (*this).getElement(i, k) * other.getElement(k, j);
                }
                newMatrix.getElement(i, j) = sum;
            }
        }
        rows = newMatrix.rows;
        cols = newMatrix.cols;
        mat = newMatrix.mat;
        return (*this);
    }
    
    // Transpose of Matrix
    void transpose() {
        Matrix<T> newMatrix(cols, rows);
        
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                newMatrix.getElement(j, i) = (*this).getElement(i, j);
            }
        }
        
        rows = newMatrix.rows;
        cols = newMatrix.cols;
        mat = newMatrix.mat;    
    }
    
};

template<typename T>
Matrix<T> operator*(const T& scale, const Matrix<T>& mat) {
    Matrix<T> newMatrix(mat.getRows(), mat.getCols());
    for(int i = 0; i < mat.getRows(); ++i) {
        for(int j = 0; j < mat.getCols(); ++j) {
            newMatrix.getElement(i, j) = scale * mat.getElement(i, j);
        }
    }
    return newMatrix;
}

template<typename T>
ostream& operator<<(ostream& stream, const Matrix<T>& mat) {
    for(int i = 0; i < mat.getRows(); ++i) {
        for(int j = 0; j < mat.getCols(); ++j) {
            stream << setw(4) << mat.getElement(i, j) << ' ';
        }
        stream << '\n';
    }
    return stream;
}

int main() {
    
    try {

        Matrix<int> mat1(2, 2, 1); // 2x2 matrix, all 1s
        Matrix<int> mat2(2, 2, 2); // 2x2 matrix, all 2s
        cout << "Matrix 1:" << endl << mat1;
        cout << "Matrix 2:" << endl << mat2;

        Matrix<int> mat3 = mat1 + mat2;
        cout << "Matrix 1 + Matrix 2:" << endl << mat3;

        mat3 -= mat1;
        cout << "Matrix 3 - Matrix 1:" << endl << mat3;

        Matrix<int> mat4 = mat1 * 3;
        cout << "Matrix 1 * 3:" << endl << mat4;

        Matrix<int> mat5 = 4 * mat1;
        cout << "4 * Matrix 1:" << endl << mat5;

        mat5 *= 2;
        cout << "Matrix 5 *= 2:" << endl << mat5;

        // Test matrix multiplication
        Matrix<int> mat6(2, 3, 1); // 2x3 matrix, all 1s
        Matrix<int> mat7(3, 2, 2); // 3x2 matrix, all 2s
        cout << "Matrix 6 (2x3):" << endl << mat6;
        cout << "Matrix 7 (3x2):" << endl << mat7;

        Matrix<int> mat8 = mat6 * mat7;
        cout << "Matrix 6 * Matrix 7:" << endl << mat8;

        mat8 *= mat7;
        cout << "Matrix 8 *= Matrix 7:" << endl << mat8;

        // Test invalid matrix multiplication (should throw exception)
        Matrix<int> mat9(2, 2);
        mat6 * mat9; // 2x3 * 2x2 should throw
        

    } catch (const std::invalid_argument& e) {
        cout << "Invalid Argument Error: " << e.what() << endl;
    } catch (const std::out_of_range& e) {
        cout << "Out of Range Error: " << e.what() << endl;
    } catch (...) {
        cout << "Unknown Error occurred" << endl;
    }
    
    return 0;
}
