# Matrix Operations Library

## Overview

The Matrix Operations Library is a C++ template-based library designed to perform fundamental matrix operations. It provides a `Matrix<T>` class that supports arithmetic operations, transposition, LU decomposition, determinant calculation, and matrix inversion. The library ensures type safety, bounds checking, and numerical stability, making it suitable for educational and research purposes in linear algebra and numerical computations.

## Features

- **Matrix Construction**:
  - Create matrices with specified dimensions and initial values.
  - Initialize matrices from 2D vectors.
- **Arithmetic Operations**:
  - Addition (`+`, `+=`) and subtraction (`-`, `-=`) of matrices with compatible dimensions.
  - Scalar multiplication (`*`, `*=`), supporting both `Matrix * scalar` and `scalar * Matrix`.
  - Matrix multiplication (`*`, `*=`) for compatible matrices.
- **Transpose**:
  - In-place transposition of a matrix, swapping rows and columns.
- **LU Decomposition**:
  - Decomposes a square matrix into lower (( L )) and upper (( U )) triangular matrices with partial pivoting, returning a permutation matrix ( P ) and swap count such that ( PA = LU ).
- **Determinant Calculation**:
  - Computes the determinant using LU decomposition, with sign adjustment based on permutation swaps.
- **Matrix Inversion**:
  - Computes the inverse of a non-singular square matrix using LU decomposition with forward and back substitution.
- **Error Handling**:
  - Throws exceptions for invalid dimensions, out-of-bounds access, incompatible operations, and singular matrices.
- **Type Safety**:
  - Supports numeric types (e.g., `int`, `double`) via `static_assert` to exclude non-arithmetic types.

## Requirements

- A C++ compiler supporting C++11 or later (e.g., GCC, Clang, MSVC).
- Standard C++ libraries (no external dependencies).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/matrix-operations-library.git
   ```
2. Include the `matrix.hpp` header file in your project.
3. Compile your program with a C++ compiler:

   ```bash
   g++ your_program.cpp -o your_program
   ```

## Usage

Below is an example demonstrating how to use the library to perform matrix operations, compute the determinant, and find the inverse.

```cpp
#include <iostream>
#include "matrix.hpp"

int main() {
    try {
        // 1. Creating matrices using different constructors
        std::cout << "=== Matrix Creation ===\n";

        // Create a 2x3 matrix initialized to 0
        Matrix<double> A(2, 3);
        std::cout << "2x3 Zero Matrix A:\n" << A << "\n";

        // Create a 3x2 matrix initialized to 5.0
        Matrix<double> B(3, 2, 5.0);
        std::cout << "3x2 Matrix B (all elements 5.0):\n" << B << "\n";

        // Create a matrix from a 2D vector
        std::vector<std::vector<double>> vec = {{1, 2, 3}, {4, 5, 6}};
        Matrix<double> C(vec);
        std::cout << "2x3 Matrix C from 2D vector:\n" << C << "\n";

        // 2. Basic Matrix Operations
        std::cout << "=== Basic Operations ===\n";

        // Matrix addition
        Matrix<double> D(vec); // Copy of C
        std::cout << "Matrix D = C\n";
        Matrix<double> E = C + D;
        std::cout << "Matrix E = C + D:\n" << E << "\n";

        // Matrix subtraction
        Matrix<double> F = C - D;
        std::cout << "Matrix F = C - D:\n" << F << "\n";

        // Scalar multiplication
        Matrix<double> G = C * 2.0;
        std::cout << "Matrix G = C * 2.0:\n" << G << "\n";

        // Matrix multiplication
        Matrix<double> H = C * B; // 2x3 * 3x2 = 2x2
        std::cout << "Matrix H = C * B (2x3 * 3x2):\n" << H << "\n";

        // 3. Transpose
        std::cout << "=== Transpose ===\n";
        Matrix<double> I = C;
        I.transpose();
        std::cout << "Transpose of Matrix C = I (2x3 -> 3x2):\n" << I << "\n";

        // 4. Advanced Operations (Determinant, Inverse, LU, QR)
        std::cout << "=== Advanced Operations ===\n";

        // Create a 3x3 matrix for determinant and inverse
        Matrix<double> J({{4, 3, 2}, {1, 2, 3}, {2, 1, 3}});
        std::cout << "3x3 Matrix J:\n" << J << "\n";

        // Determinant
        double det = J.determinant();
        std::cout << "Determinant of Matrix J: " << det << "\n\n";

        // Inverse
        Matrix<double> J_inv = J.inverse();
        std::cout << "Inverse of Matrix J:\n" << J_inv << "\n";

        // Verify inverse by multiplying J * J_inv (should be identity)
        Matrix<double> identity_check = J * J_inv;
        std::cout << "J * J_inv (should be identity):\n" << identity_check << "\n";

        // LU Decomposition
        auto [L, U, P, swapCount] = J.LU();
        std::cout << "LU Decomposition of J:\n";
        std::cout << "L Matrix:\n" << L << "\n";
        std::cout << "U Matrix:\n" << U << "\n";
        std::cout << "P Matrix:\n" << P << "\n";
        std::cout << "Swap Count: " << swapCount << "\n\n";

        // QR Decomposition
        auto [Q, R] = J.QR();
        std::cout << "QR Decomposition of J:\n";
        std::cout << "Q Matrix:\n" << Q << "\n";
        std::cout << "R Matrix:\n" << R << "\n";
        std::cout << "Q * R:\n" << Q * R << "\n";

        // 5. Inner Product and Frobenius Norm
        std::cout << "=== Inner Product and Norm ===\n";

        // Inner product of columns 0 and 1 of Matrix J
        double inner = J.innerProduct(0, 1);
        std::cout << "Inner product of columns 0 and 1 of J: " << inner << "\n";

        // Inner product of column 0 of J and column 0 of C
        double inner_cross = J.innerProduct(I, 0, 0);
        std::cout << "Inner product of column 0 of J and column 0 of I: " << inner_cross << "\n";

        // Frobenius Norm
        double norm = J.frobeniusNorm();
        std::cout << "Frobenius Norm of Matrix J: " << norm << "\n";

        // 6. Exception Handling Example
        std::cout << "=== Exception Handling ===\n";
        try {
            Matrix<double> K(2, 2);
            K.getElement(3, 0); // Should throw out_of_range
        } catch (const std::out_of_range& e) {
            std::cout << "Caught expected error: " << e.what() << "\n";
        }

        try {
            Matrix<double> L(2, 3);
            L.determinant(); // Should throw invalid_argument (not square)
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught expected error: " << e.what() << "\n";
        }
        
        std::cout << "The End!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```
### Output

```bash
=== Matrix Creation ===
2x3 Zero Matrix A:
0 0 0
0 0 0

3x2 Matrix B (all elements 5.0):
5 5
5 5
5 5

2x3 Matrix C from 2D vector:
1 2 3
4 5 6

=== Basic Operations ===
Matrix D = C
Matrix E = C + D:
2 4  6
8 10 12

Matrix F = C - D:
0 0 0
0 0 0

Matrix G = C * 2.0:
2 4  6
8 10 12

Matrix H = C * B (2x3 * 3x2):
30 30
75 75

=== Transpose ===
Transpose of Matrix C = I (2x3 -> 3x2):
1 4
2 5
3 6

=== Advanced Operations ===
3x3 Matrix J:
4 3 2
1 2 3
2 1 3

Determinant of Matrix J: 15

Inverse of Matrix J:
0.2  -0.466667 0.333333
0.2  0.533333  -0.666667
-0.2 0.133333  0.333333

J * J_inv (should be identity):
1  0 0
-0 1 0
0  0 1

LU Decomposition of J:
L Matrix:
1    0    0
0.25 1    0
0.5  -0.4 1

U Matrix:
4 3    2
0 1.25 2.5
0 0    3

P Matrix:
1 0 0
0 1 0
0 0 1

Swap Count: 0

QR Decomposition of J:
Q Matrix:
0.872872 -0.0354   -0.486664
0.218218 0.92039   0.324443
0.436436 -0.389396 0.811107

R Matrix:
4.582576 3.491486 3.709704
0        1.345185 1.522183
0        0        2.433321

Q * R:
4 3 2
1 2 3
2 1 3

=== Inner Product and Norm ===
Inner product of columns 0 and 1 of J: 16
Inner product of column 0 of J and column 0 of I: 12
Frobenius Norm of Matrix J: 7.54983
=== Exception Handling ===
Caught expected error: Matrix Index Out of Bounds: i=3, j=0
Caught expected error: Matrix should be Square (For Determinant/Inverse): 2x3
The End!
```

## Error Handling

The library uses exception-based error handling to ensure correctness:

- `std::invalid_argument`:
  - Invalid dimensions during matrix construction (e.g., non-positive rows/columns).
  - Incompatible dimensions for arithmetic operations (e.g., addition, multiplication).
  - Non-square or singular matrices for operations requiring invertibility (e.g., inverse).
- `std::out_of_range`:
  - Out-of-bounds access when using `getElement`.
- `std::runtime_error`:
  - Singular or near-singular matrices during LU decomposition.

Users are encouraged to use `try/catch` blocks to handle these exceptions, especially when inputs are dynamic or user-provided.


## Limitations
- LU decomposition requires the matrix to be square and non-singular (or not near-singular).
- Determinant and inverse calculations are under development.

## Contributing
Contributions are welcome! Please submit issues or pull requests to the GitHub repository. Ensure that your code follows the existing style and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
- Preet Siddhapura (GitHub: Preet2204)

## Acknowledgments
- Inspired by standard linear algebra libraries like Eigen and NumPy.
- Developed as part of a B. Tech. project in Mathematics and Computing.
