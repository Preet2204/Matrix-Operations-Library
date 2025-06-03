# Matrix Operations Library

## Overview
The Matrix Operations Library is a C++ template-based library designed to perform fundamental matrix operations. It provides a `Matrix<T>` class that supports arithmetic operations, transposition, and LU decomposition, with plans to implement determinant and inverse calculations. The library ensures type safety, bounds checking, and numerical stability, making it suitable for educational and research purposes in linear algebra and numerical computations.

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
  - Decomposes a square matrix into lower (\( L \)) and upper (\( U \)) triangular matrices with partial pivoting, returning a permutation matrix \( P \) such that \( PA = LU \).
  - Ensures numerical stability by handling near-singular matrices.
- **Error Handling**:
  - Throws exceptions for invalid dimensions, out-of-bounds access, and singular matrices during LU decomposition.
- **Type Safety**:
  - Supports numeric types (e.g., `int`, `double`) via `static_assert` to exclude non-arithmetic types.

## Planned Features
- **Determinant Calculation**:
  - Compute the determinant using LU decomposition.
- **Matrix Inverse**:
  - Compute the inverse using LU decomposition with forward and back substitution.

## Requirements
- A C++ compiler supporting C++11 or later (e.g., GCC, Clang, MSVC).
- Standard C++ libraries (no external dependencies).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/matrix-operations-library.git
   ```
2. Include the `matrix_operations.cpp` file in your project.
3. Compile your program with a C++ compiler:
   ```bash
   g++ your_program.cpp -o your_program
   ```

## Usage
Below is an example demonstrating how to use the library to perform matrix operations and LU decomposition.

```cpp
#include <iostream>
#include "matrix_operations_with_lu_improved.cpp"

int main() {
    try {
        // Create two 2x2 matrices
        Matrix<double> mat1(2, 2, 1.0); // All elements are 1.0
        Matrix<double> mat2(2, 2, 2.0); // All elements are 2.0

        // Addition
        Matrix<double> mat3 = mat1 + mat2;
        std::cout << "Matrix 1 + Matrix 2:" << std::endl << mat3;

        // Scalar multiplication
        Matrix<double> mat4 = mat1 * 3.0;
        std::cout << "Matrix 1 * 3:" << std::endl << mat4;

        // Transpose
        mat1.transpose();
        std::cout << "Matrix 1 after transpose:" << std::endl << mat1;

        // LU decomposition
        Matrix<double> mat5(3, 3);
        mat5.getElement(0, 0) = 1; mat5.getElement(0, 1) = 2; mat5.getElement(0, 2) = 3;
        mat5.getElement(1, 0) = 4; mat5.getElement(1, 1) = 5; mat5.getElement(1, 2) = 6;
        mat5.getElement(2, 0) = 7; mat5.getElement(2, 1) = 8; mat5.getElement(2, 2) = 10;
        auto [L, U, perm] = mat5.LU();
        std::cout << "Lower triangular matrix L:" << std::endl << L;
        std::cout << "Upper triangular matrix U:" << std::endl << U;
        std::cout << "Permutation Matrix:" << std::endl << perm;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    return 0;
}
```

## Limitations
- Currently supports only numeric types (excluding `bool`).
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