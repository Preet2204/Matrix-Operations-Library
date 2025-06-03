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
        // Create a 3x3 matrix: [1 2 3; 4 5 6; 7 8 10]
        Matrix<double> mat(3, 3);
        mat.getElement(0, 0) = 1; mat.getElement(0, 1) = 2; mat.getElement(0, 2) = 3;
        mat.getElement(1, 0) = 4; mat.getElement(1, 1) = 5; mat.getElement(1, 2) = 6;
        mat.getElement(2, 0) = 7; mat.getElement(2, 1) = 8; mat.getElement(2, 2) = 10;

        // Print the matrix
        std::cout << "Matrix:" << std::endl << mat;

        // Compute and print the determinant
        std::cout << "Determinant: " << mat.determinant() << std::endl;

        // Compute and print the inverse
        Matrix<double> inv = mat.inverse();
        std::cout << "Inverse:" << std::endl << inv;

        // Verify inverse by multiplying: should be identity
        Matrix<double> identityCheck = mat * inv;
        std::cout << "Matrix * Inverse (should be identity):" << std::endl << identityCheck;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    return 0;
}
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