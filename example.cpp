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
        std::cout << "Frobenius Norm of Matrix J: " << norm << "\n\n";

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
            std::cout << "Caught expected error: " << e.what() << "\n\n";
        }
        
        // 7. Eigenvalue Computation
        std::cout << "=== Eigenvalue Computation ===\n";
        Matrix<double> M({{5, 0, 0}, {0, 2, 0}, {0, 0, 4}});
        std::cout << "M Matrix:\n" << M;
        std::cout << "Eigenvalue of M = " << M.eigenvalue() << "\n\n";
        
        std::cout << "The End!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}