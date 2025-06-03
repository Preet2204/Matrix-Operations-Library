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