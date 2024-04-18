# Numerical Methods for Linear Systems

### Overview

This Python script implements several numerical methods for solving systems of linear equations. It provides implementations for the Jacobi Method, Gauss-Seidel Method, Successive Over-Relaxation (SOR) Method, and Iterative Refinement Method. Additionally, the script includes a direct solution using NumPy's linear solver for comparison purposes.

### Methods Implemented

- **Jacobi Method:** An iterative technique for solving the diagonal entries of a matrix equation.
- **Gauss-Seidel Method:** An improved version of the Jacobi Method that uses the latest updated values for convergence.
- **SOR Method:** Extends the Gauss-Seidel method by using a relaxation factor to potentially accelerate the convergence.
- **Iterative Refinement Method:** Refines a given solution by iteratively correcting the residual error.
- **Direct Solution (for verification and not asked in the assignment):** Uses NumPy's linalg.solve function to compute the exact solution, providing a benchmark for verifying the iterative solutions.

### Requirements

Requirements will be found in the requirements.txt file in the repository

### Usage

To use the script, ensure that Python 3 and NumPy are installed on your system. You can run the script from the command line as follows:

python main.py

### Output

The results are saved to numerical_methods_output.txt in the same directory as the script. The output file will contain the results for each method, formatted clearly for easy comparison.

### Conclusion

This script is useful for educational purposes to understand different iterative methods for solving linear systems and for verifying the convergence and accuracy of these methods against a direct solution.
