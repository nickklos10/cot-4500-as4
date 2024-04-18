import numpy as np

# Jacobi Iterative Method
def jacobi_method(A, b, x0=None, max_iterations=1000, tolerance=1e-10):
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)

    x = np.copy(x0)
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            break
        x = x_new

    return x

# Gauss-Seidel Iterative Method 
def gauss_seidel_method(A, b, x0=None, max_iterations=1000, tolerance=1e-10):
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)

    x = np.copy(x0)
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            break
        x = x_new

    return x

# SOR Method
def sor_method(A, b, omega, x0=None, max_iterations=1000, tolerance=1e-10):
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b)

    x = np.copy(x0)
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - s1 - s2) / A[i, i] - x[i])

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            break
        x = x_new

    return x

# Iterative Refinement Method
def iterative_refinement(A, b, x0, max_iterations=10, tolerance=1e-10):
    x = np.copy(x0)
    for _ in range(max_iterations):
        r = b - np.dot(A, x)  # Compute residual
        delta_x = np.linalg.solve(A, r)  # Solve correction equation
        x_new = x + delta_x  # Update solution
        if np.linalg.norm(delta_x, ord=np.inf) < tolerance:
            break
        x = x_new
    return x

# Example matrix (diagonal dominant) and vector
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 4]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

# Initial guess
x0 = np.array([0, 0, 0], dtype=float)

# Test the methods
jacobi_result = jacobi_method(A, b, x0)
gauss_seidel_result = gauss_seidel_method(A, b, x0)
sor_result = sor_method(A, b, omega=1.25, x0=x0)
iterative_refinement_result = iterative_refinement(A, b, x0)
direct_result = np.linalg.solve(A, b)

# Output results to a file
output_filename = "numerical_methods_output.txt"
with open(output_filename, "w") as file:
    file.write("Jacobi Method Result:\n")
    file.write(np.array2string(jacobi_result) + "\n")
    file.write("Gauss-Seidel Method Result:\n")
    file.write(np.array2string(gauss_seidel_result) + "\n")
    file.write("SOR Method Result:\n")
    file.write(np.array2string(sor_result) + "\n")
    file.write("Iterative Refinement Method Result:\n")
    file.write(np.array2string(iterative_refinement_result) + "\n")
    file.write("Direct Method Result:\n")
    file.write(np.array2string(direct_result) + "\n")