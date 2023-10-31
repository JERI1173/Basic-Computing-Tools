"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""

# Problem 0

# Part A...


# Problem 1
import numpy as np
# Part A
from scipy.linalg import cholesky, solve_triangular
def solve_chol(A, b):
    """Solve a linear algebra problem with Cholesky decomposition

    Parameters:
    - A: ndarray, SPD matrix
    - b: ndarray, vector

    Return
    - x: ndarray, vector
    """
    # Obtain L
    L = cholesky(A, lower = True) # A = L @ L.T
    
    # Obtain y 
    y = solve_triangular(L, b, lower=True) # L @ y = b

    # Obtain x 
    x = solve_triangular(L.T, y) # L.T @ x = y
    
    return x

# Part B

# Part C
from numpy.linalg import eigh
def matrix_pow(A, n):
    """Compute the power of a symmetric matrix with eigenvalue decomposition
    
    Parameters:
    - A: ndarray, symmetric matrix
    - n: int, power
    
    Return:
    - A_n: ndarray, the matrix A**n
    """
    # Compute eigenvalue decomposition of A
    L, Q = eigh(A) # A = Q @ L @ Q.T, L is a diagonal matrix

    # Compute L**n element-wise
    L_n = np.diag(L**n) # L is diagonal

    # Compute A**n using the eigenvalue decomposition
    A_n = Q @ L_n @ Q.T

    return A_n

# part D
from scipy.linalg import lu
import numpy.linalg as la

def abs_det(A):
    """Compute the determinant with LU decomposition
    
    Parameters:
    - A: ndarray, square matrix
    
    Return:
    - abs(det_A): int, absolute value of determinant of matrix A
    """
    P, L, U = lu(A) # A = P @ L @ U
    det_A = np.prod(np.diag(U)) # det(A) = det(P) * det(L) * det(U)
    
    return abs(det_A)

# Problem 2
class my_complex:
    def __init__(self, real, imag):
        self._real = real
        self._imag = imag

    def __add__(self, other):
        return my_complex(self._real + other._real, self._imag + other._imag)

    def __mul__(self, other):
        real_part = self._real * other._real - self._imag * other._imag
        imag_part = self._real * other._imag + self._imag * other._real
        return my_complex(real_part, imag_part)

    def conj(self):
        return my_complex(self._real, -self._imag)
    
    def real(self):
        return self._real
    
    def imag(self):
        return self._imag



def generate_complex_vector(n):
    """Generate a random element of C^n using a list of instances of my_complex

    Parameters
    -n: int, dimension of the complex vector

    Returns
    -vector: list, n instances of my_complex
    """
    # Create a vector where the real part increases by 1, while the "imaginary" part decreases by 1
    vector = [my_complex(i+1, -(i+1)) for i in range(n)]
    
    return vector

def complex_dot_product(v1, v2):
    """Compute the dot product of two complex vectors

    Parameters
    -v1, v2: list, lists of instances of my_complex representing the vectors

    Returns
    -dot_prod: ?, the dot product of v1 and v2
    """
    # Initialize the dot product with a zero complex number
    dot_prod = my_complex(0, 0)

    # Compute the dot product taking into account the complex conjugate
    for i in range(len(v1)):
        dot_prod += v1[i].conj() * v2[i]

    return dot_prod

import numpy as np
import matplotlib.pyplot as plt
import time

# Generate vector using my_complex, and compute its norm
def generate_norm_my_complex(n):
    vector = generate_complex_vector(n)
    norm_squared = my_complex(0, 0)
    for z in vector:
        norm_squared += z.conj() * z
    norm = norm_squared.get_real() ** 0.5
    return norm

# Generate vector using numpy.cdouble, and compute its norm
def generate_norm_numpy(n):
    vector = np.array([complex(i+1, -(i+1)) for i in range(n)], dtype=np.cdouble)
    norm = np.linalg.norm(vector)
    return norm