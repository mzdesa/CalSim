"""
This file contains utilities for control design.
"""
import numpy as np

def ctrb(A, B):
    """
    Function to compute the controllability matrix of a system xDot = Ax + Bu
    Inputs:
        A (nxn NumPy Array): A matrix of a system
        B (nxm NumPy Array): B matrix of a system
    Returns:
        [B AB A^2B ... A^(n-1)B]
    """
    #initialize controllabiity matrix as B
    P = B
    for i in range(1, A.shape[0]):
        P = np.hstack((P, np.linalg.matrix_power(A, i) @ B))
    #return controllability matrix
    return P

def obsv(A, C):
    """
    Function to compute the observability matrix of a system xDot = Ax + Bu, y = Cx
    Inputs:
        A (nxn NumPy Array): A matrix of a system
        C (mxn NumPy Array): C matrix of a system
    Returns:
        [C; CA; CA^2; ...; CA^n-1]
    """
    #initialize controllabiity matrix as B
    O = C
    for i in range(1, A.shape[0]):
        P = np.vstack((P, C @ np.linalg.matrix_power(A, i)))
    #return controllability matrix
    return O

def is_ctrb(A, B):
    """
    Verify if (A, B) is a controllable pair. Returns true if controllable.
    """
    return np.linalg.matrix_rank(ctrb(A, B)) == A.shape[0]

def is_obsv(A, C):
    """
    Verify if (A, C) is an observable pair. Returns true if observable.
    """
    return np.linalg.matrix_rank(ctrb(A, C)) == A.shape[0]

def calc_cl_poles(A, B, K):
    """
    Function to calculate the closed loop poles of a system xDot = Ax + Bu
    using state feedback with gain matrix K.
    Inputs:
        A (nxn NumPy Array): A matrix
        B (nxm NumPy Array): B Matrix
        K (mxn NumPy Array): Gain matrix
    Returns:
        [lambda1, lambda2, ..., lambdan] (list of floats): closed loop poles of the system with gain K
    """
    return np.linalg.eigvals(A - B @ K).tolist()

def place_poles(A, B, pole_list):
    """
    Function to compute the gain K to place the poles of A - BK at desired posiions using Ackermann's formula.
    This function depends on invertibility of the controllability matrix.
    Inputs:
        A (nxn NumPy Array): A matrix
        B (nxm NumPy Array): B Matrix
        pole_list (list of n complex/float numbers): list of desired pole positions for the closed loop system
    Returns:
        K (mxn NumPy Array): State feedback gain matrix to place the poles of the system in the desired locations
    """
    #find the char. polyn of A
    char_poly = np.poly(np.linalg.eigvals(A))

    #find the desired char. polyn
    char_poly_des = np.poly(pole_list)

    #subtract the desired poles
    rowVec = (char_poly_des[1:] - char_poly[1:]).reshape((1, A.shape[0]))

    #compute the W terms
    Wr = ctrb(A, B)

    #assemble Wr tilda (inverse of toeplitz matrix)
    Wrt = np.eye(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j - i >= 0:
                #check the relation between i and j
                Wrt[i, j] = char_poly[j - i]
            else:
                Wrt[i, j] = 0
    Wrt = np.linalg.pinv(Wrt)

    #return the gain
    return rowVec @ np.linalg.pinv(Wr) @ Wrt