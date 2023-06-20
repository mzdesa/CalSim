"""
This file includes various classes and functions involved in rigid body transformations.
"""
import numpy as np
from exceptions import *

def hat_3d(w):
    """
    Function to compute the hat map of a 3x1 vector omega.
    Inputs:
        w (3x1 NumPy Array): vector to compute the hat map of
    Returns:
        w_hat (3x3 NumPy Array): skew symmetrix hat map matrix
    """
    #reshape the w vector to be a 3x1
    w = w.reshape((3, 1))
    #compute and return its hat map
    w_hat = np.array([[0, -w[2, 0], w[1, 0]], 
                      [w[2, 0], 0, -w[0, 0]], 
                      [-w[1, 0], w[0, 0], 0]])
    return w_hat

def hat_6d(xi):
    """
    Function to compute the hat map of a 6x1 twist xi
    Inputs:
        xi (6x1 NumPy Array): (v, omega) twist
    Returns:
        xi_hat (4x4 NumPy Array): hat map matrix
    """
    #reshpae to a 6x1 Vector
    xi = xi.reshape((6, 1))
    #compute the hat map of omega
    w_hat = hat_3d(xi[3:])
    #extract and reshape v
    v = xi[0:3].reshape((3, 1))
    #compute and return the hat map of xi
    xi_hat = np.hstack((w_hat, v))
    xi_hat = np.vstack((xi_hat, np.zeros((1, 4))))
    return xi_hat

def hat(x):
    """
    Function to compute the hat map of a 6x1 or 3x1 vector x
    Inputs:
        x (6x1 or 3x1 NumPy Array)
    Returns:
        x_hat: hat map of the vector x
    """
    if x.size == 3:
        return hat_3d(x)
    elif x.size == 6:
        return hat_6d(x)
    else:
        #raise error: input vector is of incorrect shape
        raise ShapeError()

def rodrigues(w, theta):
    """
    Function to compute the matrix exponential of an angular velocity vector
    using rodrigues' formula.
    Inputs:
        w (3x1 NumPy Array): angular velocity vector (may be unit or not unit)
        theta (float): angle of rotation in radians
    Returns:
        exp(w_hat*theta): rotation matrix associated with w and theta
    """
    #check shape of w
    if w.size != 3:
        raise ShapeError()
    
    #reshape w
    w = w.reshape((3, 1))

    #compute Rodrigues formula (using the non-unit w assumption)
    wNorm = np.linalg.norm(w)
    wHat = hat(w)
    exp_w_theta = np.eye(3) + (wHat/wNorm)*np.sin(wNorm*theta) + (wHat@wHat)/(wNorm**2)*(1-np.cos(wNorm*theta))
    
    #return matrix exponential
    return exp_w_theta

def calc_Rx(phi):
    """
    Function to copute the X Euler angle rotation matrix
    Inputs:
        phi (float): angle of rotation
    Returns:
        R_x(phi) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[1, 0, 0]]), phi)

def calc_Ry(beta):
    """
    Function to copute the Y Euler angle rotation matrix
    Inputs:
        beta (float): angle of rotation
    Returns:
        R_y(beta) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[0, 1, 0]]), beta)

def calc_Rz(alpha):
    """
    Function to copute the Z Euler angle rotation matrix
    Inputs:
        alpha (float): angle of rotation
    Returns:
        R_z(alpha) (3x3 NumPy Array)
    """
    return rodrigues(np.array([[0, 0, 1]]), alpha)

def calc_Rzyz(alpha, beta, gamma):
    """
    Calculate a rotation matrix based on ZYZ Euler angles
    Inputs:
        alpha, beta, gamma: rotation angles
    Returns:
        Rz(alpha)Ry(beta)Rz(gamma)
    """
    return calc_Rz(alpha)@calc_Ry(beta)@calc_Rz(gamma)

def calc_Rzyx(psi, theta, phi):
    """
    Calculate a rotation matrix based on ZYX Euler angles
    Inputs:
        psi, theta, phi: rotation angles
    Returns:
        Rz(psi)Ry(theta)Rx(phi)
    """
    return calc_Rz(psi)@calc_Ry(theta)@calc_Rx(phi)


def quat_2_rot(Q):
    """
    Calculate the rotation matrix associated with a unit quaternion Q
    Inputs:
        Q (4x1 NumPy Array): [q, q0] unit quaternion, where q is 3x1 and q0 is scalar
    Returns:
        R (3x3 NumPy Array): rotation matrix associated with quaternion
    """
    #extract the quaternion components
    b, c, d, a = Q.reshape((4, ))
    #compute and return the rotation matrix
    R = np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c], 
                  [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b], 
                  [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]])
    return R