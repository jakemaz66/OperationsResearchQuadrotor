import numpy as np
import scipy.io
from pathlib import Path

#variables:
#Mass of the quadrotor (kilograms)
Mq = 0.6
#Arm length of the quadrotor (meters)
L = (0.2159/2)
#Acceleration due to gravity
g = 9.81
#Mass of the central sphere of the quadrotor (Kilograms)
Ms = 0.410
#Radius of the central sphere (meters)
R = 0.0503513
#Mass of the propellor (kilograms)
Mprop = 0.00311
#Mass of motor + propellor
Mm = 0.036 + Mprop
#Moment of inertia for x direction
Jx = ((2 * Ms * R**2) / 5) + (2 * L**2 * Mm)
#Moment of inertia for y direction
Jy = ((2 * Ms * R**2) / 5) + (2 * L**2 * Mm)
#Moment of inertia for z direction
Jz = ((2 * Ms * R**2) / 5) + (4 * L**2 * Mm)
m = Mq
# A is a 12x12 matrix used for linear simulation (given in slide 35, lecture 3)
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  -g, 0,  0],
    [1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, g,  0,  0,  0,  0],
    [0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0]
]) 

# B is a 12x4 matrix used for linear simulation (given in slide 35)
B = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [-1/m, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1/Jx, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1/Jy, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1/Jz],
    [0, 0, 0, 0]
])

C = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

#The control matrix K
ControllerMatrix = scipy.io.loadmat(Path("ControllerMatrices") / "K.mat")
K = ControllerMatrix['K']
#The integral control matrix Kc
ControllerMatrixC = scipy.io.loadmat(Path("ControllerMatrices") / "Kc.mat")
Kc = ControllerMatrixC['Kc']