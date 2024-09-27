import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from mpl_toolkits.mplot3d import Axes3D


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

#The control matrix K
ControllerMatrix = scipy.io.loadmat("ControllerMatrices\K.mat")
K = ControllerMatrix['K']
#The integral control matrix Kc
ControllerMatrixC = scipy.io.loadmat("ControllerMatrices\Kc.mat")
Kc = ControllerMatrixC['Kc']


def nonlinear_dynamics( t, curstate, A, B, target_state):
    """Modeling the non-linear dynamics of the quadrotor. The non-linear dynamics tell you how
       the state of the quadrotor changes with regards to three things
       1. The current state of the quadrotor
       2. The input controls into the quadrotor
       3. The current forces of physics

    Args:
        state (array): A vector of the current state of the quadrotor
        control_input (dictionary): A dictionary defining the input forces, will get transformed into vector

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor
    """

    u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r = curstate

    # Initialize an empty list to store the errors
    error = []
    
    # Iterate exactly 12 times (assuming target_state and curstate have at least 12 elements)
    for i in range(12):
        target = target_state[i]
        state = curstate[i]

        # Compute the difference between target and state
        difference = state - target


        # Append the result to the error list
        error.append(difference)

    #Controlling by multiplying error by negative feedback matrix K "−K(x−xeq)"
    control = -K @ (error)
    #print(f"Time: {t}, State: {curstate}")

    F = control[0] + Mq * g # Force -> Throttle
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque 
    TauPsi = control[3]     # Yaw Torque
    #print(control)

    #Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
    uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (g * np.cos(theta) * np.cos(phi)) + (-F / m)

    #Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
    pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
    qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
    rDot = (((Jx - Jy) / Jz) * p * q) + ((1 / Jz) * TauPsi)

    PxDot = u
    PyDot = v
    PzDot = w

    # u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r
    return [uDot, PxDot, vDot, PyDot, wDot, PzDot, pDot, p, qDot,q, rDot, r]

    
def linear_dynamics(t, curstate, A, B, target_state):
    """Modeling the linear dynamics of the quadrotor. The linear dynamics tell you how
       the state of the quadrotor changes with regards to two things
       1. The current state of the quadrotor
       2. The input controls into the quadrotor

    Args:
        t (list): The time interval
        state (array): A vector of the current state of the quadrotor
        control_input (array): A vector of the input forces into the quadrotor, size 1X4
        A (nd-array): A given matrix for state linear dynamics (given in slides)
        B (nd-array): A given matrix for input linear dynamics (given in slides)

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor
    """

    u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r = curstate

    # Initialize an empty list to store the errors
    error = []

    # Iterate exactly 12 times (assuming target_state and curstate have at least 12 elements)
    for i in range(12):
        target = target_state[i]
        state = curstate[i]

        # Compute the difference between target and state
        difference = state - target
        #print(target, " ", state)

        # Append the result to the error list
        error.append(difference)

    # Controlling by multiplying error by negative feedback matrix K "−K(x−xeq)"
    control = -K @ (error)
    # print(target_state)

    # The change in state (xDot) is equal to Ax + Bu (@ is matrix-multiplication operator)
    # x in this equation is equal to the current state and u is equal to the control control_input
    xDot = (A @ curstate) + B @ control
    return xDot


def linear_dynamics_integral(t, curstate, A, B, target_state):
    """Modeling the linear dynamics of the quadrotor with integral control. The linear dynamics tell you how
       the state of the quadrotor changes with regards to two things
       1. The current state of the quadrotor
       2. The input controls into the quadrotor

    Args:
        t (list): The time interval
        state (array): A vector of the current state of the quadrotor
        control_input (array): A vector of the input forces into the quadrotor, size 1X4
        A (nd-array): A given matrix for state linear dynamics (given in slides)
        B (nd-array): A given matrix for input linear dynamics (given in slides)

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor
    """

    u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r = curstate

    #Putting together 4x1 error vector
    errorX = target_state[0] - curstate[0]
    errorY = target_state[1] - curstate[1]
    errorZ = target_state[2] - curstate[2]
    errorPsi = target_state[8] - curstate[8]

    error = np.array([errorX, errorY, errorZ, errorPsi]).reshape(4, 1)
    
    #Obtaining integral output
    control = (-K @ curstate) + (Kc @ error).reshape(1, 4)[0]
    
    xDot = (A @ curstate) + (B @ control)

    return xDot


def simulate_quadrotor_linear_integral_controller(control_input, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span = [0, 10]):
    """This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but uses the integral controller matrix Kc

    Args:
        control_input (dict): A dictionary of the input force, roll, pitch, and yaw
    """



    #The control_input into the quadrotor
    #These are the coordinates we want the quadrotor to go to and hover at (the desired state)
    #The angular velocities and attitude are in radians so put accordingly. 
    desired_coordinates = [
                0, 0, 8,     # Position: x, y, z
                0, 0, 0,     # Linear velocity: x, y, z
                np.pi, 0, 0, # Orientation: roll = 180 deg, pitch = 0, yaw = 0
                0, 0, 0]

    control_input = integral_controller(desired_coordinates, initial_state, K, Kc)

    #A is a 12x12 matrix used for linear simulation (given in slide 35, lecture 3)
    # Change to make it same as on slide.?
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

    #B is a 12x4 matrix used for linear simulation (given in slide 35)
    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1/m, 0, 0, 0],
        [0, 1/Jx, 0, 0],
        [0, 0, 1/Jy, 0],
        [0, 0, 0, 1/Jz],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    sol_linear = solve_ivp(linear_dynamics, time_span, control_input, args=(A, B, target_state) , dense_output=True)
    
    plt.figure()
    #plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Nonlinear z(t)')
    plt.plot(sol_linear.t, sol_linear.y[2], label='Linear z(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.show()

    x, y, z = sol_linear.y[0], sol_linear.y[1], sol_linear.y[2]
    roll, pitch, yaw = sol_linear.y[3], sol_linear.y[4], sol_linear.y[5]
    t = sol_linear.t  

    fig = plt.figure(figsize=(12, 8))

    #3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='3D Trajectory')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('Quadrotor 3D Trajectory')
    ax1.legend()

    #Position over Time (x, y, z)
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)')
    ax2.plot(t, y, label='y(t)')
    ax2.plot(t, z, label='z(t)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs. Time')
    ax2.legend()

    #Orientation over Time (roll, pitch, yaw)
    ax3 = fig.add_subplot(223)
    ax3.plot(t, roll, label='Roll (rad)')
    ax3.plot(t, pitch, label='Pitch (rad)')
    ax3.plot(t, yaw, label='Yaw (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (rad)')
    ax3.set_title('Orientation vs. Time')
    ax3.legend()

    #Height over Time (for hovering behavior)
    ax4 = fig.add_subplot(224)
    ax4.plot(t, z, label='Height (z)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Height vs. Time')
    ax4.legend()

    plt.tight_layout()
    plt.show()



def simulate_quadrotor_linear_controller(target_state, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span = [0, 10]):
    """This function simulates the quadrotor moving from a an initial state, taking off to a desired state,
       and hovering around this desired point using only the feedback matrix K of the feedforward controller.
       This function models the change in the quadrotor's movement using only linear dynamics.

    Args:
        initial_state (array): The initial state of the quadrotor,defaul 0
        time_span (array)[start, end]: The time span over which to model the flight of the quadrotor, default 0, 10
        target_state (array): Goal state of the drone. 
    """
    
    #A is a 12x12 matrix used for linear simulation (given in slide 35, lecture 3)
    # Change to make it same as on slide.?
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

    #B is a 12x4 matrix used for linear simulation (given in slide 35)
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
    
    
    sol_linear = solve_ivp(linear_dynamics_integral, time_span, initial_state, args=(A, B, target_state) , dense_output=True)

    print("HERE")
    print()
    print(sol_linear)


    # Obtain results from the solved differential equations
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]  # x, y, z positions
    roll, pitch, yaw = sol_linear.y[7], sol_linear.y[9], sol_linear.y[11]  # roll, pitch, yaw
    t = sol_linear.t  # Time

    # Create plots and figures
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Simulating the Quadrotor with Linear Dynamics and Feedforward Controller. "
                f"Desired State is X:{target_state[1]}, Y:{target_state[3]}, Z:{target_state[5]}")

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='3D Trajectory', color='b', linewidth=2, marker='o')
    ax1.set_title('Quadrotor 3D Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True)

    # Position over Time (x, y, z)
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)', color='r', linestyle='--', marker='s', markersize=4)
    ax2.plot(t, y, label='y(t)', color='g', linestyle='-.', marker='^', markersize=4)
    ax2.plot(t, z, label='z(t)', color='b', linestyle='-', marker='o', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('Position vs. Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True)

    # Orientation over Time (roll, pitch, yaw)
    ax3 = fig.add_subplot(223)
    ax3.plot(t, roll, label='Roll (rad)', color='r', linestyle='-', marker='s', markersize=4)
    ax3.plot(t, pitch, label='Pitch (rad)', color='g', linestyle='--', marker='^', markersize=4)
    ax3.plot(t, yaw, label='Yaw (rad)', color='b', linestyle='-.', marker='o', markersize=4)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Orientation (rad)', fontsize=12)
    ax3.set_title('Orientation vs. Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True)

    # Height over Time (for hovering behavior)
    ax4 = fig.add_subplot(224)
    ax4.plot(t, z, label='Height (z)', color='m', linestyle='-', marker='o', markersize=4)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Height (m)', fontsize=12)
    ax4.set_title('Height vs. Time', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout(pad=3.0)

    # Show the figure
    plt.show()


def simulate_quadrotor_nonlinear_controller(target_state, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span=[0,10]):
    """
    This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but using non-linear dynamics

    Args:
        control_input (dict): A dictionary of the input force, roll, pitch, and yaw
    """
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

    #B is a 12x4 matrix used for linear simulation (given in slide 35)
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


    sol_nonlinear = solve_ivp(nonlinear_dynamics, time_span, initial_state, args=(A, B, target_state), dense_output=True)


    # Obtain results from the solved differential equations
    x, y, z = sol_nonlinear.y[1], sol_nonlinear.y[3], sol_nonlinear.y[5]  # x, y, z positions
    roll, pitch, yaw = sol_nonlinear.y[7], sol_nonlinear.y[9], sol_nonlinear.y[11]  # roll, pitch, yaw
    t = sol_nonlinear.t  # Time

    # Create plots and figures
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Simulating the Quadrotor with Linear Dynamics and Feedforward Controller. "
                f"Desired State is X:{target_state[1]}, Y:{target_state[3]}, Z:{target_state[5]}")

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='3D Trajectory', color='b', linewidth=2, marker='o')
    ax1.set_title('Quadrotor 3D Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True)

    # Position over Time (x, y, z)
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)', color='r', linestyle='--', marker='s', markersize=4)
    ax2.plot(t, y, label='y(t)', color='g', linestyle='-.', marker='^', markersize=4)
    ax2.plot(t, z, label='z(t)', color='b', linestyle='-', marker='o', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('Position vs. Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True)

    # Orientation over Time (roll, pitch, yaw)
    ax3 = fig.add_subplot(223)
    ax3.plot(t, roll, label='Roll (rad)', color='r', linestyle='-', marker='s', markersize=4)
    ax3.plot(t, pitch, label='Pitch (rad)', color='g', linestyle='--', marker='^', markersize=4)
    ax3.plot(t, yaw, label='Yaw (rad)', color='b', linestyle='-.', marker='o', markersize=4)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Orientation (rad)', fontsize=12)
    ax3.set_title('Orientation vs. Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True)

    # Height over Time (for hovering behavior)
    ax4 = fig.add_subplot(224)
    ax4.plot(t, z, label='Height (z)', color='m', linestyle='-', marker='o', markersize=4)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Height (m)', fontsize=12)
    ax4.set_title('Height vs. Time', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout(pad=3.0)

    # Show the figure
    plt.show()



if __name__ == '__main__':
    print("Main started")

    target_state= [
        0, 0.5,   # velocity and position on x
        0, 0.5,    # velocity and position on y
        0, 1,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0]     # angular velocity and position psi ]
    


    # THIS WORKS! At least for height, not sure if it the trajectory is wrong, or wrong display?
    # PLEASE DO NOT break THIS "simulate_quadrotor_linear_controller"
    #Run simulation
    simulate_quadrotor_linear_controller(target_state)
    #
    # next (from romagnolli: ) put bounds on control, like 0-7 ?  To see how controller reacts
    # also put bounds on torques 0.01 < x < - 0.01
    # this would be similar than saturated contraints in crazys simulation

    # this works too
    simulate_quadrotor_nonlinear_controller(target_state=target_state)


    #simulate_quadrotor_linear_integral_controller(control_input, g, Mq, Jx, Jy, Jz, 0.1)
    #simulate_figure_8(A=9, B=33, omega=.5, z0=12)



def figure_8_trajectory(t_steps, A, B, omega, z0):
    """The figure eight dynamics given by Dr. Romagnoli in the project writeup. The goal is to get the quadrotor
       to follow a figure eight trajectory.

    Args:
        t (float): The time steps of the simulation
        A (float): The amplitude along the x-axis
        B (float): The amplitude along the y-axis
        omega (float):  The angular velocity
        z0 (float): The constant altitude

    Returns:
        array: A vector of the trajectory (3 coordinates)
    """

    x_t = A * np.sin(omega * t_steps)

    y_t = B * np.sin(2 * omega * t_steps)

    z_t = z0

    return np.array([x_t, y_t, z_t])


def simulate_figure_8(A=2, B=1, omega=0.5, z0=1):
    """This function simulates the flight of a quadrotor in a figure-eight trajectory

    Args:
        A (float): amplitude along x-axis
        B (float): amplitude along y-axis
        omega (float): angular velocity
        z0 (float): constant altitude
    """

    #5000 timesteps inbetween 0 and 10 seconds
    time_interval_range = np.linspace(0, 20, 5000)

    #Obtaining the coordinates (trajectory) for each timestep in time_interval_range
    trajectory = np.array([figure_8_trajectory(t_steps = t,A=A, B=B, omega=omega, z0=z0) for t in time_interval_range])
    
    #Creating subplots for the figure-8 graphics
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))  
    fig.suptitle(f"Quadrotor Figure-8 Path\nHaving parameters Amplitude X: {A}, Amplitude Y: {B}, and Angular Velocity {omega}", 
                fontsize=12)

    #Plotting trajectory in X-Y plane (Figure-8 Path)
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="Path of Quadrotor in Figure Eight")
    axs[0, 0].set_xlabel('X meters')
    axs[0, 0].set_ylabel('Y meters')
    axs[0, 0].set_title("Quadrotor Figure-8 Path")


    #Plotting X-coordinate over time
    axs[0, 1].plot(time_interval_range, trajectory[:, 0], label="Change in X-coordinate over time")
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('X meters')
    axs[0, 1].set_title("X-Coordinates of the Quadrotor Over Time")


    #Plotting Y-coordinate over time
    axs[1, 0].plot(time_interval_range, trajectory[:, 1], label="Change in Y-coordinate over time")
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Y meters')
    axs[1, 0].set_title("Y-Coordinates of the Quadrotor Over Time")


    #Plotting Z-coordinate over time
    axs[1, 1].plot(time_interval_range, trajectory[:, 2], label="Change in Z-coordinate over time")
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Z meters')
    axs[1, 1].set_title("Z-Coordinates of the Quadrotor Over Time")
  
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

def simulate_quadrotor_linear_integral_controller(target_state, dt, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span = [0, 10]):
    """This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but uses the integral controller matrix Kc

    Args:
        target_state (dict): A dictionary of the input force, roll, pitch, and yaw
        dt (float): The time step for the integral controller
    """



    control_input = integral_controller(target_state, initial_state, np.zeros_like(initial_state), K, Kc, dt)

    #A is a 12x12 matrix used for linear simulation (given in slide 35, lecture 3)
    # Change to make it same as on slide.?
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

    #B is a 12x4 matrix used for linear simulation (given in slide 35)
    B = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1/m, 0, 0, 0],
        [0, 1/Jx, 0, 0],
        [0, 0, 1/Jy, 0],
        [0, 0, 0, 1/Jz],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    sol_linear = simulate_linear(control_input, time_span, initial_state, A, B)
    
    plt.figure()
    #plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Nonlinear z(t)')
    plt.plot(sol_linear.t, sol_linear.y[2], label='Linear z(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.show()

    x, y, z = sol_linear.y[0], sol_linear.y[1], sol_linear.y[2]
    roll, pitch, yaw = sol_linear.y[3], sol_linear.y[4], sol_linear.y[5]
    t = sol_linear.t  

    fig = plt.figure(figsize=(12, 8))

    #3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='3D Trajectory')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('Quadrotor 3D Trajectory')
    ax1.legend()

    #Position over Time (x, y, z)
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)')
    ax2.plot(t, y, label='y(t)')
    ax2.plot(t, z, label='z(t)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs. Time')
    ax2.legend()

    #Orientation over Time (roll, pitch, yaw)
    ax3 = fig.add_subplot(223)
    ax3.plot(t, roll, label='Roll (rad)')
    ax3.plot(t, pitch, label='Pitch (rad)')
    ax3.plot(t, yaw, label='Yaw (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (rad)')
    ax3.set_title('Orientation vs. Time')
    ax3.legend()

    #Height over Time (for hovering behavior)
    ax4 = fig.add_subplot(224)
    ax4.plot(t, z, label='Height (z)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Height vs. Time')
    ax4.legend()

    plt.tight_layout()
    plt.show()






# def simulate_nonlinear(t_span, initial_state, state, control_input):
#     """This function simulates the non-linear dynamics of the quadrotor by solving the 
#        the differential equations. We use SciPy solve_ivp to solve the equations which give us predictions
#        of the next state of the quadrotor given the control control_input and current state.
#        The solve_ivp module uses the Range-Kutta method of solving ODEs, which is 
#        a more advanced form of Euler's method.

#     Args:
#         t_span (array): The span of time to simulate
#         initial_state (array): The initial state of the quadrotor
#         state (array): _description_
#         control_input (array): A vector of input control forces

#     Returns:
#         Scipy : A solved differential equation object

#     SciPy's solve_ivp is a package to solve ordinary differntial equations of the form dt/dy = f(t,y)
#     it provides use with the trajectory of the quadrotor over time in regard to it's dynamics, the
#     input forces, and the desired/initial state. We solve for non-linear dynamics here.

#     """
#     #solve_ivp numerically solves differential equations through integration given an initial value
#     sol = solve_ivp(nonlinear_dynamics, t_span, initial_state, args=((state, control_input, g, m, Jx, Jy, Jz)), dense_output=True)
#     return sol