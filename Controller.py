import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def figure_eight(Mq, L, g, Ms, R, Mprop, Mm, ):
    pass


def nonlinear_dynamics(t, state, inputs):
    """Non-linear dynamics for the quadrotor

    Args:
        t (_type_): _description_
        state (array): A vector of the current state of the quadrotor
        inputs (dictionary): A dictionary of the input forces into the quadrotor

    Returns:
        _type_: _description_

    State variables are Linear position, Linear Velocities, Attitude, and Angular Velocities:
        Px, Py, Pz -> These are the linear position coordinates of the quadrotor
        u, v, q -> These are the linear velocities of the quadrotor
        phi, theta, psi -> These are the Attitude values of the quadrotor (tilt/position)
        p, q, r -> These are the angular velocities of the quadrotor

    Inputs into the system (quadrotor) are Force (throttle), Roll Torque, Pitch Torque, and Yaw Torque:
        F -> Force
        TauPhi -> Roll torque
        TauTheta -> Pitch torque
        TauPsi -> Yaw torque
    """
    Px, Py, Pz, u, v, w, phi, theta, psi, p, q, r = state
    
    
    F = inputs['Force']
    TauPhi = inputs['Roll']
    TauTheta = inputs['Pitch']
    TauPsi = inputs['Yaw']
    
    
def linear_dynamics(t, state, inputs):
    """Modeling the linear dynamics of the quadrotor

    Args:
        t (_type_): _description_
        state (array): A vector of the current state of the quadrotor
        inputs (array): A vector of the input forces into the quadrotor

    Returns:
        float: a floating point number indicating the change in x
    """
    Px, Py, Pz, u, v, w, phi, theta, psi, p, q, r = state
    
    #A is a 12x12 matrix of zeroes
    A = np.zeros((12, 12))  
    #B is a 12x4 matrix of zeroes
    B = np.zeros((12, 4))   
    
    #The change in X (xDot) is equal to Ax + Bu (@ is matrix-multiplication operator)
    #x in this equation is equal to the state and u is equal to the control inputs
    xDot = (A @ state) + B @ inputs

    return xDot


#The controller for the feedforward method
def feedforward_controller(targetState, state, K):
    """The feedforward controller. This controller only uses the feedback matrix K
        to adjust the inputs. The goal is the minimze the error between the current
        state and the goal state for the quadrotor

    Args:
        targetState (array): A vector representing the target (goal) state of the quadrotor
        state (array): A vector representing the current state of the quadrotor
        K (nd-array): A multi-dimensional array (matrix) of feedback

    Returns:
        array: A new vector to adjust inputs
    """

    #The error is the difference between the goal state and the current state
    error = state - targetState

    #Controlling by multiplying error by negative feedback matrix K
    control = -K @ (error)

    return control


def integral_controller(targetState, state, integral, K, Kc):
    """An integral controller 'remembers' errors and can adjust the quadrotor even
       in the presence of small errors.

    Args:
        targetState (array): A vector representing the target (goal) state of the quadrotor
        state (array): A vector representing the current state of the quadrotor
        integral (_type_): _description_
        K (nd-array): A multi-dimensional array (matrix) of feedback
        Kc (_type_): _description_

    Returns:
        _type_: _description_
    """
    return -K @ (state - targetState) - Kc @ integral

# Figure-8 trajectory
def figure_8_trajectory(t, A, B, omega, z0):
    x = A * np.sin(omega * t)
    y = B * np.sin(2 * omega * t)
    z = z0
    return np.array([x, y, z])

# Simulation
def simulate_nonlinear(u_func, t_span, y0):
    sol = solve_ivp(nonlinear_dynamics, t_span, y0, args=(u_func,), dense_output=True)
    return sol

def simulate_linear(u_func, t_span, y0):
    sol = solve_ivp(linear_dynamics, t_span, y0, args=(u_func,), dense_output=True)
    return sol

# Main simulation function
def simulate_quadrotor():
    # Initial state (at rest)
    y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Time span
    t_span = [0, 10]
    
    # Control input (example feedforward control)
    u_func = lambda t: feedforward_controller([0, 0, 1, 0], y0, K)
    
    # Simulate nonlinear dynamics
    sol_nonlinear = simulate_nonlinear(u_func, t_span, y0)
    
    # Simulate linear dynamics
    sol_linear = simulate_linear(u_func, t_span, y0)
    
    # Plot results
    plt.figure()
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Nonlinear z(t)')
    plt.plot(sol_linear.t, sol_linear.y[2], label='Linear z(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.show()

# Figure-8 movement simulation
def simulate_figure_8():
    A = 2
    B = 1
    omega = 0.5
    z0 = 1
    
    t_vals = np.linspace(0, 10, 500)
    trajectory = np.array([figure_8_trajectory(t, A, B, omega, z0) for t in t_vals])
    
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Figure-8 Path")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title("Quadrotor Figure-8 Path")
    plt.legend()
    plt.show()


if __name__ == '__main__':
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

    # Run simulation
    simulate_quadrotor()
    simulate_figure_8()
