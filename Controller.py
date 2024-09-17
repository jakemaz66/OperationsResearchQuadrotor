import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from mpl_toolkits.mplot3d import Axes3D


def nonlinear_dynamics(state, inputs, g, m, Jx, Jy, Jz):
    """Modeling the non-linear dynamics of the quadrotor. The non-linear dynamics tell you how
       the state of the quadrotor changes with regards to three things
       1. The current state of the quadrotor
       2. The input controls into the quadrotor
       3. The current forces of physics

    Args:
        state (array): A vector of the current state of the quadrotor
        inputs (dictionary): A dictionary defining the input forces, will get transformed into vector
        g (float): The constant for gravity
        m (float): The mass of the quadrotor
        Jx (float): The moment of inertia in the x direction
        Jy (float): The moment of inertia in the y direction
        Jz (float): The moment of inertia in the z direction

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor

    State variables are Linear position, Linear Velocities, Attitude, and Angular Velocities:
        Px, Py, Pz -> These are the linear position coordinates of the quadrotor
        u, v, w -> These are the linear velocities of the quadrotor
        phi, theta, psi -> These are the Attitude values of the quadrotor (tilt/position), they are measured in radians
        p, q, r -> These are the angular velocities of the quadrotor, they are measured in radians

    Inputs into the system (quadrotor) are Force (throttle), Roll Torque, Pitch Torque, and Yaw Torque:
        F -> Force
        TauPhi -> Roll torque
        TauTheta -> Pitch torque
        TauPsi -> Yaw torque
    This produces an input array of type [F, TauPhim TauTheta, TauPsi]
    """

    #Unpacking the variables of the state
    Px, Py, Pz, u, v, w, phi, theta, psi, p, q, r = state
    
    #Obtaining the inputs into the quadrotor
    F = inputs['Force']
    TauPhi = inputs['Roll']
    TauTheta = inputs['Pitch']
    TauPsi = inputs['Yaw']

    #Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
    uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (g * np.cos(theta) * np.cos(phi)) + (-F / m)

    #Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
    pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
    qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
    rDot = (((Jx - Jy) / Jz) * q * q) + ((1 / Jz) * TauPsi)

    #Calculating the non-linear change dynamics for angular positions (equation 2 slide 31)
    phiDot = (1 * p) + (np.sin(phi) * np.tan(theta) * q) + (np.cos(phi) * np.tan(theta) * r)
    thetaDot = (0 * p) + (np.cos(theta) * q) + (-np.sin(phi) * r)
    psiDot = (0 * p) + (np.sin(phi) / np.cos(theta) * q) + (np.cos(phi) / np.cos(theta) * r)

    #Calculating the non-linear change dynamics for linear positions (equation 1 slide 31)
    pDotx = (u * np.cos(theta) * np.cos(psi)) + (v * (np.sin(phi) * np.sin(theta) * np.cos(psi)) - (np.cos(phi) * np.sin(psi))) + \
    (w * (np.cos(phi) * np.sin(theta) * np.cos(psi)) + (np.sin(phi) * np.sin(psi)))

    pDoty = (u * np.cos(theta) * np.cos(psi)) + (v * (np.sin(phi) * np.sin(theta) * np.cos(psi)) + (np.cos(phi) * np.sin(psi))) + \
    (w * (np.cos(phi) * np.sin(theta) * np.cos(psi)) - (np.sin(phi) * np.sin(psi)))

    pDotz = (u * np.sin(theta)) + (v * -np.sin(psi) * np.cos(theta)) + (w * -np.cos(psi) * np.cos(theta))
    

    #Returning the vector that indicates the change for each part of the state
    return [pDotx, pDoty, pDotz, uDot, vDot, wDot, phiDot, thetaDot, psiDot, pDot, qDot, rDot]
    
    
def linear_dynamics(t, state, A, B, inputs):
    """Modeling the linear dynamics of the quadrotor. The linear dynamics tell you how
       the state of the quadrotor changes with regards to two things
       1. The current state of the quadrotor
       2. The input controls into the quadrotor

    Args:
        t (list): The time interval
        state (array): A vector of the current state of the quadrotor
        inputs (array): A vector of the input forces into the quadrotor, size 1X4
        A (nd-array): A given matrix for state linear dynamics (given in slides)
        B (nd-array): A given matrix for input linear dynamics (given in slides)

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor
    """
    Px, Py, Pz, u, v, w, phi, theta, psi, p, q, r = state
    
    #The change in state (xDot) is equal to Ax + Bu (@ is matrix-multiplication operator)
    #x in this equation is equal to the current state and u is equal to the control inputs
    xDot = (A @ state) + B @ inputs

    return xDot


#The controller for the feedforward method
def feedforward_controller(targetState, state, K):
    """The feedforward controller. This controller uses only the feedback matrix K
        to adjust the inputs of the quadrotor. The goal is the minimize the error 
        between the current state and the targetState state for the quadrotor. It
        uses the formula −K(x−xeq) to produce a new control vector

    Args:
        targetState (array): A vector representing the target (goal) state of the quadrotor
        state (array): A vector representing the current state of the quadrotor
        K (nd-array): A multi-dimensional array (matrix) of feedback. K is determined through
                      LQR or other optimization methods (given to us by Dr. Romagnoli)

    Returns:
        array: A new vector of 4 values to adjust the inputs
    """

    #The error is the difference between the goal state and the current state
    error = [state - target for target, state in zip(targetState, state)]

    #Controlling by multiplying error by negative feedback matrix K "−K(x−xeq)"
    control = -K @ (error)

    return control


def integral_controller(targetState, state, integral, K, Kc, dt):
    """An integral controller 'remembers' errors and can adjust the quadrotor even
       in the presence of small errors. It uses the K and Kc matrix to 
       return a new input vector of forces to control the quadrotor

    Args:
        targetState (array): A vector representing the target (goal) state of the quadrotor
        state (array): A vector representing the current state of the quadrotor
        integral (array): _description_
        K (nd-array): A multi-dimensional array (matrix) of feedback
        Kc (nd-array): A multi-dimensional array (matrix) of feedback
        dt (float): A number to indicate the timestep 

    Returns:
        array: A new vector of 4 values to adjust the inputs
    """

    # − [K −Kc] xa

    error = [state - target for target, state in zip(targetState, state)]
    error = np.array(error, dtype=np.float32)
    dt = np.float32(dt)
    integral = np.array(integral)

    integral = integral + np.dot(error, dt)

    #Error here because integral is (12x1) and Kc is (4x4)
    integral_control  = K @ error + Kc @ integral

    return integral_control


def simulate_nonlinear(t_span, initial_state, state, inputs, g, m, Jx, Jy, Jz):
    """This function simulates the non-linear dynamics of the quadrotor. 

    Args:
        t_span (array): The span of time to simulate
        initial_state (array): The initial state of the quadrotor
        state (array): _description_
        inputs (array): A vector of input control forces
        g (float): The constant of acceleration due to gravity
        m (float): The mass of the quadrotor
        Jx (float): The moment of inertia in the x direction
        Jy (float): The moment of inertia in the y direction
        Jz (float): The moment of inertia in the z direction

    Returns:
        Scipy : A solved differential equation object

    SciPy's solve_ivp is a package to solve ordinary differntial equations of the form dt/dy = f(t,y)
    it provides use with the trajectory of the quadrotor over time in regard to it's dynamics, the
    input forces, and the desired/initial state. We solve for non-linear dynamics here.

    """
    #solve_ivp numerically solves differential equations through integration given an initial value
    sol = solve_ivp(nonlinear_dynamics, t_span, initial_state, args=((state, inputs, g, m, Jx, Jy, Jz)), dense_output=True)
    return sol


def simulate_linear(inputs, t_span, initial_state, A, B):
    """This function simulates the linear dynamics of the quadrotor by solving the 
       the differential equations. We use SciPy solve_ivp to solve the equations.

    Args:
        inputs (array): A vector of input control forces
        t_span (array): The span of time to simulate
        initial_state (array): The initial state of the quadrotor
        A (nd-array): A control array given in slide 35
        B (nd-array): A control array given in slide 35

    Returns:
        Scipy : A solved differential equation object

    SciPy's solve_ivp is a package to solve ordinary differntial equations of the form dt/dy = f(t,y)
    it provides use with the trajectory of the quadrotor over time in regard to it's dynamics, the
    input forces, and the desired/initial state. We solve for linear dynamics here.

    """

    sol = solve_ivp(linear_dynamics, t_span, initial_state, args=(A, B, inputs), dense_output=True)
    return sol


def simulate_quadrotor_linear_controller(inputs, g, m, Jx, Jy, Jz):
    """This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point using only the feedback matrix K

    Args:
        inputs (dict): A dictionary of the input force, roll, pitch, and yaw
        g (float): The constant for gravity
        m (float): The mass of the quadrotor
        Jx (float): The moment of inertia in the x direction
        Jy (float): The moment of inertia in the y direction
        Jz (float): The moment of inertia in the z direction
    """
    #Initial state of the quadrtor at rest, has no velocity or positive position coordinates
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #The time span for the quadrotor
    time_span = [0, 10]

    #The control matrix K
    ControllerMatrix = scipy.io.loadmat("ControllerMatrices\K.mat")
    K = ControllerMatrix['K']

    
    #The inputs into the quadrotor
    #These are the coordinates we want the quadrotor to go to and hover at (the desired state)
    #The angular velocities and attitude are in radians so put accordingly. 
    
    desired_coordinates = [0, 0, 8,     #Position: x, y, z
                 0, 0, 0,     #Linear velocity: x, y, z
                 np.pi, 0, 0, #Orientation: roll = 180 deg, pitch = 0, yaw = 0
                 0, 0, 0]

    #Receive optimal inputs into the quadrotor through the controller
    inputs = feedforward_controller(desired_coordinates, initial_state, K)
    

    #A is a 12x12 matrix for linear simulation (given in slide 35)
    A = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0, -g, 0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, g,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0]
    ]) 

    #B is a 12x4 matrix for linear simulation (given in slide 35)
    B = np.array([
    [0, 0, 0, 0], [0, 0, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 0],
    [0, 0, 0, 0], [1/m, 0, 0, 0],
    [0, 1/Jx, 0, 0], [0, 0, 1/Jy, 0],
    [0, 0, 0, 1/Jz], [0, 0, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 0]
    ])
    
    #Simulating the flight of the quadrotor
    sol_linear = simulate_linear(inputs, time_span, initial_state, A, B)
    

    #Obtaining results from the solved differential equations
    x, y, z = sol_linear.y[0], sol_linear.y[1], sol_linear.y[2]
    roll, pitch, yaw = sol_linear.y[3], sol_linear.y[4], sol_linear.y[5]
    t = sol_linear.t  

    #Creating plots and figures from the simulation
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


def simulate_quadrotor_linear_integral_controller(inputs, g, m, Jx, Jy, Jz, dt):
    """This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but uses the integral controller matrix Kc

    Args:
        inputs (dict): A dictionary of the input force, roll, pitch, and yaw
        g (float): The constant for gravity
        m (float): The mass of the quadrotor
        Jx (float): The moment of inertia in the x direction
        Jy (float): The moment of inertia in the y direction
        Jz (float): The moment of inertia in the z direction
        dt (float): The time step for the integral controller
    """
    #Initial state of the quadrtor at rest, has no velocity or positive position coordinates
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #The time span for the quadrotor
    time_span = [0, 10]

    #The control matrix K
    ControllerMatrix = scipy.io.loadmat("ControllerMatrices\K.mat")
    K = ControllerMatrix['K']

    #The integral control matrix Kc
    ControllerMatrixC = scipy.io.loadmat("ControllerMatrices\Kc.mat")
    Kc = ControllerMatrixC['Kc']
    
    #The inputs into the quadrotor
    #These are the coordinates we want the quadrotor to go to and hover at (the desired state)
    #The angular velocities and attitude are in radians so put accordingly. 
    desired_coordinates = [0, 0, 8,     # Position: x, y, z
                 0, 0, 0,     # Linear velocity: x, y, z
                 np.pi, 0, 0, # Orientation: roll = 180 deg, pitch = 0, yaw = 0
                 0, 0, 0]

    inputs = integral_controller(desired_coordinates, initial_state, np.zeros_like(initial_state), K, Kc, dt)

    #A is a 12x12 matrix for linear simulation
    A = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0, -g, 0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, g,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0]
    ]) 

    #B is a 12x4 matrix for linear simulation
    B = np.array([
    [0, 0, 0, 0], [0, 0, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 0],
    [0, 0, 0, 0], [1/m, 0, 0, 0],
    [0, 1/Jx, 0, 0], [0, 0, 1/Jy, 0],
    [0, 0, 0, 1/Jz], [0, 0, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 0]
    ])

    sol_linear = simulate_linear(inputs, time_span, initial_state, A, B)
    
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


def simulate_quadrotor_nonlinear_controller(inputs, g, m, Jx, Jy, Jz):
    """This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but using non-linear dynamics

    Args:
        inputs (dict): A dictionary of the input force, roll, pitch, and yaw
        g (float): The constant for gravity
        m (float): The mass of the quadrotor
        Jx (float): The moment of inertia in the x direction
        Jy (float): The moment of inertia in the y direction
        Jz (float): The moment of inertia in the z direction
    """
    #Initial state of the quadrtor at rest, has no velocity or positive position coordinates
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #The time span for the quadrotor
    time_span = [0, 10]

    #The control matrix K
    ControllerMatrix = scipy.io.loadmat("ControllerMatrices\K.mat")
    K = ControllerMatrix['K']

    #The inputs into the quadrotor
    #These are the coordinates we want the quadrotor to go to and hover at (the desired state)
    #The angular velocities and attitude are in radians so put accordingly. 
    desired_coordinates = [0, 0, 8,     # Position: x, y, z
                 0, 0, 0,     # Linear velocity: x, y, z
                 np.pi, 0, 0, # Orientation: roll = 180 deg, pitch = 0, yaw = 0
                 0, 0, 0]

    inputs = feedforward_controller(desired_coordinates, initial_state, K)
    
    #Simulating dynamics
    sol_nonlinear = simulate_nonlinear(time_span, initial_state, desired_coordinates, inputs, g, m, Jx, Jy, Jz)
    
    plt.figure()
    #plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Nonlinear z(t)')
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label='Linear z(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.show()

    x, y, z = sol_nonlinear.y[0], sol_nonlinear.y[1], sol_nonlinear.y[2]
    roll, pitch, yaw = sol_nonlinear.y[3], sol_nonlinear.y[4], sol_nonlinear.y[5]
    t = sol_nonlinear.t  

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


def figure_8_trajectory(t, A, B, omega, z0):
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

    x_t = A * np.sin(omega * t)

    y_t = B * np.sin(2 * omega * t)

    z_t = z0

    return np.array([x_t, y_t, z_t])


def simulate_figure_8(A=2, B=1, omega=0.5, z0=1):
    """This function simulates the flight of a quadrotor in a figure-eight trajectory

    Args:
        A (float): amplitude along x-axis
        B (_type_): amplitude along y-axis
        omega (float): angular velocity
        z0 (float): constant altitude
    """

    #5000 timesteps inbetween 0 and 10 seconds
    time_interval_range = np.linspace(0, 20, 5000)

    #Obtaining the coordinates (trajectory) for each timestep in time_interval_range
    trajectory = np.array([figure_8_trajectory(t, A, B, omega, z0) for t in time_interval_range])
    
    #Plotting the result
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Path of Quadrotor in Figure Eight")
    plt.xlabel('X meters')
    plt.ylabel('Y meters')
    plt.suptitle("Quadrotor Figure-8 Path")
    plt.show()

    plt.plot(trajectory[:, 0], time_interval_range, label="Change in x-coordinate over time")
    plt.xlabel('Time')
    plt.ylabel('X meters')
    plt.suptitle("The X-Coordinates of the quadrotor over time")
    plt.show()

    plt.plot(trajectory[:, 1], time_interval_range, label="Change in y-coordinate over time")
    plt.xlabel('Time')
    plt.ylabel('Y meters')
    plt.suptitle("The Y-Coordinates of the quadrotor over time")
    plt.show()

    plt.plot(trajectory[:, 2], time_interval_range, label="Change in z-coordinate over time")
    plt.xlabel('Time')
    plt.ylabel('Z meters')
    plt.suptitle("The Z-Coordinate of the quadrotor over time")
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

    inputs = {}
    inputs['Force'] = 5
    inputs['Roll'] = 1
    inputs['Pitch'] = 1
    inputs['Yaw'] = 1

    #Run simulation
    #simulate_quadrotor_linear_controller(inputs, g, Mq, Jx, Jy, Jz)
    simulate_quadrotor_linear_integral_controller(inputs, g, Mq, Jx, Jy, Jz, 0.1)
    #simulate_figure_8()

