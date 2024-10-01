import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# All parameters are in seperate file and imported here.
from ControllerMatrices.quadrotor_parameters import Mq, L, g, Ms, R, Mprop, Mm, Jx, Jy, Jz, m, A, B, C, K, Kc



def nonlinear_dynamics( t, curstate, A, B, target_state):
    """
    Modeling the non-linear dynamics of the quadrotor. The non-linear dynamics tell you how
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
        state (array): A vector of the current state of the quadrotor
        control_input (array): A vector of the input forces into the quadrotor, size 1X4

    Returns:
        array: An array defining the change in the state for each dimension of the quadrotor
    """

    error = []

    # Iterate exactly 12 times (assuming target_state and curstate have at least 12 elements)
    for i in range(12):
        target = target_state[i]
        state = curstate[i]
        # Compute the difference between target and state
        difference = state - target
        # Append the result to the error list
        error.append(difference)


    control = -K @ (error)

    xDot = (A @ curstate) + B @ control
    return xDot


def linear_dynamics_integral(t, curstate_integral, Ac, Bc, target_state):
    Dotx = Ac @ curstate_integral + Bc @ target_state
    return Dotx


def simulate_quadrotor_linear_integral_controller(target_state, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span = [0, 10]):
    """
        This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but uses the integral controller matrix Kc.
       Initial state has 16 components for integral controller specifically.

    """


    Ac = np.zeros((4,4))
    Bc = np.eye(4)

    Acl = np.block(
        [
            [A - B @ K, B @ Kc],   # Block 1: A-BK and BKc should have the same row dimensions
            [-Bc @ C, Ac]          # Block 2: -BcC and Ac should have the same row dimensions
        ]
    )    
    Bcl = np.vstack([np.zeros((12, 4)), Bc])

    sol_linear = solve_ivp(linear_dynamics_integral, time_span, initial_state, args=(Acl, Bcl, target_state) , dense_output=True)

    # Obtain results from the solved differential equations
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]  # x, y, z positions
    roll, pitch, yaw = sol_linear.y[7], sol_linear.y[9], sol_linear.y[11]  # roll, pitch, yaw
    t = sol_linear.t  # Time

    display_plot(t, x, y, z)



def simulate_quadrotor_linear_controller(target_state, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span = [0, 10]):
    """This function simulates the quadrotor moving from a an initial state, taking off to a desired state,
       and hovering around this desired point using only the feedback matrix K of the feedforward controller.
       This function models the change in the quadrotor's movement using only linear dynamics.

    Args:
        initial_state (array): The initial state of the quadrotor,default 0
        time_span (array)[start, end]: The time span over which to model the flight of the quadrotor, default 0, 10
        target_state (array): Goal state of the drone. ( 12 x 1 array)
    """
    
    # Solve the linear dynamics using solve_ivp
    sol_linear = solve_ivp(linear_dynamics, time_span, initial_state, args=(A, B, target_state), dense_output=True)

    # Obtain results from the solved differential equations
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]  # x, y, z positions
    t = sol_linear.t 
    display_plot(t, x, y, z)

def display_plot(t, x, y, z):
    """
    This function takes time and position data and creates 3D and 2D plots
    to visualize the trajectory and movement of the quadrotor.
    This still does not work perfectly.

    Args:
        t (array): Time data.
        x (array): X position data.
        y (array): Y position data.
        z (array): Z position data.
    """

    # Check if all values in x and y are zero or very close to zero
    if np.allclose(x, 0) and np.allclose(y, 0):
        x = np.ones_like(t)  # Set all x values to 1
        y = np.ones_like(t)  # Set all y values to 1

    # Create a figure for the Position vs. Time plot and the 2D plane plots
    fig = plt.figure(figsize=(15, 12))

    # Subplot 1: 3D Trajectory Plot (Figure-8)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='Trajectory', color='b', linewidth=2)
    ax1.set_title('Quadrotor 3D Trajectory', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_zlabel('Z Position (m)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True)

    # Manually set limits for x, y axes to avoid labels like 1e-15
    if np.allclose(x, 0):
        ax1.set_xlim([0, 1])
    if np.allclose(y, 0):
        ax1.set_ylim([0, 1])

    # Subplot 2: Position over Time (x, y, z)
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)', color='r', linestyle='--', marker='s', markersize=4)
    ax2.plot(t, y, label='y(t)', color='g', linestyle='-.', marker='^', markersize=4)
    ax2.plot(t, z, label='z(t)', color='b', linestyle='-', marker='o', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('Position vs. Time (x, y, z)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True)

    # Subplot 3: XY Plane (Top-down view of x and y positions)
    ax3 = fig.add_subplot(223)
    ax3.plot(x, y, color='purple', linestyle='-', marker='o', markersize=4)
    ax3.set_xlabel('X Position (m)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.set_title('XY Plane (Top-down view)', fontsize=14, fontweight='bold')
    ax3.grid(True)

    # Subplot 4: XZ Plane (Side view of x and z positions)
    ax4 = fig.add_subplot(224)
    ax4.plot(x, z, color='orange', linestyle='-', marker='o', markersize=4)
    ax4.set_xlabel('X Position (m)', fontsize=12)
    ax4.set_ylabel('Z Position (m)', fontsize=12)
    ax4.set_title('XZ Plane (Side view)', fontsize=14, fontweight='bold')
    ax4.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plots
    plt.show()


def simulate_quadrotor_nonlinear_controller(target_state, initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span=[0,10]):
    """
    This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but using non-linear dynamics

    Args:
        target_state: Goal state where drone needs to go to
    """


    sol_nonlinear = solve_ivp(nonlinear_dynamics, time_span, initial_state, args=(A, B, target_state), dense_output=True)

    # Obtain results from the solved differential equations
    x, y, z = sol_nonlinear.y[1], sol_nonlinear.y[3], sol_nonlinear.y[5]  # x, y, z positions
    roll, pitch, yaw = sol_nonlinear.y[7], sol_nonlinear.y[9], sol_nonlinear.y[11]  # roll, pitch, yaw
    t = sol_nonlinear.t  # Time

    display_plot(t, x, y, z)


if __name__ == '__main__':
    print("Main started")

    target_state= [
        0, 0,   # velocity and position on x
        0, 0,    # velocity and position on y
        0, 100,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0]     # angular velocity and position psi ]
    

    target_state_integral = [
        0, 1,   # x, y
        1, 0]   # z, psy
    
    #Run simulation
    simulate_quadrotor_linear_controller(target_state)

    # next (from romagnolli: ) put bounds on control, like 0-7 ?  To see how controller reacts
    # also put bounds on torques 0.01 < x < - 0.01
    # this would be similar than saturated contraints in crazys simulation

    #simulate_quadrotor_nonlinear_controller(target_state=target_state)

    #simulate_quadrotor_linear_integral_controller(target_state=target_state_integral)

    #Have not tested, or verified this simulate_figure_8 funtion
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

