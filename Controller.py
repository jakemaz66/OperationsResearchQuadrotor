import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# All parameters are in seperate file and imported here.
from ControllerMatrices.quadrotor_parameters import Mq, L, g, Ms, R, Mprop, Mm, Jx, Jy, Jz, m, A, B, C, K, Kc
import os

force_before_bound = []
force_after_bound = []

tauPhi_before_bound = []
tauTheta_before_bound = []
tauPsi_before_bound = []

tauPhi_after_bound = []
tauTheta_after_bound = []
tauPsi_after_bound = []


def nonlinear_dynamics(t, curstate, target_state, bounds):
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

    u, Px, v, Py, w, Pz, phi, p, theta, q, psi, r = curstate
    force_bound, torque_bound = bounds

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

    # Controlling by multiplying error by negative feedback matrix K "−K(x−xeq)"
    control = -K @ (error)

    F = control[0] + Mq * g  # Force -> Throttle
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque
    TauPsi = control[3]     # Yaw Torque

    force_before_bound.append(F)
    tauPhi_before_bound.append(TauPhi)
    tauTheta_before_bound.append(TauTheta)
    tauPsi_before_bound.append(TauPsi)

    if force_bound:
        F = np.clip(F, None, force_bound)

    if torque_bound:
        TauPhi = np.clip(TauPhi, -torque_bound, torque_bound)
        TauTheta = np.clip(TauTheta, -torque_bound, torque_bound)
        TauPsi = np.clip(TauPsi, -torque_bound, torque_bound)

    force_after_bound.append(F)
    tauPhi_after_bound.append(TauPhi)
    tauTheta_after_bound.append(TauTheta)
    tauPsi_after_bound.append(TauPsi)

    # Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
    uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (g * np.cos(theta) * np.cos(phi)) + (-F / Mq)

    # Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
    pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
    qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
    rDot = (((Jx - Jy) / Jz) * p * q) + ((1 / Jz) * TauPsi)

    PxDot = u
    PyDot = v
    PzDot = w

    # u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r
    return [uDot, PxDot, vDot, PyDot, wDot, PzDot, pDot, p, qDot, q, rDot, r]


def linear_dynamics(t, curstate, A, B, target_state, bounds):
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
    force_bound, torque_bound = bounds

    # Iterate exactly 12 times (assuming target_state and curstate have at least 12 elements)
    for i in range(12):
        target = target_state[i]
        state = curstate[i]
        # Compute the difference between target and state
        difference = state - target
        # Append the result to the error list
        error.append(difference)

    control = -K @ (error)

    force_before_bound.append(control[0])
    tauPhi_before_bound.append(control[1])
    tauTheta_before_bound.append(control[2])
    tauPsi_before_bound.append(control[3])

    if force_bound:
        control[0] = np.clip(control[0], None, force_bound)

    if torque_bound:
        control[1] = np.clip(control[1], -torque_bound, torque_bound)
        control[2] = np.clip(control[2], -torque_bound, torque_bound)
        control[3] = np.clip(control[3], -torque_bound, torque_bound)

    force_after_bound.append(control[0])
    tauPhi_after_bound.append(control[1])
    tauTheta_after_bound.append(control[2])
    tauPsi_after_bound.append(control[3])

    xDot = (A @ curstate) + B @ control
    return xDot


def linear_dynamics_integral(t, curstate_integral, Ac, Bc, target_state):

    target_state = np.array(target_state)
    curstate_integral = np.array(curstate_integral)
    
    #Getting 4x1 Integral u
    uc = (target_state[[1, 3, 5, 11]] - curstate_integral[[1, 3, 5, 11]])

    Dotx = Ac @ curstate_integral + Bc @ uc
    return Dotx 


def nonlinear_dynamics_integral(t, curstate_integral, Ac, Bc, target_state, bounds):
    

    u, Px, v, Py, w, Pz, phi, p, theta, q, psi, r, i1, i2, i3, i4 = curstate_integral
    force_bound, torque_bound = bounds

    target_state = np.array(target_state)
    curstate_integral = np.array(curstate_integral)

    error = curstate_integral[:12]  - target_state[:12] 
    integral_error = curstate_integral[12:] - target_state[12:] 

    #Implementing integral control
    control = -K @ error - Kc @ integral_error

    F = control[0] + Mq * g  # Force -> Throttle
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque
    TauPsi = control[3]     # Yaw Torque

    force_before_bound.append(F)
    tauPhi_before_bound.append(TauPhi)
    tauTheta_before_bound.append(TauTheta)
    tauPsi_before_bound.append(TauPsi)

    # if force_bound:
    #     F = np.clip(F, None, force_bound)

    # if torque_bound:
    #     TauPhi = np.clip(TauPhi, -torque_bound, torque_bound)
    #     TauTheta = np.clip(TauTheta, -torque_bound, torque_bound)
    #     TauPsi = np.clip(TauPsi, -torque_bound, torque_bound)

    force_after_bound.append(F)
    tauPhi_after_bound.append(TauPhi)
    tauTheta_after_bound.append(TauTheta)
    tauPsi_after_bound.append(TauPsi)

    # Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
    uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (g * np.cos(theta) * np.cos(phi)) + (-F / m)

    # Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
    pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
    qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
    rDot = (((Jx - Jy) / Jz) * p * q) + ((1 / Jz) * TauPsi)

    PxDot = u
    PyDot = v
    PzDot = w

    # u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r
    state_change = [uDot, PxDot, vDot, PyDot, wDot, PzDot, pDot, p, qDot, q, rDot, r, i1, i2, i3, i4]

    state = []
    for el in state_change:
        el = float(el)
        state.append(el)

    return state


def simulate_linear_integral_with_euler(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                        total_time=10, time_step=0.00001):
    """This method simulates flight using the Euler method

    Args:   
        target_state (array): The target state of quadrotor
        initial_state (list, optional): The initial state of quadrotor. Defaults to [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
    """

    #Constructing time steps
    time_range = np.arange(0, total_time + time_step, time_step)
    total_time_steps = len(time_range)

    x0 = np.zeros(16,)

    #12 is the number of state elements, total_time_steps is the state at each time step
    x_tot = np.zeros((16, total_time_steps))

    #The first time step is equal to the initial state of the quadrotor
    x_tot[:, 0] = initial_state

    #Defining the blocks for integral control
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block(
        [
            # Block 1: A-BK and BKc should have the same row dimensions
            [A - B @ K, B @ Kc],
            # Block 2: -BcC and Ac should have the same row dimensions
            [-Bc @ C, Ac]
        ]
    )
    Bcl = np.vstack([np.zeros((12, 4)), Bc])

    #At each time-step, update the position array x_tot with the linear_dynamics
    for j in range (1, total_time_steps):
        x0 = linear_dynamics_integral(j, x0, Acl, Bcl, target_state) * time_step + x0
        x_tot[:, j] = x0 * 2

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Simulating with Euler and Linear Integral Control")
    
    # Change in X over time
    axs[0, 0].plot(time_range, x_tot[1, :], label='X Change')
    axs[0, 0].set_title('Change in X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('X Position')

    # Change in Y over time
    axs[0, 1].plot(time_range, x_tot[3, :], label='Y Change', color='orange')
    axs[0, 1].set_title('Change in Y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Y Position')

    # Change in Z over time
    axs[1, 0].plot(time_range, x_tot[5, :], label='Z Change', color='green')
    axs[1, 0].set_title('Change in Z')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Z Position')

    # 3D Trajectory Plot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(x_tot[1, :], x_tot[3, :], x_tot[5, :], label='3D Trajectory', color='red')
    ax3d.set_title('3D Trajectory')
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Z Position')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def simulate_nonlinear_integral_with_euler(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                        total_time=10, time_step=0.0001, bounds=(0,0)):
    """This method simulates flight using the Euler method

    Args:   
        target_state (array): The target state of quadrotor
        initial_state (list, optional): The initial state of quadrotor. Defaults to [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
    """

    #Constructing time steps
    time_range = np.arange(0, total_time + time_step, time_step)
    total_time_steps = len(time_range)

    x0 = np.zeros(16,)

    #12 is the number of state elements, total_time_steps is the state at each time step
    x_tot = np.zeros((16, total_time_steps))

    #The first time step is equal to the initial state of the quadrotor
    x_tot[:, 0] = initial_state
    Acl = None
    Bcl = None

    #At each time-step, update the position array x_tot with the linear_dynamics
    for j in range (1, total_time_steps):
        x0 = np.array(nonlinear_dynamics_integral(j, x0, Acl, Bcl, target_state, bounds)) * time_step + x0
        x_tot[:, j] = x0

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Simulating with Euler and Non-linear Integral Control")
    
    # Change in X over time
    axs[0, 0].plot(time_range, x_tot[1, :], label='X Change')
    axs[0, 0].set_title('Change in X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('X Position')

    # Change in Y over time
    axs[0, 1].plot(time_range, x_tot[3, :], label='Y Change', color='orange')
    axs[0, 1].set_title('Change in Y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Y Position')

    # Change in Z over time
    axs[1, 0].plot(time_range, x_tot[5, :], label='Z Change', color='green')
    axs[1, 0].set_title('Change in Z')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Z Position')

    # 3D Trajectory Plot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(x_tot[1, :], x_tot[3, :], x_tot[5, :], label='3D Trajectory', color='red')
    ax3d.set_title('3D Trajectory')
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Z Position')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def simulate_quadrotor_linear_integral_controller(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span=[0, 10]):
    """
        This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but uses the integral controller matrix Kc.
       Initial state has 16 components for integral controller specifically.

    """

    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block(
        [
            # Block 1: A-BK and BKc should have the same row dimensions
            [A - B @ K, B @ Kc],
            # Block 2: -BcC and Ac should have the same row dimensions
            [-Bc @ C, Ac]
        ]
    )
    Bcl = np.vstack([np.zeros((12, 4)), Bc])

    sol_linear = solve_ivp(linear_dynamics_integral, time_span, initial_state, args=(
        Acl, Bcl, target_state), dense_output=True)

    # Obtain results from the solved differential equations
    # x, y, z positions
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]
    # roll, pitch, yaw
    roll, pitch, yaw = sol_linear.y[7], sol_linear.y[9], sol_linear.y[11]
    t = sol_linear.t  # Time

    display_plot2(t, x, y, z)

    bound_time_steps = np.linspace(
        time_span[0], time_span[1], len(force_before_bound))
    plot_force_comparison(bound_time_steps)


def simulate_quadrotor_linear_controller(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span=[0, 10], bounds=(0, 0)):
    """This function simulates the quadrotor moving from a an initial state, taking off to a desired state,
       and hovering around this desired point using only the feedback matrix K of the feedforward controller.
       This function models the change in the quadrotor's movement using only linear dynamics.

    Args:
        initial_state (array): The initial state of the quadrotor,default 0
        time_span (array)[start, end]: The time span over which to model the flight of the quadrotor, default 0, 10
        target_state (array): Goal state of the drone. ( 12 x 1 array)
    """

    # Solve the linear dynamics using solve_ivp
    sol_linear = solve_ivp(linear_dynamics, time_span, initial_state, args=(
        A, B, target_state, bounds), dense_output=True)

    # Obtain results from the solved differential equations
    # x, y, z positions
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]
    t = sol_linear.t
    display_plot2(t, x, y, z)

    bound_time_steps = np.linspace(
        time_span[0], time_span[1], len(force_before_bound))
    plot_force_comparison(bound_time_steps)


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
    ax2.plot(t, x, label='x(t)', color='r',
             linestyle='--', marker='s', markersize=4)
    ax2.plot(t, y, label='y(t)', color='g',
             linestyle='-.', marker='^', markersize=4)
    ax2.plot(t, z, label='z(t)', color='b',
             linestyle='-', marker='o', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('Position vs. Time (x, y, z)',
                  fontsize=14, fontweight='bold')
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


def display_plot2(t, x, y, z):
    """
    This function takes time and position data and creates 3D and 2D plots
    to visualize the trajectory and movement of the quadrotor.

    Args:
        t (array): Time data.
        x (array): X position data.
        y (array): Y position data.
        z (array): Z position data.
    """

    epsilon = 1e-12  # Set a threshold for noise (adjust as necessary)
    x[np.abs(x) < epsilon] = 0
    y[np.abs(y) < epsilon] = 0

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

    # Subplot 2: X position over Time
    ax2 = fig.add_subplot(222)
    ax2.plot(t, x, label='x(t)', color='r',
             linestyle='-', marker='s', markersize=4)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('X Position (m)', fontsize=12)
    ax2.set_title('X Position vs. Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True)

    # Subplot 3: Y position over Time
    ax3 = fig.add_subplot(223)
    ax3.plot(t, y, label='y(t)', color='g',
             linestyle='-', marker='^', markersize=4)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Y Position (m)', fontsize=12)
    ax3.set_title('Y Position vs. Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True)

    # Subplot 4: Z position over Time
    ax4 = fig.add_subplot(224)
    ax4.plot(t, z, label='z(t)', color='b',
             linestyle='-', marker='o', markersize=4)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Z Position (m)', fontsize=12)
    ax4.set_title('Z Position vs. Time', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_force_comparison(time_values):
    """
    Plots the force values and torques before and after clipping over time.
    """
    plt.figure(figsize=(15, 10))  # Increase figure size for more subplots

    # Plot nonlinear dynamics forces (before and after clipping)
    # plt.subplot(4, 2, 1)
    plt.plot(time_values, force_before_bound,
             label='Force before clipping', color='r', linestyle='--')
    plt.plot(time_values, force_after_bound,
             label='Force after clipping', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Force Before and After Clipping')
    plt.legend()
    plt.grid(True)

    # Plot tauPhi before and after clipping
    # plt.subplot(4, 2, 2)
    # plt.scatter(time_values, tauPhi_before_bound,
    #             label='TauPhi before clipping', color='r', s=0.3)
    # plt.scatter(time_values, tauPhi_after_bound,
    #             label='TauPhi after clipping', color='b', s=0.3)
    # plt.xlabel('Time (s)')
    # plt.ylabel('TauPhi (N·m)')
    # plt.title('TauPhi Before and After Clipping')
    # plt.legend()
    # plt.grid(True)

    # Plot tauTheta before and after clipping
    # plt.subplot(4, 2, 3)
    # plt.scatter(time_values, tauTheta_before_bound,
    #             label='TauTheta before clipping', color='r', s=0.3)
    # plt.scatter(time_values, tauTheta_after_bound,
    #             label='TauTheta after clipping', color='b', s=0.3)
    # plt.xlabel('Time (s)')
    # plt.ylabel('TauTheta (N·m)')
    # plt.title('TauTheta Before and After Clipping')
    # plt.legend()
    # plt.grid(True)

    # Plot tauPsi before and after clipping
    # plt.subplot(4, 2, 4)
    # plt.scatter(time_values, tauPsi_before_bound,
    #             label='TauPsi before clipping', color='r', s=0.3)
    # plt.scatter(time_values, tauPsi_after_bound,
    #             label='TauPsi after clipping', color='b', s=0.3)
    # plt.xlabel('Time (s)')
    # plt.ylabel('TauPsi (N·m)')
    # plt.title('TauPsi Before and After Clipping')
    # plt.legend()
    # plt.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def simulate_quadrotor_nonlinear_controller(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], time_span=[0, 20], bounds=(0, 0)):
    """
    This function simulates the quadrotor moving from a an initial point, taking off to a desired point,
       and hovering around this desired point, but using non-linear dynamics

    Args:
        target_state: Goal state where drone needs to go to
    """

    sol_nonlinear = solve_ivp(nonlinear_dynamics, time_span, initial_state, args=(
        target_state, bounds), dense_output=True)

    # x, y, z positions
    x, y, z = sol_nonlinear.y[1], sol_nonlinear.y[3], sol_nonlinear.y[5]
    # roll, pitch, yaw
    roll, pitch, yaw = sol_nonlinear.y[7], sol_nonlinear.y[9], sol_nonlinear.y[11]
    t = sol_nonlinear.t  # Time

    display_plot2(t, x, y, z)

    bound_time_steps = np.linspace(
        time_span[0], time_span[1], len(force_before_bound))
    plot_force_comparison(bound_time_steps)


def clear_bound_values():
    """
    Clears the values of the force before and after clipping.
    """
    force_before_bound.clear()
    force_after_bound.clear()
    tauPhi_before_bound.clear()
    tauTheta_before_bound.clear()
    tauPsi_before_bound.clear()
    tauPhi_after_bound.clear()
    tauTheta_after_bound.clear()
    tauPsi_after_bound.clear()


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


def simulate_figure_8(At=2, Bt=1, omega=0.5, z0=1):
    """This function simulates the flight of a quadrotor in a figure-eight trajectory

    Args:
        A (float): amplitude along x-axis
        B (float): amplitude along y-axis
        omega (float): angular velocity
        z0 (float): constant altitude
    """

    # 5000 timesteps inbetween 0 and 10 seconds
    time_interval_range = np.linspace(0, 30, 50000)
    time_step = 20/5000

    # Obtaining the coordinates (trajectory) for each timestep in time_interval_range
    trajectory = np.array([figure_8_trajectory(
        t_steps=t, A=At, B=Bt, omega=omega, z0=z0) for t in time_interval_range])
    
    #Constructing time steps
    total_time_steps = len(time_interval_range)

    x0 = np.zeros(12,)

    #12 is the number of state elements, total_time_steps is the state at each time step
    x_tot = np.zeros((12, total_time_steps))

    #The first time step is equal to the initial state of the quadrotor
    x_tot[:, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #At each time-step, update the position array x_tot with the linear_dynamics
    for j in range (1, total_time_steps):
        target_state = np.array([0, trajectory[j][0], 0, trajectory[j][1], 0, trajectory[j][2], omega, 0, omega, 0, omega, 0])
        x0 = linear_dynamics(j, x0, A, B, target_state, (0, 0)) * time_step + x0
        x_tot[:, j] = x0

    # Creating subplots for the figure-8 graphics
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Quadrotor Figure-8 Path\nHaving parameters Amplitude X: {At}, Amplitude Y: {Bt}, and Angular Velocity {omega}",
                 fontsize=12)

    # Plotting trajectory in X-Y plane (Figure-8 Path)
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1],
                   label="Path of Quadrotor in Figure Eight")
    axs[0, 0].set_xlabel('X meters')
    axs[0, 0].set_ylabel('Y meters')
    axs[0, 0].set_title("Quadrotor Figure-8 Path")

    # Plotting X-coordinate over time
    axs[0, 1].plot(time_interval_range, trajectory[:, 0],
                   label="Change in X-coordinate over time")
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('X meters')
    axs[0, 1].set_title("X-Coordinates of the Quadrotor Over Time")

    # Plotting Y-coordinate over time
    axs[1, 0].plot(time_interval_range, trajectory[:, 1],
                   label="Change in Y-coordinate over time")
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Y meters')
    axs[1, 0].set_title("Y-Coordinates of the Quadrotor Over Time")

    # Plotting Z-coordinate over time
    axs[1, 1].plot(time_interval_range, trajectory[:, 2],
                   label="Change in Z-coordinate over time")
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Z meters')
    axs[1, 1].set_title("Z-Coordinates of the Quadrotor Over Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def SRG_Simulation(time_steps=0.01, desired_coord=[1,1,1,0]):

    #x0 is the inital state of the quadrotor
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  

    #Transforming the desired (target) point into a 4x1 vector
    desired_coord = np.array(desired_coord).reshape(-1, 1)  
    
    #Initial feasible control vk (the first valid point along the line from A to B), this serves as the starting point
    vk = 0.01 * desired_coord  

    #ell_star is the number of iterations to perform when forming the contraint matrices Hx, Hv, and h
    ell_star = 1000

    #Time interval for the continuous-time system
    time_interval = np.arange(0, 10 + time_steps, time_steps)  

    #Number of time steps for Euler’s method loop
    N = len(time_interval)  

    #Initialize matrices for the reference governor Hx, Hv, and h (the contraint matrices)

    #S is a constraint matrix that uses the feedback matrices K and Kc
    S = np.block([[-K, Kc], [K, -Kc]])

    #This function constructs the constraint matrix Hx
    def construct_Hx(S, A, x, ell_star):
        Hx = []
        #Loop to construct the stacked Hx matrix
        for l in range(ell_star + 1):

            # A^l @ x
            Al_x = np.linalg.matrix_power(A, l).dot(x)  
            
            #S @ Al_x
            Hx.append(S.dot(Al_x))  

        return np.vstack(Hx) 
    
    #This function constructs the constraint matrix Hv
    def construct_Hv(S, A, B, ell_star):

        Hv = []
        
        #Identity matrix with same dimensions as A
        I = np.eye(A.shape[0])  

        #First block is zero
        Hv.append(np.zeros((S.shape[0], B.shape[1])))  
        
        #Loop to construct the other blocks: SB and the inverse term
        for l in range(1, ell_star + 1):

            Al_B = np.linalg.matrix_power(A, l-1).dot(B)

            Hv.append(S.dot(Al_B))
        
        #Adding the final block
        final_block = np.linalg.inv(I - A).dot(I - np.linalg.matrix_power(A, ell_star)).dot(B)
        Hv.append(S.dot(final_block))
        
        return np.vstack(Hv) 
    
    #This function constructs h
    def construct_h(s, epsilon, ell_star):

        h = [s] * ell_star  
        #Last element is s - epsilon
        h.append(s - epsilon) 

        return np.array(h).reshape(-1, 1)
    
    Hx = construct_Hx(S, A, x0, ell_star)
    Hv = construct_Hv(S, A, B, ell_star)

    #The constraints
    s = [6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005].T
    h = construct_h(s, 0.005, ell_star)

    # Initialize x array, xx is the evolving state over time (what gets updated at each step of the reference governor)
    xx = np.zeros((16, N))  # Assuming xx has 16 rows for states and N columns for time steps
    xx[:, 0] = x0.flatten()  # Set initial state

    #Defining the blocks for integral control
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block(
        [
            # Block 1: A-BK and BKc should have the same row dimensions
            [A - B @ K, B @ Kc],
            # Block 2: -BcC and Ac should have the same row dimensions
            [-Bc @ C, Ac]
        ]
    )
    Bcl = np.vstack([np.zeros((12, 4)), Bc])

    #Define the function rg for reference governor computation 
    def rg(Hx, Hv, h, desired_coord, vk, state):
        """This function is not yet complete

        Args:
            Hx (_type_): _description_
            Hv (_type_): _description_
            h (_type_): _description_
            desired_coord (_type_): _description_
            vk (_type_): _description_
            previous_time_step (_type_): _description_
        """
        kappa = (h - (Hx.T @ state) - (Hv.T @ vk)) / (Hv.T * (desired_coord - vk))
        return vk, kappa

    
    ts1 = time_steps 

    #Compute the matrices for the reference governor (the simulation loop)
    for i in range(1, N):

        t = (i - 1) * time_steps

        #Update the reference governor if ts1 > ts
        if (t % ts1) < time_steps:

            k, kappa = rg(Hx, Hv, h, desired_coord, vk, xx[:, i - 1])  # rg function call

            vk = vk + kappa * (desired_coord - vk)  # Update vk

        # Compute state updates
        xx[12:16, i] = xx[12:16, i - 1] + (vk - xx[1, i - 1:i]) * time_steps  # Update for states 13 to 16
        xx[:12, i] = linear_dynamics_integral(i, xx[:12, i-1], Acl, Bcl, target_state) * time_steps + x0  # Update for states 1 to 12


    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Simulating with Euler and Non-linear Integral Control")
    
    # Change in X over time
    axs[0, 0].plot(time_interval, xx[1, :], label='X Change')
    axs[0, 0].set_title('Change in X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('X Position')

    # Change in Y over time
    axs[0, 1].plot(time_interval, xx[3, :], label='Y Change', color='orange')
    axs[0, 1].set_title('Change in Y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Y Position')

    # Change in Z over time
    axs[1, 0].plot(time_interval, xx[5, :], label='Z Change', color='green')
    axs[1, 0].set_title('Change in Z')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Z Position')

    # 3D Trajectory Plot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(time_interval[1, :], time_interval[3, :], time_interval[5, :], label='3D Trajectory', color='red')
    ax3d.set_title('3D Trajectory')
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Z Position')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Main started")

    target_state = [
        0, 10,   # velocity and position on x
        0, 0,    # velocity and position on y
        0, 0,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0]     # angular velocity and position psi ]

    target_state_integral = [
        0, 3,   # velocity and position on x
        0, 3,    # velocity and position on y
        0, 3,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha                                                       
        0, 0,     # angular velocity and position psi ]
        0, 0,     # Integral states are 0s
        0, 0]     

    # if no bounds are passed default will be unbounded (bounds = 0)

    # Run simulation
    #simulate_nonlinear_integral_with_euler(target_state=target_state_integral)
    #simulate_linear_integral_with_euler(target_state=target_state_integral)

    # clear_bound_values()
    #simulate_quadrotor_nonlinear_controller(target_state)
    # print(f'Max force before bound: {np.max(force_before_bound)}')
    # print(f'Max force after bound: {np.max(force_after_bound)}')

    # clear_bound_values()
    # simulate_quadrotor_linear_controller(target_state, bounds=(0.4, 0))

    #clear_bound_values()
    SRG_Simulation()
    simulate_figure_8()
    simulate_quadrotor_nonlinear_controller(target_state=target_state)
    print(f'Max force before bound: {np.max(force_before_bound)}')
    print(f'Max force after bound: {np.max(force_after_bound)}')

    clear_bound_values()
    simulate_quadrotor_nonlinear_controller(
        target_state=target_state, bounds=(6, 0))

    # simulate_quadrotor_linear_integral_controller(target_state=target_state_integral)

    # Have not tested, or verified this simulate_figure_8 funtion
    # simulate_figure_8(A=9, B=33, omega=.5, z0=12)
