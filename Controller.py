import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, cont2discrete
from scipy.integrate import odeint
import control as ctrl

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


def nonlinear_dynamics_ivp(t, curstate, target_state, bounds):
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
    error = curstate - target_state

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


def linear_dynamics_ivp(t, curstate, A, B, target_state, bounds):
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


def linear_dynamics_ode(current_state, t, K, A, B, target_state):
    # We must have different functions for ode and ivp even though they perform the same calculation because the order
    # of the arguments is different

    error = current_state - target_state
    control = -K @ error
    dx = A @ current_state + B @ control
    return dx


def nonlinear_dynamics_ode(current_state, t, K, target_state):
    # We must have different functions for ode and ivp even though they perform the same calculation because the order
    #  of the arguments is different

    error = current_state - target_state
    control = -K @ error

    F = control[0] + Mq * g  # Force -> Throttle
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque
    TauPsi = control[3]     # Yaw Torque

    u, Px, v, Py, w, Pz, phi, p, theta, q, psi, r = current_state

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


def linear_dynamics_integral(t, curstate_integral, Ac, Bc, target_state):
    """The dynamics for linear integral control (used in Euler simulation)

    Args:
        t (float): The time
        curstate_integral (array): The current state of quadrotor
        Ac (matrix): Control matrix for integral control
        Bc (matrix): Control matrix for integral control
        target_state (array): The target state of quadrotor

    Returns:
        array: The state-change array
    """

    target_state = np.array(target_state)
    curstate_integral = np.array(curstate_integral)

    # # Getting 4x1 Integral u (error between X, Y, Z and Yaw)
    # uc = (target_state[[1, 3, 5, 11]] - curstate_integral[[1, 3, 5, 11]])

    # changed by maxi no error needed
    uc = target_state[[1, 3, 5, 11]]

    # Control law
    Dotx = Ac @ curstate_integral + Bc @ uc

    return Dotx, uc


def nonlinear_dynamics_integral(t, curstate_integral, target_state):
    """
    The dynamics for nonlinear integral control.

    Args:
        t (float): The time
        curstate_integral (array): The current state of the quadrotor
        target_state (array): The target state of the quadrotor
        control (array): The control input computed outside this function

    Returns:
        array: The state-change array
    """

    u, Px, v, Py, w, Pz, p, phi, q, theta, r, psi, i1, i2, i3, i4 = curstate_integral

    control = calculate_control(curstate_integral)

    # Updating integral states
    error = np.array(target_state)[[1, 3, 5, 11]] - \
        np.array(curstate_integral)[[1, 3, 5, 11]]
    curstate_integral[12:] = error

    F0 = control[0]  # + Mq * g
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque
    TauPsi = control[3]     # Yaw Torque

    # Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
    uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (g * np.cos(theta) * np.cos(phi)) + (-F0 / Mq)

    # Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
    pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
    qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
    rDot = (((Jx - Jy) / Jz) * p * q) + ((1 / Jz) * TauPsi)

    PxDot = u
    PyDot = v
    PzDot = w

    state_dot = [
        uDot, PxDot, vDot, PyDot, wDot, PzDot,
        pDot, p, qDot, q, rDot, r,
        error[0], error[1], error[2], error[3]
    ]

    state_dot = np.array([float(el) for el in state_dot])
    control = np.array([float(el) for el in control])

    return state_dot, control


def simulate_linear_integral_with_euler(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        total_time=10, time_step=0.001):
    """This method simulates flight using the Euler method and linear integral control

    Args:
        target_state (array): The target state of quadrotor
        initial_state (list, optional): The initial state of quadrotor.
        Defaults to [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
    """

    # Constructing time steps
    time_range = np.arange(0, total_time + time_step, time_step)
    total_time_steps = len(time_range)

    # total_time_steps = np.arange(0, total_time, time_step)

    x0 = np.zeros(16,)

    # 12 is the number of state elements, total_time_steps is the state at each time step
    x_tot = np.zeros((16, total_time_steps))

    # The first time step is equal to the initial state of the quadrotor
    x_tot[:, 0] = initial_state

    # Defining the blocks for integral control
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

    control_matrix = np.zeros((4, total_time_steps))

    # At each time-step, update the position array x_tot with the linear_dynamics (Euler method loop)
    for j in range(1, total_time_steps):
        # control law:
        # control = -K @ X0 + Kc @ Xc(Xc = control)

        # update current state:
        # X0 = x0 + linear_quad function

        state_change, control = linear_dynamics_integral(
            j, x0, Acl, Bcl, target_state)

        x0 = state_change * time_step + x0
        control_matrix[:, j] = control

        x_tot[:, j] = x0

    return x_tot, control, time_range

# time steps should also be 0.001


def simulate_nonlinear_integral_with_euler(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           total_time=10, time_step=0.001, bounds=(0, 0)):
    """This method simulates flight using the Euler method and nonlinear dynamics with integral control

    Args:
        target_state (array): The target state of quadrotor
        initial_state (list, optional): The initial state of quadrotor. Defaults to [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
    """

    # Constructing time steps
    time_range = np.arange(0, total_time + time_step, time_step)
    total_time_steps = len(time_range)

    x0 = np.zeros(16,)

    # 12 is the number of state elements, total_time_steps is the state at each time step
    x_tot = np.zeros((16, total_time_steps))

    # The first time step is equal to the initial state of the quadrotor
    x_tot[:, 0] = initial_state

    # For plotting controls to see if constraints violated
    control_matrix = np.zeros((4, total_time_steps))

    # At each time-step, update the position array x_tot with the nonlinear dynamics (Euler Loop)
    for j in range(1, total_time_steps):
        state_change, control = nonlinear_dynamics_integral(
            j, x0, target_state)
        x0 = state_change * time_step + x0
        control_matrix[:, j] = control
        x_tot[:, j] = x0 * 3

    return x_tot, control_matrix, time_range


def simulate_quadrotor_linear_integral_controller(target_state, initial_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  time_span=[0, 10]):
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
    sol_linear = solve_ivp(linear_dynamics_ivp, time_span, initial_state, args=(
        A, B, target_state, bounds), dense_output=True)

    # Obtain results from the solved differential equations
    # x, y, z positions
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]
    t = sol_linear.t
    display_plot(t, x, y, z)

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

    sol_nonlinear = solve_ivp(nonlinear_dynamics_ivp, time_span, initial_state, args=(
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


def simulate_figure_8_srg(At=2, Bt=1, omega=0.5, z0=1):
    """This function simulates the flight of a quadrotor in a figure-eight trajectory using the State Reference Governor (SRG) method

    Args:
        A (float): amplitude along x-axis
        B (float): amplitude along y-axis
        omega (float): angular velocity
        z0 (float): constant altitude
    """

    # Simulate only for one full time period T (start and end at the same point)
    T = 2*np.pi/omega
    # Number of time steps, this determines the precision of the figure-8 trajectory
    tt = np.linspace(0, T, 100)

    # Obtaining the coordinates (trajectory) for each timestep in time_interval_range
    # Since z0 is constant, we'll add it manually here.
    x = At * np.sin(omega * tt)
    y = Bt * np.sin(2 * omega * tt)

    # Create waypoints to represent target state at each time step in tt
    target_state = np.array([
        [0, x[i], 0, y[i], 0, z0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(tt))
    ])

    # Create a zero state, because it has to start from (0, 0, 0) and move to (0, 0, 1) before starting the figure-8
    zero_state = np.zeros(16)
    target_state = np.vstack((zero_state, target_state))

    # This is to remember the whole journey including intermediate steps between waypoints
    x_trajectory = []

    # Now make it move according to the new target states
    # Main loop that runs the figure-8 trajectory

    length, _ = target_state.shape
    for i in range(length - 1):
        print(f'\r\033[92mIteration: {i+1}/{length}\033[0m', end='')
        # results, control, time_interval, _ = SRG_Simulation_Linear(
        #     desired_state=target_state[i+1], initial_state=target_state[i])
        results, control, time_interval, _ = SRG_Simulation_Nonlinear(
            desired_state=target_state[i+1], initial_state=target_state[i], ell_star_figure8=True)
        x_trajectory.append(results[:, -1])

    print("\n")

    # Converting the python list into numpy array
    x_trajectory = np.vstack(x_trajectory)

    # Extract x, y, z from indices 1, 3, and 5 in x_trajectory
    x_vals = x_trajectory[:, 1]
    y_vals = x_trajectory[:, 3]
    z_vals = x_trajectory[:, 5]
    time_vals = np.linspace(0, T, len(x_trajectory))

    # Plotting
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"""Quadrotor Figure-8 Path With Amplitude X: {At}, Amplitude Y: {
                 Bt}, and Angular Velocity {omega}""", fontsize=14)

    # Create 2x2 subplots
    axs = fig.add_subplot(2, 2, 1)
    axs.set_title("Quadrotor Figure-8 Path")
    axs.plot(x_vals, y_vals, label="Path of Quadrotor in Figure Eight")
    axs.set_xlabel("X (meters)")
    axs.set_ylabel("Y (meters)")

    # Plotting X-coordinate over time
    axs = fig.add_subplot(2, 2, 2)
    axs.plot(time_vals, x_vals, label="X-coordinate over time")
    axs.set_xlabel("Time (seconds)")
    axs.set_ylabel("X (meters)")
    axs.set_title("X-Coordinates of the Quadrotor Over Time")

    # Plotting Y-coordinate over time
    axs = fig.add_subplot(2, 2, 3)
    axs.plot(time_vals, y_vals, label="Y-coordinate over time")
    axs.set_xlabel("Time (seconds)")
    axs.set_ylabel("Y (meters)")
    axs.set_title("Y-Coordinates of the Quadrotor Over Time")

    # Plotting Z-coordinate over time (constant altitude)
    axs = fig.add_subplot(2, 2, 4)
    axs.plot(time_vals, z_vals, label="Z-coordinate over time")
    axs.set_xlabel("Time (seconds)")
    axs.set_ylabel("Z (meters)")
    axs.set_title("Z-Coordinates of the Quadrotor Over Time")

    # Adding a 3D plot to show x_vals, y_vals, and z_vals trajectory
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(x_vals, y_vals, z_vals,
               label="3D Path of Quadrotor in Figure Eight")
    ax_3d.set_xlabel("X (meters)")
    ax_3d.set_ylabel("Y (meters)")
    ax_3d.set_zlabel("Z (meters)")
    ax_3d.set_title("3D Trajectory of Quadrotor Figure-8 Path")
    ax_3d.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_SRG_simulation(time_interval, xx, target_state, kappas):
    """Plots the results of the SRG simulation

    Args:
        time_interval (Vector): A vector of time steps
        xx (Matrix): The stacked matrix containing states at each time step from Euler simulation
        target_state (Vector): The target states for X, Y, and Z
        kappas (Vector): The values of kappa over time
    """

    # Plotting results
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"""Reference Governor Flight. \n Target state X: {
                 target_state[1]}, Y: {target_state[3]}, and Z: {target_state[5]}""")

    # Change in X over time
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(xx[1, :], label='X Change')
    ax1.set_title('Change in X')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X Position')

    # Change in Y over time
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(xx[3, :], label='Y Change', color='orange')
    ax2.set_title('Change in Y')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y Position')

    # Change in Z over time
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(xx[5, :], label='Z Change', color='green')
    ax3.set_title('Change in Z')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z Position')

    # 3D plot of X, Y, Z states
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax4.plot(xx[1, :], xx[3, :], xx[5, :],
             label='3D Trajectory', color='purple')
    ax4.set_title('3D State Trajectory')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('Z Position')

    # Kappas over time
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(kappas, label='Kappas', color='red')
    ax5.set_title('Kappas over Time')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Kappa Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig("simulation.png", dpi=300)
    plt.close(fig)  # Close the figure to free memory


def plot_SRG_controls(time_interval, controls, target_state):
    """Plots the control variables with constraints for the SRG simulation.

    Args:
        time_interval (Vector): A vector of time steps.
        controls (Matrix): A 4xN matrix where each row represents a control variable over time.
        target_state (Vector): The target state
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"""SRG Control Variables with Constraints and Target X = {
                 target_state[1]}, Y = {target_state[3]}, Z = {target_state[5]}""")

    # Plot for controls[0]
    axs[0, 0].plot(controls[0, :], label='Control 1')
    axs[0, 0].axhline(y=6, color='red', linestyle='--',
                      label='Constraint at 6')
    axs[0, 0].set_title('Control Variable 1')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Control 1')
    axs[0, 0].legend()

    # Plot for controls[1]
    axs[0, 1].plot(controls[1, :], label='Control 2')
    axs[0, 1].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[0, 1].set_title('Control Variable 2')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Control 2')
    axs[0, 1].legend()

    # Plot for controls[2]
    axs[1, 0].plot(controls[2, :], label='Control 3')
    axs[1, 0].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[1, 0].set_title('Control Variable 3')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Control 3')
    axs[1, 0].legend()

    # Plot for controls[3]
    axs[1, 1].plot(controls[3, :], label='Control 4')
    axs[1, 1].axhline(y=0.005, color='red', linestyle='--',
                      label='Constraint at 0.005')
    axs[1, 1].set_title('Control Variable 4')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Control 4')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("controls.png", dpi=300)
    plt.close(fig)


def SRG_Simulation_Linear(desired_state, time_steps=0.001,
                          initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), figure8waypoints=None):
    """The state-reference governor simulation with linear dynamics

    Args:
        desired_state (vector): The target state
        time_steps (float, optional): The time steps. Defaults to 0.001.
        initial_state (vector, optional): The initial state. Defaults to np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).

    Returns:
        Matrix: The state-matrix of the Euler simulation
    """
    print("In Linear SRG")
    x0 = initial_state
    desired_coord = None

    desired_coord = np.zeros(4)
    desired_coord[0] = desired_state[1]
    desired_coord[1] = desired_state[3]
    desired_coord[2] = desired_state[5]
    desired_coord[3] = desired_state[11]

    # Initial feasible control vk (a 4x1 vector)
    # (the first valid point along the line from A to B), this serves as the starting point
    vk = 0.001 * desired_coord

    # ime interval for the continuous-time system
    # change time to longer!!!! ERROR
    time_interval = np.arange(0, 65 + time_steps, time_steps)
    # This is filled with 0.001, 0.002, 0.003 etc

    # Number of time steps for Euler’s method loop
    N = len(time_interval)

    # S is a constraint matrix that uses the feedback matrices K and Kc. S @ x needs to be less than the constraints
    S = np.block([[-K, Kc],
                  [K, -Kc]])

    # Defining the blocks for integral control, we need to use discrete versions of A_cl and B_cl
    # so we obtain Ad and Bd to use in governor
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block([
        [A - B @ K, B @ Kc],
        [-Bc @ C, Ac]
    ])

    Bcl = np.vstack([np.zeros((12, 4)), Bc])
    Ccl = np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))
    Dcl = np.zeros((C.shape[0], B.shape[1]))

    sys = ctrl.ss(Acl, Bcl, Ccl, Dcl)
    sysd = ctrl.c2d(sys, 0.01)

    # The final discrete matrices to use in the closed-loop system
    Ad = sysd.A
    Bd = sysd.B

    # Reference Governor Initialization
    lstar = 1000
    I = np.eye(16)
    Hx = []
    Hv = []
    h = []
    s = np.array([6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005])
    S = np.block([[-K, Kc], [K, -Kc]])

    # Build Hx, Hv, and h matrices
    for l in range(1, lstar + 1):
        Hx.append(S @ np.linalg.matrix_power(Ad, l))
        Hv.append(S @ np.linalg.inv(I - Ad) @
                  (I - np.linalg.matrix_power(Ad, l)) @ Bd)
        h.append(s)

    Hx = np.vstack(Hx + [np.zeros((8, 16))])
    Hv = np.vstack(Hv + [S @ np.linalg.inv(I - Ad) @ Bd])
    h = np.hstack(h + [s * 0.99])

    # Initialize x array (evolving state over time)
    xx = np.zeros((16, N))
    xx[:, 0] = x0.flatten()

    # The control function for Euler simulation

    def qds_dt(x, uc, Acl, Bcl):
        # Integral control (linear version)
        a = Acl @ x + Bcl @ uc
        return a

    # Main simulation loop for SRG Euler simulation
    ts1 = 0.01
    vk_values = []
    controls = np.zeros((4, N))
    kappas = []
    x_old = None
    indexwaypoints = 0

    for i in range(1, N):

        t = round((i - 1) * time_steps, 6)

        if (t % ts1) < time_steps and i != 1:

            if x_old is not None:

                # Check for figure eight
                if figure8waypoints:

                    desired_coord, indexwaypoints = update_waypoints_function(
                        xx=x_old, waypoints=figure8waypoints, current_waypoint_index=indexwaypoints)

                kappa = rg(Hx, Hv, h, desired_coord, vk, x_old)
                kappas.append(kappa)
                vk = vk + kappa * (desired_coord - vk)

            x_old = np.block([xx[:, i-1]])

        vk_values.append(vk)

        # Integral control
        u = -K @ xx[:12, i - 1] + Kc @ xx[12:16, i - 1]
        controls[:, i] = u.reshape(1, 4)[0]
        xx[12:, i] = xx[12:, i-1] + \
            (vk.reshape(1, 4)[0] - xx[[1, 3, 5, 11], i-1]) * time_steps
        xx[:12, i] = xx[:12, i-1] + \
            qds_dt(xx[:, i-1], u, Acl, Bcl)[:12] * time_steps

    return xx, controls, time_interval, kappas, vk_values


def rg(Hx, Hv, h, r, vk, xx):
    M = Hx.shape[0]
    k = np.zeros(M)

    for j in range(M):

        alpha = Hv[j, :] @ (r - vk)
        beta = h[j] - Hx[j, :] @ xx - Hv[j, :] @ vk

        if alpha > 0:
            k[j] = beta / alpha
        else:
            k[j] = 1
        if beta < 0:
            k[j] = 0

        kappa = min(k)
    return kappa


def calculate_control(curstate_integral):
    control = np.block([-K, Kc]) @ curstate_integral
    return control


def update_waypoints_function(xx, waypoints, current_waypoint_index):
    """
    Used to update waypoints and fly figure 8 in simulations.
    """
    delta_x = waypoints[current_waypoint_index][0] - xx[1]  # Index 1 for x
    delta_y = waypoints[current_waypoint_index][1] - xx[3]  # Index 3 for y
    delta_z = waypoints[current_waypoint_index][2] - xx[5]  # Index 5 for z

    # Compute Euclidean distance
    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    if distance < 0.1 and current_waypoint_index < len(waypoints) - 1:
        # Move to the next waypoint
        new_waypoint_index = current_waypoint_index + 1
        print(f"Reached waypoint {current_waypoint_index}. Switching to waypoint {
              new_waypoint_index}.")
        return waypoints[new_waypoint_index], new_waypoint_index
    else:
        return waypoints[current_waypoint_index], current_waypoint_index


def SRG_Simulation_Nonlinear(desired_state, time_steps=0.001,
                             initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), figure8waypoints=None):

    print("In SRG Simulation_Nonlinear function")
    x0 = initial_state
    desired_coord = None

    desired_coord = np.zeros(4)
    desired_coord[0] = desired_state[1]
    desired_coord[1] = desired_state[3]
    desired_coord[2] = desired_state[5]
    desired_coord[3] = desired_state[11]

    # Initial feasible control vk (a 4x1 vector)
    # (the first valid point along the line from A to B), this serves as the starting point
    vk = 0.001 * desired_coord

    # time interval for the continuous-time system
    time_interval = np.arange(0, 65 + time_steps, time_steps)
    # This is filled with 0.001, 0.002, 0.003 etc

    # Number of time steps for Euler’s method loop
    N = len(time_interval)

    # Defining the blocks for integral control, we need to use discrete versions of A_cl and B_cl
    # so we obtain Ad and Bd to use in governor
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block([
        [A - B @ K, B @ Kc],
        [-Bc @ C, Ac]
    ])

    Bcl = np.vstack([np.zeros((12, 4)), Bc])
    Ccl = np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))
    Dcl = np.zeros((C.shape[0], B.shape[1]))

    sys = ctrl.ss(Acl, Bcl, Ccl, Dcl)
    sysd = ctrl.c2d(sys, 0.01)

    # The final discrete matrices to use in the closed-loop system
    Ad = sysd.A
    Bd = sysd.B

    # Reference Governor Initialization
    lstar = 100
    I = np.eye(16)
    Hx = []
    Hv = []
    h = []
    s = np.array([6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005])
    S = np.block([[-K, Kc], [K, -Kc]])

    # Build Hx, Hv, and h matrices
    for l in range(1, lstar + 1):
        Hx.append(S @ np.linalg.matrix_power(Ad, l))
        Hv.append(S @ np.linalg.inv(I - Ad) @
                  (I - np.linalg.matrix_power(Ad, l)) @ Bd)
        h.append(s)

    Hx = np.vstack(Hx + [np.zeros((8, 16))])
    Hv = np.vstack(Hv + [S @ np.linalg.inv(I - Ad) @ Bd])
    h = np.hstack(h + [s * 0.99])

    # Initialize x array (evolving state over time)
    xx = np.zeros((16, N))
    xx[:, 0] = x0.flatten()

    # The control function for Euler simulation
    def qds_dt_nonlinear(x, target, vk):

        u, Px, v, Py, w, Pz, p, phi, q, theta, r, psi, i1, i2, i3, i4 = x

        control = calculate_control(x)

        # Updating integral states
        error = np.array([vk[0] - x[1], vk[1] - x[3],
                         vk[2] - x[5], vk[3] - x[11]])

        F0 = control[0]  # + Mq * g
        TauPhi = control[1]     # Roll torque
        TauTheta = control[2]   # Pitch torque
        TauPsi = control[3]     # Yaw Torque

        # Calculating the non-linear change dynamics for linear velocities (equation 3 slide 32)
        uDot = ((r * v) - (q * w)) + (-g * np.sin(theta))
        vDot = ((p * w) - (r * u)) + (g * np.cos(theta) * np.sin(phi))
        wDot = ((q * u) - (p * v)) + \
            (g * np.cos(theta) * np.cos(phi)) + (-F0 / Mq)

        # Calculating the non-linear change dynamics for angular velocities (equation 4 slide 32)
        pDot = (((Jy - Jz) / Jx) * q * r) + ((1 / Jx) * TauPhi)
        qDot = (((Jz - Jx) / Jy) * p * r) + ((1 / Jy) * TauTheta)
        rDot = (((Jx - Jy) / Jz) * p * q) + ((1 / Jz) * TauPsi)

        PxDot = u
        PyDot = v
        PzDot = w

        state_dot = [
            uDot, PxDot, vDot, PyDot, wDot, PzDot,
            pDot, p, qDot, q, rDot, r,
            error[0], error[1], error[2], error[3]
        ]

        state_dot = np.array([float(el) for el in state_dot])

        return state_dot, control

    # Main simulation loop for SRG Euler simulation
    # Sampling time for reference governor (ts1 > time_steps)
    # This ts1 should be 0.01 since the rg is running slower
    ts1 = 0.01
    controls = np.zeros((4, N))
    kappas = []
    vk_values = []
    x_old = None
    indexwaypoints = 0

    for i in range(1, N):
        print(f"\r\033[92mIteration: {i}/{N}\033[0m", end='')
        t = round((i - 1) * time_steps, 6)
        if (t % ts1) < time_steps and i != 1:
            if x_old is not None:
                # only used when giving multiple waypoints
                if figure8waypoints:
                    desired_coord, indexwaypoints = update_waypoints_function(
                        xx=x_old, waypoints=figure8waypoints, current_waypoint_index=indexwaypoints)

                kappa = rg(Hx, Hv, h, desired_coord, vk, x_old)
                kappas.append(kappa)
                vk = vk + kappa * (desired_coord - vk)

            x_old = np.block([xx[:, i-1]])

        vk_values.append(vk)
        state_change, control = qds_dt_nonlinear(xx[:, i-1], desired_coord, vk)
        xx[:, i] = xx[:, i-1] + state_change * time_steps
        controls[:, i] = control.reshape(1, 4)[0]

    return xx, controls, time_interval, kappas, vk_values


def plot_vk_values(time_interval, vk_values):
    """
    Plots the evolution of vk components over time.
    """
    # Ensure vk_values is a NumPy array
    vk_values = np.array(vk_values)

    # Ensure the time_interval matches the vk_values length
    time_axis = time_interval[:len(vk_values)]

    # Plotting vk components
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Evolution of $v_k$ Components Over Time")

    # Plot each component of vk
    # vk_values.shape[1] = number of vk components
    for i in range(vk_values.shape[1]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(time_axis, vk_values[:, i], label=f'''$v_k[{
                i}]$''', color=plt.cm.viridis(i / vk_values.shape[1]))
        ax.set_title(f'$v_k[{i}]$ Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'$v_k[{i}]$ Value')
        ax.legend()
        ax.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig("vk_values.png", dpi=300)
    plt.close(fig)  # Close the figure to free memory


def generate_figure8_waypoints(radius=1.5, num_waypoints=25, z_fixed=0.5):
    """
    Generate waypoints for a figure-8 trajectory. - used as input to simulate figure 8
    """
    waypoints = []

    # Generate points along the figure-8 curve
    for i in range(num_waypoints):
        theta = i * (2 * np.pi / num_waypoints)  # Evenly spaced angles

        # Parametric equation for the figure-8
        x = radius * np.sin(theta)
        y = radius * np.sin(theta) * np.cos(theta)  # Shape of the figure-8
        z = z_fixed  # Fixed z value
        # Placeholder for any additional parameter (e.g., yaw or speed)
        u = 0.0

        # Add the waypoint as a list of floats
        waypoints.append([float(x), float(y), float(z), float(u)])
    # Ensure the last waypoint is exactly at [0, 0, z, 0] to close the figure-8
    waypoints[-1] = [0.0, 0.0, float(z_fixed), 0.0]

    return waypoints


def plot_figure88(inputarrfigure8):
    """
    Plots the figure-8 waypoints on a 2D graph (XY plane). Function used to test create figure 8 wp's function
    """
    # Extract x and y coordinates from the waypoints
    x_coords = [wp[0] for wp in inputarrfigure8]
    y_coords = [wp[1] for wp in inputarrfigure8]
    # Plot the waypoints on a 2D graph
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', color='b', label="Figure 8 Path")
    plt.title("Figure 8 Waypoints in XY Plane")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_trajectory_vs_waypoints(xx, waypoints):
    """
    Plots the actual trajectory (from xx) over the figure-8 waypoints. 
    To show how close simulation is to actual trajectory
    """
    # Extract the actual trajectory (x, y) from xx
    x_actual = xx[1, :]  # xx[1, :] corresponds to x position over time
    y_actual = xx[3, :]  # xx[3, :] corresponds to y position over time
    # Extract the waypoints (x, y) for the figure-8 path
    x_waypoints = [wp[0] for wp in waypoints]  # Waypoint x values
    y_waypoints = [wp[1] for wp in waypoints]  # Waypoint y value
    # Plot the actual trajectory (xx) and the figure-8 waypoints
    plt.figure(figsize=(8, 8))
    # Plot the waypoints (figure-8)
    plt.plot(x_waypoints, y_waypoints, 'g--',
             label="Figure-8 Waypoints", alpha=0.6)
    # Plot the actual trajectory (xx)
    plt.plot(x_actual, y_actual, 'b-', label="Actual Trajectory", alpha=0.8)
    plt.title("Actual Trajectory vs Figure-8 Waypoints")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    print("Main started")

    target_state = [
        0, 100,   # velocity and position on x
        0, 100,    # velocity and position on y
        0, 100,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0]     # angular velocity and position psi ]

    target_state_16 = [
        0, 100,   # velocity and position on x
        0, 100,    # velocity and position on y
        0, 100,    # velocity and position on z
        0, 0,     # angular velocity and position thi
        0, 0,     # angular velocity and position thetha
        0, 0,     # angular velocity and position psi ]
        0, 0,     # Integral states are 0s
        0, 0]

    # Run simulation for EULERS METHOD: This should work like it is Part1 in project writeup
    # In my oppinion there is a wrong frequency in this used.? 0.001 should be used
    # results, control, time_interval = simulate_nonlinear_integral_with_euler(target_state=target_state_16)
    # results, control, time_interval = simulate_linear_integral_with_euler(
    #     target_state=target_state)
    # need better plots, that show anything else than the trajectory?
    # plot_SRG_simulation(time_interval, results,
    #                     target_state, kappas=[0]*10001)


# ----------------------------------------------------------------
# Display functions do not work/ do not display full length because now longer when flying to many wp

    # These 2 functions work. They take a while to run ~ 1hour. But then display result
    # inputarrfigure8  = [
    # [0, 0, 1,0],    # Move up to z = 1
    # [1, 0, 1,0],    # First segment of the 8: Right loop starts
    # # [1, 1, 1,0],    # Continue in a straight line
    # # [0, 1, 1,0],    # Middle of the "8"
    # # [-1, 1, 1,0],   # Left loop starts
    # # [-1, 0, 1,0],   # Continue in a straight line
    # # [0, 0, 1,0],    # Return to the center
    # # [0, 0, 0,0],    # Descend back to starting point
    # ]

    # Display functions should not use time, so that they display all of it

    # waypointsfigure8 = generate_figure8_waypoints(
    #     radius=1.0, num_waypoints=20, z_fixed=0.5)
    # print(waypointsfigure8)
    # plot_figure88(waypointsfigure8)

    # exit()

    # CHANGE 300 and 17 !! Time and lstar!!! BEFORE NIGHT !!!

    # xx, controls, time_interval, kappas, vk_values = SRG_Simulation_Linear(
    #     desired_state=target_state_16, figure8waypoints=None)
    xx, controls, time_interval, kappas, vk_values = SRG_Simulation_Nonlinear(
        desired_state=target_state_16, figure8waypoints=None)
    print("Done")

    plot_vk_values(time_interval, vk_values)

    plot_SRG_simulation(time_interval, xx,
                        target_state=target_state_16, kappas=kappas)

    plot_SRG_controls(time_interval, controls, target_state_16)
    # plot_trajectory_vs_waypoints(xx, waypointsfigure8)

    # TODO:
    # Write the update waypoint function and include them into the loop. DONE

    # ALso throw out all the old stuff of this file .?
    # Also check if this should run with 0.01 and 0.1 like it says on file?

# ------------------------------------------------------------------

    # Chris Function: Takes forever????
    # simulate_figure_8_srg(At=9, Bt=33, omega=.5, z0=12)

    # simulate_figure8_srg_max()


# I think old stuff??    # simulate_figure_8()
    # simulate_quadrotor_nonlinear_controller(target_state=target_state)
    # print(f'Max force before bound: {np.max(force_before_bound)}')
    # print(f'Max force after bound: {np.max(force_after_bound)}')

    # clear_bound_values()
    # simulate_quadrotor_nonlinear_controller(
    # target_state=target_state, bounds=(6, 0))

    # simulate_quadrotor_linear_integral_controller(target_state=target_state_16)

    # Have not tested, or verified this simulate_figure_8 funtion
    # simulate_figure_8(At=9, Bt=33, omega=.5, z0=12)

    # # clear_bound_values()
    # simulate_quadrotor_nonlinear_controller(target_state)
    # print(f'Max force before bound: {np.max(force_before_bound)}')
    # simulate_quadrotor_linear_controller(target_state, bounds=(0.4, 0))
    # print(f'Max force after bound: {np.max(force_after_bound)}')

    # clear_bound_values()
    # simulate_quadrotor_linear_controller(target_state, bounds=(0.4, 0))

    # clear_bound_values()
    # xx, controls, time_interval, kappas= SRG_Simulation_Linear(desired_state=target_state_16)
