import numpy as np
import scipy.io
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


class QuadrotorParameters:
    def __init__(self, Mq):
        # Mass of the quadrotor (kilograms)
        self.Mq = Mq
        # Arm length of the quadrotor (meters)
        self.L = 0.2159 / 2
        # Acceleration due to gravity
        self.g = 9.81
        # Mass of the central sphere of the quadrotor (kilograms)
        self.Ms = 0.410
        # Radius of the central sphere (meters)
        self.R = 0.0503513
        # Mass of the propeller (kilograms)
        self.Mprop = 0.00311
        # Mass of motor + propeller
        self.Mm = 0.036 + self.Mprop
        # Moments of inertia
        self.Jx = ((2 * self.Ms * self.R**2) / 5) + (2 * self.L**2 * self.Mm)
        self.Jy = ((2 * self.Ms * self.R**2) / 5) + (2 * self.L**2 * self.Mm)
        self.Jz = ((2 * self.Ms * self.R**2) / 5) + (4 * self.L**2 * self.Mm)
        self.m = self.Mq

        # State-space matrices
        self.A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  -self.g, 0,  0],
            [1, 0, 0, 0, 0, 0, 0, 0,  0,  0,       0,  0],
            [0, 0, 0, 0, 0, 0, 0, self.g,  0,  0,   0,  0],
            [0, 0, 1, 0, 0, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 1, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 1, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  1,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,        0,  0],
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,        1,  0]
        ])

        self.B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1 / self.m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1 / self.Jx, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1 / self.Jy, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1 / self.Jz],
            [0, 0, 0, 0]
        ])

        self.C = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x position
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # y position
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # z position
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # psi (yaw)
        ])

        # Load control matrices K and Kc
        ControllerMatrix = scipy.io.loadmat(
            Path("ControllerMatrices") / "K.mat")
        self.K = ControllerMatrix['K']
        ControllerMatrixC = scipy.io.loadmat(
            Path("ControllerMatrices") / "Kc.mat")
        self.Kc = ControllerMatrixC['Kc']


def nonlinear_dynamics(t, curstate, target_state, params):
    u, Px, v, Py, w, Pz, phi, p, theta, q, psi, r = curstate

    # Calculate the error between current and target states
    error = curstate - target_state

    # Control input using feedback matrix K
    control = -params.K @ error

    F = control[0] + params.Mq * params.g  # Force (Throttle)
    TauPhi = control[1]     # Roll torque
    TauTheta = control[2]   # Pitch torque
    TauPsi = control[3]     # Yaw Torque

    # Non-linear dynamics equations
    uDot = ((r * v) - (q * w)) + (-params.g * np.sin(theta))
    vDot = ((p * w) - (r * u)) + (params.g * np.cos(theta) * np.sin(phi))
    wDot = ((q * u) - (p * v)) + (params.g *
                                  np.cos(theta) * np.cos(phi)) - (F / params.m)

    pDot = (((params.Jy - params.Jz) / params.Jx)
            * q * r) + (TauPhi / params.Jx)
    qDot = (((params.Jz - params.Jx) / params.Jy)
            * p * r) + (TauTheta / params.Jy)
    rDot = (((params.Jx - params.Jy) / params.Jz)
            * p * q) + (TauPsi / params.Jz)

    PxDot = u
    PyDot = v
    PzDot = w

    phiDot = p
    thetaDot = q
    psiDot = r

    return [uDot, PxDot, vDot, PyDot, wDot, PzDot, phiDot, pDot, thetaDot, qDot, psiDot, rDot]


def linear_dynamics(t, curstate, target_state, params):
    error = curstate - target_state
    control = -params.K @ error
    xDot = params.A @ curstate + params.B @ control
    return xDot


def linear_dynamics_integral(t, curstate_integral, target_state_integral, params):
    # Extended state-space matrices for integral controller
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    Acl = np.block([
        [params.A - params.B @ params.K, params.B @ params.Kc],
        [-Bc @ params.C, Ac]
    ])

    Bcl = np.vstack([np.zeros((12, 4)), Bc])

    xDot = Acl @ curstate_integral + Bcl @ target_state_integral
    return xDot


def simulate_quadrotor_linear_integral_controller(target_state_integral, params, initial_state=None, time_span=[0, 10]):
    """
    Simulates the quadrotor using an integral controller matrix Kc.
    """
    if initial_state is None:
        # For integral controller, state dimension is 12 + 4 = 16
        initial_state = np.zeros(16)

    sol_linear = solve_ivp(
        linear_dynamics_integral, time_span, initial_state,
        args=(target_state_integral, params), dense_output=True
    )

    # Extract positions from the solution
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]
    t = sol_linear.t  # Time

    display_plot(t, x, y, z, plot_title="Linear Integral Quadrotor Plot")


def simulate_quadrotor_linear_controller(target_state, params, initial_state=None, time_span=[0, 10]):
    """
    Simulates the quadrotor using only the feedback matrix K of the feedforward controller.
    """
    if initial_state is None:
        initial_state = np.zeros(12)

    # Solve the linear dynamics using solve_ivp
    sol_linear = solve_ivp(
        linear_dynamics, time_span, initial_state,
        args=(target_state, params), dense_output=True
    )

    # Extract positions from the solution
    x, y, z = sol_linear.y[1], sol_linear.y[3], sol_linear.y[5]
    t = sol_linear.t
    display_plot(t, x, y, z, plot_title="Linear Quadrotor Plot")


def display_plot(t, x, y, z, plot_title="Quadrotor Plot"):
    """
    Creates 3D and 2D plots to visualize the trajectory and movement of the quadrotor.
    """

    # Check if all values in x and y are zero or very close to zero
    if np.allclose(x, 0) and np.allclose(y, 0):
        x = np.ones_like(t)  # Set all x values to 1
        y = np.ones_like(t)  # Set all y values to 1

    # Create a figure for the Position vs. Time plot and the 2D plane plots
    fig = plt.figure(figsize=(15, 12))

    # Set the overall figure title
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')

    # Subplot 1: 3D Trajectory Plot
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


def simulate_quadrotor_nonlinear_controller(target_state, params, initial_state=None, time_span=[0, 10]):
    """
    Simulates the quadrotor using non-linear dynamics.
    """
    if initial_state is None:
        initial_state = np.zeros(12)

    sol_nonlinear = solve_ivp(
        nonlinear_dynamics, time_span, initial_state,
        args=(target_state, params), dense_output=True
    )

    # Extract positions from the solution
    x, y, z = sol_nonlinear.y[1], sol_nonlinear.y[3], sol_nonlinear.y[5]
    t = sol_nonlinear.t  # Time

    display_plot(t, x, y, z, plot_title="Non-Linear Quadrotor Plot")


def figure_8_trajectory(t_steps, A, B, omega, z0):
    """
    Generates the figure-eight trajectory for the quadrotor.
    """
    x_t = A * np.sin(omega * t_steps)
    y_t = B * np.sin(2 * omega * t_steps)
    z_t = z0

    return np.array([x_t, y_t, z_t])


def simulate_figure_8(A=2, B=1, omega=0.5, z0=1):
    """
    Simulates the flight of a quadrotor in a figure-eight trajectory.
    """
    # 5000 timesteps between 0 and 20 seconds
    time_interval_range = np.linspace(0, 20, 5000)

    # Obtain the trajectory for each timestep
    trajectory = np.array([figure_8_trajectory(t, A, B, omega, z0)
                          for t in time_interval_range])

    # Create subplots for the figure-8 graphics
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Quadrotor Figure-8 Path\nAmplitude X: {A}, Amplitude Y: {B}, Angular Velocity: {omega}",
                 fontsize=16, fontweight='bold')

    # Define styles and colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', '-.', ':']

    # Plot trajectory in X-Y plane
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1],
                   color=colors[0], linestyle=linestyles[0], marker=markers[0], markersize=5,
                   label="XY Path (Figure-8)")
    axs[0, 0].set_xlabel('X (meters)', fontsize=12)
    axs[0, 0].set_ylabel('Y (meters)', fontsize=12)
    axs[0, 0].set_title("Quadrotor Figure-8 Path",
                        fontsize=14, fontweight='bold')
    axs[0, 0].legend(loc='best', fontsize=10)
    axs[0, 0].grid(True)

    # Plot X-coordinate over time
    axs[0, 1].plot(time_interval_range, trajectory[:, 0],
                   color=colors[1], linestyle=linestyles[1], marker=markers[1], markersize=5,
                   label="X-Coordinate over Time")
    axs[0, 1].set_xlabel('Time (seconds)', fontsize=12)
    axs[0, 1].set_ylabel('X (meters)', fontsize=12)
    axs[0, 1].set_title("X-Coordinates over Time",
                        fontsize=14, fontweight='bold')
    axs[0, 1].legend(loc='best', fontsize=10)
    axs[0, 1].grid(True)

    # Plot Y-coordinate over time
    axs[1, 0].plot(time_interval_range, trajectory[:, 1],
                   color=colors[2], linestyle=linestyles[2], marker=markers[2], markersize=5,
                   label="Y-Coordinate over Time")
    axs[1, 0].set_xlabel('Time (seconds)', fontsize=12)
    axs[1, 0].set_ylabel('Y (meters)', fontsize=12)
    axs[1, 0].set_title("Y-Coordinates over Time",
                        fontsize=14, fontweight='bold')
    axs[1, 0].legend(loc='best', fontsize=10)
    axs[1, 0].grid(True)

    # Plot Z-coordinate over time
    axs[1, 1].plot(time_interval_range, trajectory[:, 2],
                   color=colors[3], linestyle=linestyles[3], marker=markers[3], markersize=5,
                   label="Z-Coordinate over Time")
    axs[1, 1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1, 1].set_ylabel('Z (meters)', fontsize=12)
    axs[1, 1].set_title("Z-Coordinates over Time",
                        fontsize=14, fontweight='bold')
    axs[1, 1].legend(loc='best', fontsize=10)
    axs[1, 1].grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    print("Default Parameters:")

    # Create parameters instance
    params = QuadrotorParameters(Mq=0.6)

    # Define target states
    target_state = np.array([
        0, 0,   # u, Px
        0, 0,   # v, Py
        0, 100,  # w, Pz
        0, 0,   # phi, p
        0, 0,   # theta, q
        0, 0    # psi, r
    ])

    target_state_integral = np.array([
        0, 1,   # x position, y position
        1, 0    # z position, psi (yaw)
    ])

    # Run simulations
    simulate_quadrotor_linear_controller(target_state, params)
    simulate_quadrotor_nonlinear_controller(target_state, params)
    simulate_quadrotor_linear_integral_controller(
        target_state_integral, params)

    print("Modified Parameters:")

    # Change mass to triple the original value for testing
    params = QuadrotorParameters(Mq=1.8)

    # Define target states
    target_state = np.array([
        0, 0,   # u, Px
        0, 0,   # v, Py
        0, 100,  # w, Pz
        0, 0,   # phi, p
        0, 0,   # theta, q
        0, 0    # psi, r
    ])

    target_state_integral = np.array([
        0, 1,   # x position, y position
        1, 0    # z position, psi (yaw)
    ])

    # Run simulations
    simulate_quadrotor_linear_controller(target_state, params)
    simulate_quadrotor_nonlinear_controller(target_state, params)
    simulate_quadrotor_linear_integral_controller(
        target_state_integral, params)

    # Run figure-8 trajectory simulation
    simulate_figure_8(A=9, B=33, omega=0.5, z0=12)
