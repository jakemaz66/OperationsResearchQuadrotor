import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sys
import pandas as pd

from ControllerMatrices.quadrotor_parameters import (
    Mq, L, g, Ms, R, Mprop, Mm, Jx, Jy, Jz, m,
    A, B, C, K, Kc
)


def calculate_control(curstate_integral, K, Kc):
    """
    Calculate control input based on current state.

    Parameters:
        curstate_integral (np.array): Current state vector.
        K (np.array): State feedback gain matrix.
        Kc (np.array): Integral gain matrix.

    Returns:
        np.array: Control input vector.
    """
    return np.block([-K, Kc]) @ curstate_integral


def rg(Hx, Hv, h, r, vk, xx):
    """
    Reference Governor function.

    Parameters:
        Hx, Hv, h: Matrices for constraint handling.
        r (np.array): Reference vector.
        vk (np.array): Current vk vector.
        xx (np.array): Current state vector.

    Returns:
        float: Scaling factor kappa.
    """
    M = Hx.shape[0]
    k = np.zeros(M)
    for j in range(M):
        alpha = Hv[j, :] @ (r - vk)
        beta = h[j] - Hx[j, :] @ xx - Hv[j, :] @ vk
        k[j] = beta / alpha if alpha > 0 else 1
        if beta < 0:
            k[j] = 0
    return min(k)


def SRG_Simulation_Linear(desired_state, time_steps=0.001, initial_state=np.zeros(16),
                          figure8waypoints=None, enable_constraints=True, enable_modified_target=True):
    """
    Simulate the quadrotor dynamics with optional constraints and modified target.

    Parameters:
        desired_state (list or np.array): Desired state vector.
        time_steps (float): Simulation time step.
        initial_state (np.array): Initial state vector.
        figure8waypoints (list): Waypoints for figure-8 trajectory (unused in current implementation).
        enable_constraints (bool): Flag to enable or disable constraints.
        enable_modified_target (bool): Flag to enable or disable modified target.

    Returns:
        tuple: States, controls, time interval, kappas, vk_values.
    """
    x0 = initial_state
    desired_coord = np.zeros(4)

    # Apply modified target if enabled
    if enable_modified_target:
        desired_coord[0] = 4  # x = 4
        desired_coord[1] = 5  # y = 5
        desired_coord[2] = 3  # z = 3
        desired_coord[3] = desired_state[11]  # maintain target psi
    else:
        desired_coord[0] = desired_state[1]
        desired_coord[1] = desired_state[3]
        desired_coord[2] = desired_state[5]
        desired_coord[3] = desired_state[11]

    vk = 0.001 * desired_coord
    total_time = 10
    time_interval = np.arange(0, total_time + time_steps, time_steps)
    N = len(time_interval)
    Ac = np.zeros((4, 4))
    Bc = np.eye(4)

    # Construct the state-space system
    system = ctrl.ss(
        np.block([[A - B @ K, B @ Kc], [-Bc @ C, Ac]]),
        np.vstack([np.zeros((12, 4)), Bc]),
        np.hstack((C, np.zeros((C.shape[0], C.shape[0])))),
        np.zeros((C.shape[0], B.shape[1]))
    )
    sysd = ctrl.c2d(system, 0.01)
    Ad, Bd = sysd.A, sysd.B
    lstar = 1000
    I = np.eye(16)
    Hx, Hv, h, s = [], [], [], np.array(
        [6, 0.005, 0.005, 0.005, 6, 0.005, 0.005, 0.005])
    S = np.block([[-K, Kc], [K, -Kc]])
    for l in range(1, lstar + 1):
        Hx.append(S @ np.linalg.matrix_power(Ad, l))
        Hv.append(S @ np.linalg.inv(I - Ad) @
                  (I - np.linalg.matrix_power(Ad, l)) @ Bd)
        h.append(s)
    Hx = np.vstack(Hx + [np.zeros((8, 16))])
    Hv = np.vstack(Hv + [S @ np.linalg.inv(I - Ad) @ Bd])
    h = np.hstack(h + [s * 0.99])

    xx = np.zeros((16, N))
    xx[:, 0] = x0.flatten()
    ts1 = 0.01
    vk_values = []
    controls = np.zeros((4, N))
    kappas = []
    x_old = None

    print(
        f"\nStarting simulation with constraints {'enabled' if enable_constraints else 'disabled'}.")

    for i in range(1, N):
        progress = int(50 * i / N)
        sys.stdout.write(
            f'\r[{"#" * progress}{"-" * (50 - progress)}] {int(100 * i / N)}%')
        sys.stdout.flush()
        t = round((i - 1) * time_steps, 6)

        # Apply constraints before computing control
        if enable_constraints:
            # Assuming xx[1, :] = x position and xx[3, :] = y position
            # Adjust indices if necessary based on your state vector
            clamped_x = np.clip(xx[1, i - 1], -3, 3)
            clamped_y = np.clip(xx[3, i - 1], -3, 3)
            if clamped_x != xx[1, i - 1] or clamped_y != xx[3, i - 1]:
                print(f"\nClamped at step {i}: x from {xx[1, i - 1]:.4f} to {clamped_x:.4f}, "
                      f"y from {xx[3, i - 1]:.4f} to {clamped_y:.4f}")
            xx[1, i - 1] = clamped_x
            xx[3, i - 1] = clamped_y

        if (t % ts1) < time_steps and i != 1:
            if x_old is not None:
                kappa = rg(Hx, Hv, h, desired_coord, vk, x_old)
                kappas.append(kappa)
                vk = vk + kappa * (desired_coord - vk)
            x_old = xx[:, i - 1].copy()

        vk_values.append(vk.copy())

        # Compute control input based on the current state
        u = -K @ xx[:12, i - 1] + Kc @ xx[12:16, i - 1]
        controls[:, i] = u.reshape(1, 4)[0]

        # Update integral states (assuming xx[12:16] are integral states)
        xx[12:, i] = xx[12:, i - 1] + \
            (vk - xx[[1, 3, 5, 11], i - 1]) * time_steps

        # Update the first 12 states based on dynamics
        dynamics = (np.block([[A - B @ K, B @ Kc],
                              [-Bc @ C, Ac]]) @ xx[:, i - 1] +
                    np.vstack([np.zeros((12, 4)), Bc]) @ u)
        xx[:12, i] = xx[:12, i - 1] + dynamics[:12] * time_steps

    print("\nSimulation complete.")
    return xx, controls, time_interval, kappas, vk_values


def plot_vk_values(time_interval, vk_values):
    """
    Plot the evolution of vk components over time.

    Parameters:
        time_interval (np.array): Array of time points.
        vk_values (list): List of vk vectors over time.
    """
    if not vk_values:
        print("No vk_values to plot.")
        return

    vk_values = np.array(vk_values)
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Evolution of $v_k$ Components Over Time")
    for i in range(vk_values.shape[1]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(time_interval[:len(vk_values)], vk_values[:, i], label=f'$v_k[{i}]$',
                color=plt.cm.viridis(i / vk_values.shape[1]))
        ax.set_title(f'$v_k[{i}]$ Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'$v_k[{i}]$ Value')
        ax.legend()
        ax.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_SRG_simulation(time_interval, xx, target_state, kappas):
    """
    Plot the SRG simulation results.

    Parameters:
        time_interval (np.array): Array of time points.
        xx (np.array): State variables over time.
        target_state (list or np.array): Target state vector.
        kappas (list): List of kappa values over time.
    """
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"Reference Governor Flight.\n Target state X: {target_state[1]}, Y: {target_state[3]}, Z: {target_state[5]}"
    )

    # Change in X
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time_interval, xx[1, :], label='X Change')
    ax1.set_title('Change in X')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position')
    ax1.legend()
    ax1.grid()

    # Change in Y
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_interval, xx[3, :], label='Y Change', color='orange')
    ax2.set_title('Change in Y')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid()

    # Change in Z
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time_interval, xx[5, :], label='Z Change', color='green')
    ax3.set_title('Change in Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position')
    ax3.legend()
    ax3.grid()

    # 3D Trajectory
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax4.plot(xx[1, :], xx[3, :], xx[5, :],
             label='3D Trajectory', color='purple')
    ax4.set_title('3D State Trajectory')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('Z Position')
    ax4.legend()

    # Kappa Values over Time
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_interval[:len(kappas)], kappas, label='Kappa', color='red')
    ax5.set_title('Kappa Values Over Time')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Kappa')
    ax5.legend()
    ax5.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_SRG_controls(time_interval, controls, target_state, enable_constraints):
    """
    Plot the SRG control variables with constraints.

    Parameters:
        time_interval (np.array): Array of time points.
        controls (np.array): Control inputs over time.
        target_state (list or np.array): Target state vector.
        enable_constraints (bool): Flag indicating if constraints were enabled.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    constraint_labels = ['Constraint at 6', 'Constraint at 0.005',
                         'Constraint at 0.005', 'Constraint at 0.005']
    fig_title = "SRG Control Variables with Constraints" if enable_constraints else "SRG Control Variables without Constraints"
    fig.suptitle(
        f"{fig_title}\nTarget X = {target_state[1]}, Y = {target_state[3]}, Z = {target_state[5]}")

    for idx in range(4):
        row, col = divmod(idx, 2)
        axs[row, col].plot(time_interval, controls[idx, :],
                           label=f'Control {idx + 1}')
        axs[row, col].axhline(y=6 if idx == 0 else 0.005, color='red', linestyle='--',
                              label=constraint_labels[idx])
        axs[row, col].set_title(f'Control Variable {idx + 1}')
        axs[row, col].set_xlabel('Time (s)')
        axs[row, col].set_ylabel(f'Control {idx + 1} Value')
        axs[row, col].legend()
        axs[row, col].grid()

    plt.tight_layout()
    plt.show()


def compare_simulations(xx_constraints, xx_no_constraints, controls_constraints, controls_no_constraints, time_interval):
    """
    Compare the simulations with and without constraints.

    Parameters:
        xx_constraints (np.array): State variables with constraints.
        xx_no_constraints (np.array): State variables without constraints.
        controls_constraints (np.array): Control inputs with constraints.
        controls_no_constraints (np.array): Control inputs without constraints.
        time_interval (np.array): Array of time points.
    """
    # Extract key metrics
    metrics = {
        "Max Deviation (x)": [
            np.max(np.abs(xx_constraints[1, :])),
            np.max(np.abs(xx_no_constraints[1, :]))
        ],
        "Max Deviation (y)": [
            np.max(np.abs(xx_constraints[3, :])),
            np.max(np.abs(xx_no_constraints[3, :]))
        ],
        "Max Deviation (z)": [
            np.max(np.abs(xx_constraints[5, :])),
            np.max(np.abs(xx_no_constraints[5, :]))
        ],
        "Final x": [
            xx_constraints[1, -1],
            xx_no_constraints[1, -1]
        ],
        "Final y": [
            xx_constraints[3, -1],
            xx_no_constraints[3, -1]
        ],
        "Final z": [
            xx_constraints[5, -1],
            xx_no_constraints[5, -1]
        ],
        "Total Control Effort": [
            np.sum(np.abs(controls_constraints)),
            np.sum(np.abs(controls_no_constraints))
        ]
    }

    # Create a DataFrame for better visualization
    df_comparison = pd.DataFrame(
        metrics, index=["With Constraints", "Without Constraints"])
    print("\nSimulation Comparison:")
    print(df_comparison)

    # Plot differences in state trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(time_interval, xx_constraints[1, :], label='x (With Constraints)')
    plt.plot(time_interval,
             xx_no_constraints[1, :], '--', label='x (Without Constraints)')
    plt.plot(time_interval, xx_constraints[3, :], label='y (With Constraints)')
    plt.plot(time_interval,
             xx_no_constraints[3, :], '--', label='y (Without Constraints)')
    plt.plot(time_interval, xx_constraints[5, :], label='z (With Constraints)')
    plt.plot(time_interval,
             xx_no_constraints[5, :], '--', label='z (Without Constraints)')
    plt.title("State Trajectories Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("State Values")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot control effort comparison
    plt.figure(figsize=(10, 6))
    control_effort_with = np.sum(np.abs(controls_constraints), axis=0)
    control_effort_without = np.sum(np.abs(controls_no_constraints), axis=0)
    plt.plot(time_interval, control_effort_with,
             label='Control Effort (With Constraints)')
    plt.plot(time_interval, control_effort_without, '--',
             label='Control Effort (Without Constraints)')
    plt.title("Control Effort Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Effort")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    """
    Main function to run simulations with and without constraints and compare the results.
    """
    print("Main started")

    # Define target state (16-dimensional)
    target_state_16 = [0, 4, 0, 5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Define figure-8 waypoints (currently unused)
    inputarrfigure8 = [[0, 0, 1, 0], [1, 0, 1, 0]]

    # Simulation with constraints enabled
    print("\nRunning simulation with constraints enabled...")
    xx_constraints, controls_constraints, time_interval_constraints, kappas_constraints, vk_values_constraints = SRG_Simulation_Linear(
        desired_state=target_state_16,
        figure8waypoints=inputarrfigure8,
        enable_constraints=True,
        enable_modified_target=True
    )

    # Simulation with constraints disabled
    print("\nRunning simulation with constraints disabled...")
    xx_no_constraints, controls_no_constraints, time_interval_no_constraints, kappas_no_constraints, vk_values_no_constraints = SRG_Simulation_Linear(
        desired_state=target_state_16,
        figure8waypoints=inputarrfigure8,
        enable_constraints=False,
        enable_modified_target=True  # Keeping modified target consistent
    )

    # Compare the results
    compare_simulations(
        xx_constraints, xx_no_constraints,
        controls_constraints, controls_no_constraints,
        time_interval_constraints
    )

    # Plot SRG simulations
    plot_SRG_simulation(time_interval_constraints,
                        xx_constraints, target_state_16, kappas_constraints)
    plot_SRG_simulation(time_interval_no_constraints,
                        xx_no_constraints, target_state_16, kappas_no_constraints)

    # Plot SRG controls
    plot_SRG_controls(time_interval_constraints, controls_constraints,
                      target_state_16, enable_constraints=True)
    plot_SRG_controls(time_interval_no_constraints, controls_no_constraints,
                      target_state_16, enable_constraints=False)

    # Plot vk values for the simulation with constraints
    plot_vk_values(time_interval_constraints, vk_values_constraints)
    # Plot vk values for the simulation without constraints
    plot_vk_values(time_interval_no_constraints, vk_values_no_constraints)

    print("Done")


if __name__ == '__main__':
    main()
