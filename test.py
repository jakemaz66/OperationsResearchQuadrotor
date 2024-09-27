# def linear_dynamics_integral(t, curstate_integral, A, B, target_state):
#     """Modeling the linear dynamics of the quadrotor with integral control. The linear dynamics tell you how
#        the state of the quadrotor changes with regards to two things
#        1. The current state of the quadrotor
#        2. The input controls into the quadrotor

#     Args:
#         curstate (array): A vector of the current state of the quadrotor, 1X16
#         A (nd-array): A given matrix for state linear dynamics (given in slides)
#         B (nd-array): A given matrix for input linear dynamics (given in slides)
#         target_state (array): A vector of the target state of the quadrotor

#     Returns:
#         array: An array defining the change in the state for each dimension of the quadrotor
#     """

#     u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r, i1, i2, i3, i4 = curstate_integral

#     #Putting together 4x1 error vector
#     errorX = target_state[0] - curstate_integral[0]
#     errorY = target_state[1] - curstate_integral[1]
#     errorZ = target_state[2] - curstate_integral[2]
#     errorPsi = target_state[8] - curstate_integral[8]

#     error = np.array([errorX, errorY, errorZ, errorPsi]).reshape(4, 1)
    

#     Ac = np.array([
#         [0, 0, 0, 0,],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0]
#     ])

#     Bc = np.array([
#         [1, 0, 0, 0,],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])

#     # Define C matrix as shown in the image (4x12 matrix)
#     C = np.array([
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     ])

#     #Get A and B hats
#     A_hat = np.block([
#         [A, np.zeros_like(B)],    
#         [-Bc @ C, Ac]             
#     ])

#     B_hat = np.block([
#         [B],
#         [Bc]
#     ])

#     x = curstate_integral[0:12] 
#     xc = curstate_integral[12:16]

#     # Compute the control
#     control = -K @ x + Kc @ xc

#     # Compute Dotx and dxc
#     Dotx = A @ x + B @ control 
#     # Integrator
#     dxc = xc + error 
#     # Return concatenated result return 
#     np.concatenate((Dotx, dxc))






# def linear_dynamics_integral(t, curstate_integral, A, B, target_state):
#     """Modeling the linear dynamics of the quadrotor with integral control. The linear dynamics tell you how
#        the state of the quadrotor changes with regards to two things
#        1. The current state of the quadrotor
#        2. The input controls into the quadrotor

#     Args:
#         curstate (array): A vector of the current state of the quadrotor, 1X16
#         A (nd-array): A given matrix for state linear dynamics (given in slides)
#         B (nd-array): A given matrix for input linear dynamics (given in slides)
#         target_state (array): A vector of the target state of the quadrotor

#     Returns:
#         array: An array defining the change in the state for each dimension of the quadrotor
#     """

#     u, Px, v, Py, w, Pz , phi, p, theta, q, psi, r, i1, i2, i3, i4 = curstate_integral

#     # Initialize an empty list to store the errors
#     errors = []

#     # Iterate exactly 12 times (assuming target_state and curstate have at least 12 elements)
#     for i in range(16):
#         target = target_state[i]
#         state = curstate_integral[i]

#         # Compute the difference between target and state
#         difference = state - target
#         #print(target, " ", state)

#         # Append the result to the error list
#         errors.append(difference)

#     #Putting together 4x1 error vector
#     errorX = target_state[1] - curstate_integral[1]
#     errorY = target_state[3] - curstate_integral[3]
#     errorZ = target_state[5] - curstate_integral[5]
#     errorPsi = target_state[11] - curstate_integral[11]

#     error = np.array([errorX, errorY, errorZ, errorPsi]).reshape(4, 1)

#     x = errors[0:12] 
#     xc = errors[12:16]

#     # Compute the control
#     control = -K @ x + Kc @ xc

#     #Compute Dotx and dxc
#     Dotx = A @ x + B @ control 
#     #Integrator
#     dxc = xc + error.T[0]
#     #Return concatenated result return 
#     return np.concatenate((Dotx, dxc))
