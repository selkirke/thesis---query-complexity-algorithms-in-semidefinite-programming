import cvxpy as cp
import numpy as np
from scipy.linalg import expm,sqrtm
import time
import matplotlib.pyplot as plt



def prepare_initial_state(noise_strength):
    p = 1 
    ideal_initial_state = p * np.outer([0, 1, 0, 0], [0, 1, 0, 0]) + (1 - p) * np.outer([0, 0, 1, 0], [0, 0, 1, 0])
    
    #define noisy state (completely mixed state)
    noise_state = np.eye(4) / 4  # Example: A 4x4 Identity matrix scaled to have trace 1
    
    #create noisy initial state using convex combination
    noisy_initial_state = (1 - noise_strength) * ideal_initial_state + noise_strength * noise_state
    
    return noisy_initial_state

#Hadamard with rotation error
def apply_hadamard_noise(base_matrix, noise_strength):
    #print(noise_strength)
    if noise_strength == 0:
        return base_matrix  #unaltered base matrix if no noise

    #calculate current theta from the base_matrix, assuming it is a Hadamard matrix
    theta = np.pi/ 4  #for Hadamard, theta is pi/4

    #apply small deviation from theta as noise
    delta_theta = noise_strength * (np.random.rand() - 0.5) * 2  # Random value between -noise_strength and +noise_strength
    noisy_theta = theta + delta_theta  # New theta after noise

    # Define the noisy Hadamard operation
    H_noisy = np.array([[np.cos(noisy_theta), np.sin(noisy_theta)],
                        [np.sin(noisy_theta), -np.cos(noisy_theta)]])
    return H_noisy

def P(f_truth_table,delta_opt,noise_level):
    noise_strength_preparation = 0.1*noise_level
    noise_strength_hadamard = 0.1*noise_level
    #number of qubits (1 input qubit + 1 ancilla qubit)
    n_qubits = 2
    n = 2 ** n_qubits  #dimensionality of matrix
    num_outputs = len(set(f_truth_table.values()))
    is_balanced = num_outputs > 1
    #define state Gram matrices for each step of the algorithm as cp variables 
    M = {0: cp.Variable((n, n), PSD=True),  #Initial state Gram matrix
        1: cp.Variable((n, n), PSD=True),  #After Hadamard gate
        2: cp.Variable((n, n), PSD=True),  #After Oracle
        3: cp.Variable((n, n), PSD=True)}  #After final Hadamard gate
    
    initial_state = prepare_initial_state(noise_strength_preparation)
    
    #Hadamard operation as a unitary matrix
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_total = np.kron(H, H)  #tensor product to get 4x4 Hadamard 
    H_noisy = np.kron(apply_hadamard_noise(H, noise_strength_hadamard), #noisy hadamard
                  apply_hadamard_noise(H, noise_strength_hadamard))
    
    O_f_value = np.diag([1, 1, -1, -1]) if is_balanced else np.eye(n)

    I = np.eye(n_qubits)
    F_H = np.kron(H,I)
    F_H_noisy = np.kron(apply_hadamard_noise(H, noise_strength_hadamard),I)
    #print(F_H_noisy)
    epsilon_lower_bound = 0.0
    epsilon_upper_bound = 0.5
    
    gamma = noise_level - 3 * noise_level ** 2 + 2 * noise_level ** 3
    noise = cp.Variable()  #slack variable representing allowable deviation for error
    delta = cp.Variable() 
    
    #Constraints for SDP
    constraints = []
    constraints = [M[j] >> 0 for j in range(n)]  # M -> PSD
    constraints+= [cp.trace(M[j]) == 1 for j in range(n)]
    for j in range(n): 
        constraints.append(M[j] == M[j].T) #M -> Hermitian
    constraints.append(delta >= delta_opt),  #Lower bound constraint (0 case = delta_opt)
    constraints.append(delta <= delta_opt + gamma)  #Upper bound constraint 
    constraints.append(noise >= epsilon_lower_bound) #lower bound for noise constraint
    constraints.append(noise <= epsilon_upper_bound) #upper bound for noise constraint (subject to change)
    
    #to choose what elements experience noise, remove the noisy constraint from comments and put the normal in comments
    
    #normal initial state constraint 
    constraints.append(M[0] == cp.outer([0, 1, 0, 0], [0, 1, 0, 0])) 
    #noisy initial state constraint 
    #constraints.append(M[0] == initial_state)

    #Hadamard constraint 1 - normal
    constraints.append(M[1] == H_total @ M[0] @ H_total.T)
    #Hadamard constraint 1 - noisy
    #constraints.append(M[1] == H_noisy @ M[0] @ H_noisy.T)

    #Oracle constraint
    constraints.append(M[2] == O_f_value @ M[1] @ O_f_value.T)
    
    #Final Hadamard constraint - normal
    constraints.append(M[3] == F_H @ M[2] @ F_H.T)
    #Final Hadamard constraint - noisy
    #constraints.append(M[3] == F_H_noisy @ M[2] @ F_H_noisy.T)
    
    rho_target = generate_rho_target_deutsch(n_qubits, is_balanced)
    M_3 = M[3]  
    #minimize trace norm distance between M_3 and rho_target
    objective = cp.Minimize(cp.norm(M_3 - rho_target, 'nuc')) 
    #Create the sdp problem and solve it based on objective and constraints
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    print(problem.value)
    if problem.status in [cp.OPTIMAL,cp.OPTIMAL_INACCURATE]: #solution found
        return problem.value
    else:
        return np.nan  #np.nan for infeasible problems

def generate_rho_target_deutsch(n_qubits, is_balanced):
    
    if is_balanced:
        #desired full state including the ancilla is |1-⟩
        state_vector = np.array([0, 0, 1/np.sqrt(2), -1/np.sqrt(2)])
    else:
        #desired full state including the ancilla is |0-⟩
        state_vector = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0, 0])

    rho_target = np.outer(state_vector, state_vector)
    return rho_target


def noise_plot(f_truth_table,delta_opt):
    noise_levels = np.linspace(0.00, 0.6, 100) #can be varied for intended noise level range
    objective_values = []
    for noise_level in noise_levels:
        objective_value = P(f_truth_table,delta_opt,noise_level)
        objective_values.append(objective_value)
    print(objective_values)
    
    #filtering out infeasible values
    filtered_noise_levels = [noise_levels[i] for i, val in enumerate(objective_values) if not np.isnan(val)]
    filtered_objective_values = [val for val in objective_values if not np.isnan(val)]

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_noise_levels, filtered_objective_values, marker='o', color='b',label = 'Feasible')
    plt.xlabel('Noise Level')
    plt.ylabel('Trace Norm Distance')
    plt.title('Objective Function vs. Noise Level for Initial State (Feasible Points)')
    plt.grid(True)
    z = np.polyfit(filtered_noise_levels, filtered_objective_values, 1)
    p = np.poly1d(z)
    plt.plot(filtered_noise_levels, p(filtered_noise_levels), "r--", label='Trend Line')
    slope = z[0]

    #calculate angle of trend line
    angle = np.arctan(slope) * 180 / np.pi
    slope = (p(filtered_noise_levels[-1]) - p(filtered_noise_levels[0])) / (filtered_noise_levels[-1] - filtered_noise_levels[0])
    plt.annotate(f'Trendline Angle = {angle:.2f}°', xy=(0.5, 0.8), xycoords='axes fraction', 
             xytext=(0, -10), textcoords='offset points', ha='center', fontsize=12)
    #annotate the angle
    plt.ylim(0, 0.25)
    infeasible_noise_levels = [noise_levels[i] for i, val in enumerate(objective_values) if np.isnan(val)]
    plt.scatter(infeasible_noise_levels, [min(filtered_objective_values)]*len(infeasible_noise_levels), color='r', label='Infeasible')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    
    #vary for intended function truth table
    f_truth_table = {'0': 1, '1': 1}
    delta_opt = 0.0 #calculated for deutsch
    noise_plot(f_truth_table,delta_opt)


