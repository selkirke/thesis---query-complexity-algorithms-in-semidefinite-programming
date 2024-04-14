import cvxpy as cp
import numpy as np
from scipy.linalg import expm,sqrtm
import matplotlib.pyplot as plt

def generate_truth_table(size, balanced=True):
    """Generates a truth table for a given size that is either balanced or constant."""
    if balanced:
        half_size = 2**size // 2
        outputs = [1] * half_size + [0] * half_size
    else:
        outputs = [1] * (2**size)
    #shuffle outputs to randomize the truth table
    np.random.shuffle(outputs)
    #generate binary input strings and pair with outputs
    inputs = [format(i, f'0{size}b') for i in range(2**size)]
    return dict(zip(inputs, outputs))

def NbitHadamard(n):
    #Hadamard operation for 2 qubits as a matrix
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_n = H
    for _ in range(n-1):
        H_n = np.kron(H_n, H)
    return H_n

def NbitFinalHadamard(n):
    #Hadamard operation for 2 qubits as a matrix
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_n = H
    for _ in range(n-2):
        H_n = np.kron(H, H_n)
    return H_n

def P(f_truth_table, delta):
    epsilon = 0.5
    #number of qubits (1 input qubit + 1 ancilla qubit)
    n_qubits = int(np.log2(len(f_truth_table)))+1
    n = 2 ** n_qubits  # Dimensionality of the matrix
    num_outputs = len(set(f_truth_table.values()))
    is_balanced = num_outputs > 1
    if is_balanced:
        print("balanced")
    #define state Gram matrices for each step of the algorithm as cp variables 
    M = {0: cp.Variable((n, n), PSD=True),  #Initial state Gram matrix
        1: cp.Variable((n, n), PSD=True),  #After Hadamard gate
        2: cp.Variable((n, n), PSD=True),  #After Oracle
        3: cp.Variable((n, n), PSD=True)}  #After final Hadamard gate
    
    #initial state
    initial_state_vector = np.zeros(n)
    initial_state_vector[1] = 1  # Adjust this index based on the desired initial state
    
    #Hadamard
    H_total = NbitHadamard(n_qubits)
    
    #Oracle
    O_f = np.diag([1 if i < n/2 else -1 for i in range(n)]) if is_balanced else np.diag([1] * n) 

    #Final Hadamard
    F_H = np.kron(NbitFinalHadamard(n_qubits),np.eye(2))
    
    gamma = epsilon - 3 * epsilon ** 2 + 2 * epsilon ** 3
    delta_opt = 0
    #Constraints for SDP (gram matrix progression/optimization)
    constraints = []
    constraints = [M[j] >> 0 for j in range(4)]  # M -> PSD
    constraints+= [cp.trace(M[j]) == 1 for j in range(4)]
    constraints.append(delta >= delta_opt),  #Lower bound constraint (0 case = delta_opt)
    constraints.append(delta <= delta_opt + gamma)  #Upper bound constraint (problematic: when removed delta is below threshold)
    for j in range(4):
        constraints.append(M[j] == M[j].T)
        
    #initial state constraint
    constraints.append(M[0] == cp.outer(initial_state_vector,initial_state_vector))
    #Hadamard constraint 1
    constraints.append(M[1] == H_total @ M[0] @ H_total.T)
    #Oracle constraint
    constraints.append(M[2] == O_f @ M[1] @ O_f.T)
    #Final Hadamard constraint
    constraints.append(M[3] == F_H @ M[2] @ F_H.T)
    
    objective = cp.Minimize(delta)
    #Create the sdp problem and solve it based on objective and constraints
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    print(f"Status: {problem.status}")
    M_final = np.round(M[3].value,3)
    #target output for balanced case
    #print(f"Final State Gram Matrix Values: \n{M_final}")
    #print(f"Desired:{rho_kron}")
    #print(f"Status: {problem.status}")
    print(f"Feasible with delta={delta.value:.4f} within epsilon={epsilon}")
    return problem.status, delta.value,M_final if problem.status == cp.OPTIMAL else None

def generate_rho_target(n_qubits, is_balanced):
    # For a balanced function, the state is not |0...0-⟩.
    if is_balanced:
        #choose arbitrary output state that isnt |0...0-⟩
        state_vector = np.zeros(2**n_qubits)
        state_vector[-2] = 1 / np.sqrt(2) 
        state_vector[-1] = -1 / np.sqrt(2) 
        
    # For a constant function, the state is |0...0-⟩. choosing random state for constant case
    else:
        state_vector = np.zeros(2**n_qubits)
        state_vector[0] = 1 / np.sqrt(2)  # |0...00⟩ component
        state_vector[1] = -1 / np.sqrt(2) # |0...01⟩ component (ancilla bit is |1⟩ for |-⟩ state)
    
    # Calculate the density matrix.
    rho = np.outer(state_vector, state_vector.conj())
    return rho

def estimate_QQC_epsilon(f_truth_table, epsilon):
    n = len(next(iter(f_truth_table)))  #no. of input bits
    #iterate over all possible t values to find the smallest t for which P(f, t, delta_t) is feasible 
    #(for deutsch-jozsa this should end at t = 1 for all n)
    for t in range(1, n + 1): 
        #define the variable representing delta_t for the current value of t
        delta = cp.Variable()
        
        #Check feasibility of P(f, t, delta_t)
        status, opt_value,M_final = P(f_truth_table, delta)
        if status == cp.OPTIMAL and opt_value is not None and opt_value <= epsilon:
          T_star = t
          #print(M_final)
          if M_final[0, 0] + M_final[1, 1] > M_final[2, 2] + M_final[3, 3]:
               print(f"The algorithm suggests the function is constant with {t} queries to the oracle.")
          else:
               print(f"The algorithm suggests the function is balanced with {t} queries to the oracle.")

        break  #stop if feasible solution found
    return 

if __name__ == "__main__":
    epsilon = 0.5
    sizes = range(1, 6)  #sample range, adjust based on what truth table sizes you wish to test
    times = []
    for size in sizes:
        #rotate between constant and balanced cases
        if size%2 == 0:
            x = generate_truth_table(size, balanced=False)
            print(x)
            #print(f"truth table {x}")
            estimate_QQC_epsilon(x, epsilon)
        else:
            x = generate_truth_table(size, balanced=True)
            print(x)
            #print(f"truth table {x}")
            estimate_QQC_epsilon(x, epsilon)
    
