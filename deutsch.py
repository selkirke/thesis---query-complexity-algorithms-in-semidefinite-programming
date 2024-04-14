import cvxpy as cp
import numpy as np

def generate_rho_target_deutsch(n_qubits, is_balanced):

    if is_balanced:
        #desired full state including the ancilla is |1-⟩
        state_vector = np.array([0, 0, 1/np.sqrt(2), -1/np.sqrt(2)])
    else:
        #desired full state including the ancilla is |0-⟩
        state_vector = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0, 0])

    #returning desired state
    rho_target = np.outer(state_vector, state_vector)
    return rho_target


def P(f_truth_table, delta,epsilon):
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

     #Hadamard operation for 2 qubits as a matrix
     H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
     H_total = np.kron(H, H)  #tensor product to get 4x4 Hadamard
     #print(H_total)
     gamma = epsilon - 3 * epsilon ** 2 + 2 * epsilon ** 3
     I = np.eye(2)
     F_H = np.kron(H,I)
     delta_opt = 0
     #oracle operation
     O_f_value = np.diag([1, 1, -1, -1]) if is_balanced else np.eye(n)
     #if is_balanced:
     #     print('balanced')
     
     #constraints for the SDP
     constraints = []
     constraints = [M[j] >> 0 for j in range(n)]  #M -> PSD
     constraints+= [cp.trace(M[j]) == 1 for j in range(n)]
     constraints.append(delta >= 0),  #lower bound constraint 
     constraints.append(delta <= delta_opt+gamma)  #upper bound constraint 
     for j in range(n):
          constraints.append(M[j] == M[j].T)
     #initial state constraint for a 2-qubit system: 
     #the input qubit |0>, and the ancilla qubit |1>, represented as |01>
     constraints.append(M[0] == cp.outer([0, 1, 0, 0], [0, 1, 0, 0])) 
     
     #Hadamard constraint 1
     constraints.append(M[1] == H_total @ M[0] @ H_total.T)

     #Oracle constraint
     constraints.append(M[2] == O_f_value @ M[1] @ O_f_value.T)

     #Final Hadamard constraint
     constraints.append(M[3] == F_H @ M[2] @ F_H.T)
     #objective function to minimize delta within constraints
     #this can be changed depending on desired objective
     rho_target = generate_rho_target_deutsch(n_qubits, is_balanced)
     M_3 = M[3]
     objective = cp.Minimize(delta) 
     #Create the sdp problem and solve it based on objective and constraints
     problem = cp.Problem(objective, constraints)
     problem.solve() 
     
     #print statements
     print(f"status: {problem.status}")
     M_final = np.round(M[3].value,3)
     M_initial = np.round(M[0].value)
     M_1 =  np.round(M[1].value,3)
     M_2 =  np.round(M[2].value,3)
     print("initial State Gram Matrix Values:")
     print(M_initial)
     print("State Gram Matrix Values after Hadamard:")
     print(M_1)
     print("State Gram Matrix Values after Oracle:")
     print(M_2)
     print("Final State Gram Matrix Values:")
     print(M_final)
     print(f"Feasible with delta={delta.value:.4f} within epsilon={epsilon}")
     return problem.status, delta.value,M_final if problem.status == cp.OPTIMAL else None

def estimate_QQC_epsilon(f_truth_table, epsilon):
#generalised for any sized input truth table for function f

    n = len(next(iter(f_truth_table)))  #no. of input bits
    T_star = n  #initialize with the maximum possible number of steps

    #iterate over all possible t values to find the smallest t for which P(f, t, delta_t) is feasible 
    #(for deustch this should end at t = 1)
    for t in range(1, n + 1): 
     print()
     print(f"for query complexity estimation = {t}: ")
     #define the variable representing delta_t for the current value of t
     delta = cp.Variable()
     
     #Check feasibility of P(f, t, delta_t)
     status, opt_value,M_final = P(f_truth_table, delta, epsilon)
     #print(M_final)
     if status == cp.OPTIMAL and opt_value is not None and opt_value <= epsilon:
          T_star = t

          #condition to check if the function is constant or balanced
          if M_final[0, 0] + M_final[2, 2] > M_final[1, 1] + M_final[3, 3]:
               print(f"The algorithm suggests the function is constant with {t} queries to the oracle.")
          else:
               print(f"The algorithm suggests the function is balanced with {t} queries to the oracle.")

     break  #stop if feasible solution found

     return T_star

if __name__ == "__main__":
     t = 1
     epsilon = 0.5
     f_truth_table = {'0': 1, '1': 0}
     estimate_QQC_epsilon(f_truth_table, epsilon)
