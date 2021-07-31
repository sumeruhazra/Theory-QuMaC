# -*- coding: utf-8 -*-
"""
Created on Tue May 19 04:15:43 2020
@author: Sumeru
"""
#=========================================================================================
from qutip import *
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import fractional_matrix_power
no_of_qubits = 2 
#=========================================================================================
def binomial(n, k): #calculates binomial coefficients
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
#=========================================================================================
def twoqubit_h(dim, w1, w2, del1, del2, g12):
    #this function defines the two qubit hamiltonian with delta being anharmonicities
    #the exchange coupling is given by g12
    a1 = tensor(destroy(dim), qeye(dim))
    a1dag = tensor(create(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim))
    a2dag = tensor(qeye(dim), create(dim))
    H1 = w1*(a1dag*a1) + (del1/2)*((a1dag*a1)*(a1dag*a1)-(a1dag*a1))
    H2 =  w2*(a2dag*a2) + (del2/2)*((a2dag*a2)*(a2dag*a2)-(a2dag*a2))
    Hc = g12*(a1dag*a2 + a1*a2dag)
    Htot = H1 + H2 + Hc
    return Htot
#=========================================================================================
# The ordering of states in qutip is in number_base(n) format as in 
# {00, 01, 02, ..., 0d, 10, 11, ..., 1d, ..., d0, ..., dd}
# We shall first truncate the Hamiltonian to limit the maximum number of excitations
# Next we rearrange the Hamiltonian in the binomial pyramid format given below:
# {00, 01, 10, 02, 11, 20 rest}
#=========================================================================================    
def red_h(dim, w1, w2, del1, del2, g12):
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    y = np.zeros(d_hs, dtype =int)
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            y[l] = dim*j + (i - j)
            l = l+1
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    base_ham = twoqubit_h(dim, w1, w2, del1, del2, g12)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):
            empty2d[i1, j1] = base_ham[y[i1], y[j1]]
    return empty2d
#=========================================================================================
"Dressed Hamiltonian" 
#=========================================================================================
def U(dim, w1, w2, del1, del2, g12):
    val, vec = np.linalg.eig(red_h(dim, w1, w2, del1, del2, g12))
    vec_arr = np.argmax(vec, axis = 0) #eigenvectors along columns
    val_sor_ind = np.argsort(vec_arr)
    unitary = vec[:,val_sor_ind]
    return unitary
#=========================================================================================
#Dressed (diagonal Hamiltonian)
def H_diag(dim, w1, w2, del1, del2, g12):
    u_op = U(dim, w1, w2, del1, del2, g12)
    u_op_dag = np.asmatrix(u_op).getH()
    matrix1 = np.matmul(u_op_dag, red_h(dim, w1, w2, del1, del2, g12))
    dressed_ham = np.matmul(matrix1, u_op)
    return dressed_ham
#=========================================================================================
"Writing the drive terms"
#=========================================================================================
#Drive operator
def x_op(dim, qubit_index):#qubit index 1 or 2    
    x1 = tensor(destroy(dim), qeye(dim)) + tensor(create(dim), qeye(dim))
    x2 = tensor(qeye(dim), destroy(dim)) + tensor(qeye(dim), create(dim))
    if qubit_index==2:
        x1 = x2
    return x1
#=========================================================================================
#Reduced drive matrix
def red_drive(dim, qubit_index):
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    y = np.zeros(d_hs, dtype =int)
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            y[l] = dim*j + (i - j)
            l = l+1
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    base_drive = x_op(dim, qubit_index)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):
            empty2d[i1, j1] = base_drive[y[i1], y[j1]]
    return empty2d
#=========================================================================================
#Drive in the dressed frame
def dressed_x_op(dim, w1, w2, del1, del2, g12, qubit_index):
    u_op = U(dim, w1, w2, del1, del2, g12)
    u_op_dag = np.asmatrix(u_op).getH()
    matrix1 = np.matmul(u_op_dag, red_drive(dim, qubit_index))
    dressed_drive = np.matmul(matrix1, u_op)
    return dressed_drive
#=========================================================================================       
"Rotating Wave Approximation (RWA)"        
#=========================================================================================
#Define a RWA hamiltonian HA that rotates both the qubits in the drive frame
def HA(dim, wd):
    a1 = tensor(destroy(dim), qeye(dim))
    a1dag = tensor(create(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim))
    a2dag = tensor(qeye(dim), create(dim))
    rotating_ham =  wd*((a1dag*a1) + (a2dag*a2))
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    y = np.zeros(d_hs, dtype =int)
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            y[l] = dim*j + (i - j)
            l = l+1
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):
            empty2d[i1, j1] = rotating_ham[y[i1], y[j1]]
    return empty2d
#=========================================================================================
#Write the full drive hamiltonian in the RWA
def drive_rwa(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2):
    dressed_drive_1 = dressed_x_op(dim, w1, w2, del1, del2, g12, 1)
    dressed_drive_2 = dressed_x_op(dim, w1, w2, del1, del2, g12, 2)
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):            
            if i1>j1:
                empty2d[i1, j1] = ((amp1/2)*np.exp(1.j*phase1)*dressed_drive_1[i1, j1] 
                + (amp2/2)*np.exp(1.j*phase2)*dressed_drive_2[i1, j1])
            else:
                empty2d[i1, j1] = ((amp1/2)*np.exp(-1.j*phase1)*dressed_drive_1[i1, j1]
                + (amp2/2)*np.exp(-1.j*phase2)*dressed_drive_2[i1, j1])
                
    return empty2d
#=========================================================================================
#Write down the total Hamiltonian under RWA
def tot_H_rwa(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd):
    H_tot = (H_diag(dim, w1, w2, del1, del2, g12) - HA(dim, wd) 
    + drive_rwa(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2))
    #change of order in the basis in the form (00, 01, 10, 11, rest)
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    permutation = np.identity(d_hs)
    permutation[3,4]=1
    permutation[4,3]=1
    permutation[3,3]=0
    permutation[4,4]=0
    per_transpose = np.asmatrix(permutation).getT()
    matrix_1 = np.matmul(per_transpose, H_tot)
    matrix_2 = np.matmul(matrix_1, permutation)
    return matrix_2
"End of RWA"
#This matrix tot_H_rwa captures the whole dynamics of the system, in the next section we
#will reduce it to an effective hamiltonian which will get rid of the fast oscillations
#=========================================================================================    
"Block diagonalization using principle of least action"
#=========================================================================================
#reference:  L S Cederbaum et al 1989 J. Phys. A: Math. Gen. 22 2427
def S_mat(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd):#eigenvector_mat
    rwa_hamiltonian = np.asarray(tot_H_rwa(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd))
    val, vec = np.linalg.eig(rwa_hamiltonian)
    vec_arr = np.argmax(vec, axis = 0) #eigenvectors along columns
    val_sor_ind = np.argsort(vec_arr)
    eigenvector_matrix = vec[:,val_sor_ind]
    return eigenvector_matrix
#=========================================================================================
#Find the block diagonal form of S in the (2*2)+(2*2)+rest blocks
def S_bd(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd):
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    #initialize the blocks: block 1, 2 and 3
    block1 = np.zeros([2, 2], dtype = complex)
    block2 = np.zeros([2, 2], dtype = complex)
    block3 = np.zeros([d_hs-4, d_hs-4], dtype = complex)
    S = S_mat(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd)
    for i1 in range(0, 2):
        for j1 in range(0, 2):
            block1[i1, j1] = S[i1, j1]
    for i1 in range(0, 2):
        for j1 in range(0, 2):
            block2[i1, j1] = S[i1+2, j1+2]
    for i1 in range(0, d_hs-4):
        for j1 in range(0, d_hs-4):
            block3[i1, j1] = S[i1+4, j1+4]
    bd_matrix = np.asarray(block_diag(block1, block2, block3))
    return bd_matrix
#=========================================================================================
def T_mat(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd):
    S_blkdg = np.asmatrix(S_bd(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd))
    S_blkdg_dag = S_blkdg.getH()
    S = np.asmatrix(S_mat(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd))
    matrix_1 = np.matmul(S, S_blkdg_dag)
    matrix_2 = np.matmul(S_blkdg, S_blkdg_dag)
    matrix_3 = fractional_matrix_power(matrix_2, -0.5)
    t_matrix = np.matmul(matrix_1, matrix_3)
    return t_matrix
#=========================================================================================
"Write down the effective hamiltonian with the transformation matrix T-mat"
def eff_H(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd):
    H_rwa_0 = tot_H_rwa(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd)
    t_matrix = T_mat(dim, w1, w2, del1, del2, g12, amp1, phase1, amp2, phase2, wd)
    t_matrix_dag = t_matrix.getH()
    matrix_1 = np.matmul(t_matrix_dag, H_rwa_0)
    eff_bd_ham = np.matmul(matrix_1, t_matrix)
    q_space = np.zeros([4, 4], dtype = complex)
    for i1 in range(0, 4):
        for j1 in range(0, 4):
            q_space[i1, j1] = np.round(eff_bd_ham[i1, j1], 6)
    return q_space
#=========================================================================================

