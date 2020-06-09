# Computing the ZZ error of two transversely coupled qubits
#given frequencies, bare coupling and anharmonicities
"""
Created on Tue May 19 04:15:43 2020

@author: Sumeru
"""
#=========================================================================================
from qutip import *
import numpy as np
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
"Next task is to diagonalize the Hamiltonian"
"Using Q.eigenenergies function in QuTiP or np.linalg does not arranges the values properly"
"we would like to keep the same order as the original basis"
#=========================================================================================
def zz_khz(dim, w1, w2, del1, del2, g12):
    val, vec = np.linalg.eig(red_h(dim, w1, w2, del1, del2, g12))
    vec_arr = np.argmax(vec, axis = 0) #eigenvectors along columns
    val_sor_ind = np.argsort(vec_arr)
    val_incr = val[val_sor_ind]
    zz_split_tot = val_incr[4] - val_incr[2] - val_incr[1] + val_incr[0]
    zz_err = zz_split_tot/2
    return np.real(zz_err*1e6)
#=========================================================================================

   
    