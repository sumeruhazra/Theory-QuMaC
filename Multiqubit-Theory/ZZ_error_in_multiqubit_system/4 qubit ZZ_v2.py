# Computing the ZZ errors of four transversely coupled qubits
#given frequencies, bare coupling and anharmonicities
"""
Created on Tue May 21 02:15:43 2020

@author: Sumeru
"""
#=====================================================================================================
from qutip import *
import numpy as np
no_of_qubits = 4
#=====================================================================================================
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
#=====================================================================================================
def fourqubit_h(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34):
    #this function defines the two qubit hamiltonian with delta being anharmonicities
    #the exchange coupling is given by g12
    a1 = tensor(destroy(dim), qeye(dim), qeye(dim), qeye(dim))
    a1dag = tensor(create(dim), qeye(dim), qeye(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim), qeye(dim), qeye(dim))
    a2dag = tensor(qeye(dim), create(dim), qeye(dim), qeye(dim))
    a3 = tensor(qeye(dim), qeye(dim), destroy(dim), qeye(dim))
    a3dag = tensor(qeye(dim), qeye(dim), create(dim), qeye(dim))
    a4 = tensor(qeye(dim), qeye(dim), qeye(dim), destroy(dim))
    a4dag = tensor(qeye(dim), qeye(dim), qeye(dim), create(dim))
    H1 = w1*(a1dag*a1) + (del1/2)*((a1dag*a1)*(a1dag*a1)-(a1dag*a1))
    H2 =  w2*(a2dag*a2) + (del2/2)*((a2dag*a2)*(a2dag*a2)-(a2dag*a2))
    H3 =  w3*(a3dag*a3) + (del3/2)*((a3dag*a3)*(a3dag*a3)-(a3dag*a3))
    H4 =  w4*(a4dag*a4) + (del4/2)*((a4dag*a4)*(a4dag*a4)-(a4dag*a4))
    Hc1 = g12*(a1dag*a2 + a1*a2dag) + g13*(a1dag*a3 + a1*a3dag) + g14*(a1dag*a4 + a1*a4dag)
    Hc2 = g23*(a2dag*a3 + a2*a3dag) + g24*(a2dag*a4 + a2*a4dag)
    Hc3 = g34*(a3dag*a4 + a3*a4dag)
    Htot = H1 + H2 + H3 + H4 + Hc1 + Hc2 + Hc3
    return Htot
#=====================================================================================================
# The ordering of states in qutip is in number_base(n) format as in 
# {0000, 0001, 0002, ..., 000d, 0010, 0011, ..., 001d, ..., 00d0, ..., 00dd, ..., 0ddd, ..., dddd}
# We shall first truncate the Hamiltonian to limit the maximum number of excitations
# Next we rearrange the Hamiltonian in the binomial pyramid format given below:
# {0(0(00)), 0(0(01)), 0(0(10)), 0(1(00)), 1(0(00)), 0(0(02)), 0(0(11)), 0(0(20)), 0(1(01)),
# 0(1(10)), 0(2(00)), 1(0(01)), 1(0(10)), 1(1(00)), 2(0(00)), ..., 0(0(0d)), ..., d(0(00))}
#=====================================================================================================   
def red_h(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34):
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    y = np.zeros(d_hs, dtype =int)
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            for k in range(0, i-j+1):
                for m in range(0, i-j-k+1):
                    y[l] = (dim**3)*j + (dim**2)*k + dim*m + (i - j - k - m)                    
                    l = l+1
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    base_ham = fourqubit_h(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):
            empty2d[i1, j1] = base_ham[y[i1], y[j1]]
    return empty2d
#=====================================================================================================
"Next task is to diagonalize the Hamiltonian"
"Using Q.eigenenergies function in QuTiP or np.linalg does not arranges the values properly"
"we would like to keep the same order as the original basis"
#=====================================================================================================
def zz_khz(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34):
    val, vec = np.linalg.eig(red_h(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34))
    vec_arr = np.argmax(vec, axis = 0) #eigenvectors along columns
    val_sor_ind = np.argsort(vec_arr)
    val_incr = val[val_sor_ind]
    #print(np.real(val_incr))
    #=================================================================================================
    #writing the following part that gived the order of the basis so that you dont have to manually search it!
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    b = np.zeros(d_hs, dtype =int) #empty array to define order of the basis
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            for k in range(0, i-j+1):
                for m in range(0, i-j-k+1):
                    b[l] = np.base_repr((dim**3)*j + (dim**2)*k + dim*m + (i - j - k - m), dim)                   
                    l = l+1
    #print (b)
    #=================================================================================================
    b0000 = np.where(b == 0)[0][0]
    b0001 = np.where(b == 1)[0][0]
    b0010 = np.where(b == 10)[0][0]
    b0100 = np.where(b == 100)[0][0]
    b1000 = np.where(b == 1000)[0][0]
    
    b0011 = np.where(b == 11)[0][0]
    b1001 = np.where(b == 1001)[0][0]
    b0101 = np.where(b == 101)[0][0]
    b1100 = np.where(b == 1100)[0][0]
    b0110 = np.where(b == 110)[0][0]
    b1010 = np.where(b == 1010)[0][0]
    #==================================================================================================
    zz_split_tot_12 = np.real(val_incr[b1100] - val_incr[b0100] - val_incr[b1000] + val_incr[b0000])
    zz_split_tot_13 = np.real(val_incr[b1010] - val_incr[b0010] - val_incr[b1000] + val_incr[b0000])
    zz_split_tot_14 = np.real(val_incr[b1001] - val_incr[b0001] - val_incr[b1000] + val_incr[b0000])
    zz_split_tot_23 = np.real(val_incr[b0110] - val_incr[b0010] - val_incr[b0100] + val_incr[b0000])
    zz_split_tot_24 = np.real(val_incr[b0101] - val_incr[b0001] - val_incr[b0100] + val_incr[b0000])
    zz_split_tot_34 = np.real(val_incr[b0011] - val_incr[b0001] - val_incr[b0010] + val_incr[b0000])
    zz_err= np.array([zz_split_tot_12/2, zz_split_tot_13/2, zz_split_tot_14/2, zz_split_tot_23/2, zz_split_tot_24/2, zz_split_tot_34/2])
    #==================================================================================================
    freq = np.array([w1, w2, w3, w4])
    anhar = np.array([del1, del2, del3, del4])
    print('====================================================')
    print('Qubit parameters:')
    print('====================================================')
    for output in range (0, 4):
        print ('Q', output+1,  ' frequency: ', freq[output], ' GHz. Anharmonicity: ', 1000*anhar[output], ' MHz', sep = "")
    print('====================================================')
    print('pairwise ZZ shifts:')
    print('====================================================')
    count = 0
    for cpl1 in range (0, 3):
        for cpl2 in range(0, 3-cpl1):
            print('Q', cpl1+1, '-Q', cpl2+cpl1+2, ' coupling: ', zz_err[count]*1e6,' kHz', sep = "")
            count = count + 1
    print('====================================================')
    #print('zz_error in kHz, zz_12, zz_13, zz_14, zz_23, zz_24, zz_34')
    #print(zz_err*1e6)
#=====================================================================================================

   
    