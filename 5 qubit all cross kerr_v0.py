# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:53:04 2021

@author: Sumeru
"""
# Computing the cross-kerr shifts of five transversely coupled qubits
#given frequencies, bare coupling and anharmonicities
#not completed

#=====================================================================================================
from qutip import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
no_of_qubits = 5
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
        
w = np.array([5.7, 5.77, 5.84, 5.91, 5.97]) #in GHz
anh = np.array([-0.33, -0.33, -0.33, -0.33, -0.33]) #in GHz
g = np.array([0.002, 0.002, -0.002, 0.00002, -0.003,  0.003, -0.002, 0.002, -0.003,  0.003]) #in GHz


def fivequbit_h(dim, w, anh, g):
#def fourqubit_h(dim, w1, w2, w3, w4, del1, del2, del3, del4, g12, g13, g14, g23, g24, g34):
    #this function defines the two qubit hamiltonian with delta being anharmonicities
    #the exchange coupling is given by g12
    a1 = tensor(destroy(dim), qeye(dim), qeye(dim), qeye(dim), qeye(dim))
    a1dag = tensor(create(dim), qeye(dim), qeye(dim), qeye(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim), qeye(dim), qeye(dim), qeye(dim))
    a2dag = tensor(qeye(dim), create(dim), qeye(dim), qeye(dim), qeye(dim))
    a3 = tensor(qeye(dim), qeye(dim), destroy(dim), qeye(dim), qeye(dim))
    a3dag = tensor(qeye(dim), qeye(dim), create(dim), qeye(dim), qeye(dim))
    a4 = tensor(qeye(dim), qeye(dim), qeye(dim), destroy(dim), qeye(dim))
    a4dag = tensor(qeye(dim), qeye(dim), qeye(dim), create(dim), qeye(dim))
    a5 = tensor(qeye(dim), qeye(dim), qeye(dim), qeye(dim), destroy(dim))
    a5dag = tensor(qeye(dim), qeye(dim), qeye(dim), qeye(dim), create(dim))
    H1 = w[0]*(a1dag*a1) + (anh[0]/2)*((a1dag*a1)*(a1dag*a1)-(a1dag*a1))
    H2 = w[1]*(a2dag*a2) + (anh[1]/2)*((a2dag*a2)*(a2dag*a2)-(a2dag*a2))
    H3 = w[2]*(a3dag*a3) + (anh[2]/2)*((a3dag*a3)*(a3dag*a3)-(a3dag*a3))
    H4 = w[3]*(a4dag*a4) + (anh[3]/2)*((a4dag*a4)*(a4dag*a4)-(a4dag*a4))
    H5 = w[4]*(a5dag*a5) + (anh[4]/2)*((a5dag*a5)*(a5dag*a5)-(a5dag*a5))
    Hc1 = g[0]*(a1dag*a2 + a1*a2dag) + g[1]*(a1dag*a3 + a1*a3dag) + g[2]*(a1dag*a4 + a1*a4dag) + g[3]*(a1dag*a5 + a1*a5dag)
    Hc2 = g[4]*(a2dag*a3 + a2*a3dag) + g[5]*(a2dag*a4 + a2*a4dag) + g[6]*(a2dag*a5 + a2*a5dag)
    Hc3 = g[7]*(a3dag*a4 + a3*a4dag) + g[8]*(a3dag*a5 + a3*a5dag)
    Hc4 = g[9]*(a4dag*a5 + a4*a5dag)
    Htot = H1 + H2 + H3 + H4 + H5 + Hc1 + Hc2 + Hc3 + Hc4
    return Htot
#=====================================================================================================
# The ordering of states in qutip is in number_base(n) format as in 
# {0000, 0001, 0002, ..., 000d, 0010, 0011, ..., 001d, ..., 00d0, ..., 00dd, ..., 0ddd, ..., dddd}
# We shall first truncate the Hamiltonian to limit the maximum number of excitations
# Next we rearrange the Hamiltonian in the binomial pyramid format given below:
# {0(0(00)), 0(0(01)), 0(0(10)), 0(1(00)), 1(0(00)), 0(0(02)), 0(0(11)), 0(0(20)), 0(1(01)),
# 0(1(10)), 0(2(00)), 1(0(01)), 1(0(10)), 1(1(00)), 2(0(00)), ..., 0(0(0d)), ..., d(0(00))}
#=====================================================================================================   
def red_h(dim, w, anh, g):
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    y = np.zeros(d_hs, dtype =int)
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            for k in range(0, i-j+1):
                for m in range(0, i-j-k+1):
                    for s in range(0, i-j-k-m+1):
                        y[l] = (dim**4)*j + (dim**3)*k + (dim**2)*m + dim*s + (i - j - k - m - s)                    
                        l = l+1
    empty2d = np.zeros([d_hs, d_hs], dtype = complex)
    base_ham = fivequbit_h(dim, w, anh, g)
    for i1 in range(0, d_hs):
        for j1 in range(0, d_hs):
            empty2d[i1, j1] = base_ham[y[i1], y[j1]]
    return empty2d
#=====================================================================================================
"Next task is to diagonalize the Hamiltonian"
"Using Q.eigenenergies function in QuTiP or np.linalg does not arranges the values properly"
"we would like to keep the same order as the original basis"
#=====================================================================================================
def zz_khz(dim, w, anh, g):
    val, vec = np.linalg.eig(red_h(dim, w, anh, g))
    vec_arr = np.argmax(vec, axis = 0) #eigenvectors along columns
    val_sor_ind = np.argsort(vec_arr)
    val_incr = val[val_sor_ind]
#    print(np.real(val_incr))
    #=================================================================================================
    #writing the following part that gived the order of the basis so that you dont have to manually search it!
    d_hs = binomial(dim + no_of_qubits - 1, no_of_qubits ) #dimension of Hilbert Space
    b = np.zeros(d_hs, dtype =int) #empty array to define order of the basis
    l = 0
    for i in range(0, dim):
        for j in range(0, i+1):
            for k in range(0, i-j+1):
                for m in range(0, i-j-k+1):
                    for s in range(0, i-j-k-m+1):
                        b[l] = np.base_repr((dim**4)*j + (dim**3)*k + (dim**2)*m + dim*s + (i - j - k - m - s), dim)                   
                        l = l+1
    print ("Solving a Hilbert sapce of dimension:", d_hs)
    #=================================================================================================

    enarr0 = np.zeros(80, dtype =int)
    enarr1 = np.zeros(80, dtype =int)
    for i in range (0, 16):
        l = np.binary_repr(i).zfill(4)
        enarr0[64+i] = l[0] + l[1] + l[2] + l[3] + '0'
        enarr1[64+i] = l[0] + l[1] + l[2] + l[3] + '1'
        enarr0[48+i] = l[0] + l[1] + l[2] + '0' + l[3]
        enarr1[48+i] = l[0] + l[1] + l[2] + '1' + l[3]
        enarr0[32+i] = l[0] + l[1] + '0' + l[2] + l[3]
        enarr1[32+i] = l[0] + l[1] + '1' + l[2] + l[3]
        enarr0[16+i] = l[0] + '0' + l[1] + l[2] + l[3]
        enarr1[16+i] = l[0] + '1' + l[1] + l[2] + l[3]
        enarr0[i] = '0' + l[0] + l[1] + l[2] + l[3]
        enarr1[i] = '1' + l[0] + l[1] + l[2] + l[3]
    enindex = np.zeros(80, dtype =float) #empty array to define order of the levels 

    for rep in range(0, 80):
        enindex[rep] = np.real(val_incr[np.where(b == enarr1[rep])[0][0]]-val_incr[np.where(b == enarr0[rep])[0][0]])
    enindex = np.reshape(enindex, (5, 16))    
    q1 = (enindex[0]-enindex[0][0])*1e6
    q2 = (enindex[1]-enindex[1][0])*1e6
    q3 = (enindex[2]-enindex[2][0])*1e6
    q4 = (enindex[3]-enindex[3][0])*1e6
    q5 = (enindex[4]-enindex[4][0])*1e6
    
    joinedarr = np.concatenate((q1, q2, q3, q4, q5))
    joined2d = np.reshape(joinedarr, (5, 16)) 
    transposedarr = np.transpose(joined2d)

    print ("Total cross kerr shifts(2ZZ) measured for each qubit:\n", transposedarr.round(decimals=3))   
    #=================================================================================================
    #PLOT
    #=================================================================================================
    
    labels = ['Qubit1', 'Qubit2', 'Qubit3', 'Qubit4', 'Qubit5']

    x = np.arange(len(labels))  # the label locations
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots()
    for i in range (0, 16):
        ax.bar(x -5*width +i*width, transposedarr[i], width, label=np.binary_repr(i).zfill(4))


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Change of freq (kHz)')
    ax.set_title('Cross-Kerr shift in 5 qubit system')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
    plt.grid(True)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()

    plt.show()

#=====================================================================================================

#example function call
#zz_khz(6, 4.74876, 4.65661, 4.63747, 4.59394, -0.31, -0.306, -0.320, -0.304, 0.0030785, 0.0027181, -0.0031776, 0.00002372, -0.00470286,  0.0046876)  
    