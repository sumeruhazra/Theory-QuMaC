# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03

@author: Kishor
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt

pi = np.pi
global psi0
global tlist

def instate(n1,Nq1,n2,Nq2):
    
    global N1,N2
    N1 = Nq1; N2 = Nq2
    
    psi0_1 = 0
    psi0_2 = 0
    
    for i in range(len(n1)):
        
        psi0_1 = psi0_1 + n1[i]*basis(N1,i)
        
    for i in range(len(n2)):
        
        psi0_2 = psi0_2 + n2[i]*basis(N2,i)
        
    psi0 = tensor(psi0_1,psi0_2)
    
    return psi0.unit()

def hamiltonian(f_q1,d_q1,f_q2,d_q2,g):
    
    w_q1 = 2*np.pi*f_q1
    w_q2 = 2*np.pi*f_q2

    d_q1 = 2*np.pi*d_q1
    d_q2 = 2*np.pi*d_q2
    
    g_q1q2= 2.0*np.pi*g
    
    global N1,N2,aq1,aq2,X1,Y1,Z1,X2,Y2,Z2,Nq1,Nq2,c_op_list
    aq1 = tensor(destroy(N1),qeye(N2))   # qubit1 annihilation op
    aq2 = tensor(qeye(N1),destroy(N2))   # qubit2 annihilation op
    
    X1=(aq1+aq1.dag()); Y1=-1j*(aq1-aq1.dag()); Z1=(1.0 - 2.0*aq1.dag()*aq1); Nq1= aq1.dag()*aq1
    X2=(aq2+aq2.dag()); Y2=-1j*(aq2-aq2.dag()); Z2=(1.0 - 2.0*aq2.dag()*aq2); Nq2= aq2.dag()*aq2
    
    global H0, H
    H0 = w_q1*Nq1 + d_q1*Nq1*(1-Nq1) + w_q2*Nq2 + d_q2*Nq2*(1-Nq2) + g*(aq1 + aq1.dag())*(aq2 + aq2.dag())
    H1 = X1
    H2 = X2
    H = [H0,[H1,H1_coeff],[H2,H2_coeff]]

def main(A1,A2,wd1,wd2,phi1,phi2,t1,t2,ts):
    
    global tlist, H
    tlist = np.linspace(t1,t2,int(t2-t1+1)*ts)
    
    global args
    args={'A1':2*pi*A1,'w_d1':2.0*np.pi*wd1,'phi1':phi1, 'A2':2*pi*A2,'w_d2':2.0*np.pi*wd2,'phi2':phi2}
    
    c_op_list=[]
    
    global output, output1
    output=mesolve(H, psi0, tlist, c_op_list, [Z1,Z2],args=args,options=None,progress_bar=None)
    output1=mesolve(H, psi0, tlist, c_op_list, [],args=args,options=None,progress_bar=None)
    
    return output.expect,output1
    
def H1_coeff(t,args):
    if t<=25:
        return -1.0*args['A1']*np.cos(args['w_d1']*t + args['phi1'])
    elif t>75:
        return -1.0*args['A1']*np.cos(args['w_d1']*t + args['phi1'])
    else:
        return 0

def H2_coeff(t,args):
    if t<=25:
        return 0
    elif t>75:
        return 0
    else:
        return -1.0*args['A2']*np.cos(args['w_d2']*t + args['phi2'])

def plot(exp,f,tlist,control_state):
    label=["X1","Y1","Z1"]
    k=0
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    for i in range(3):
        if exp[i]==1:
            ax1.plot(tlist,f[k],label=label[i])
            k+=1
        else:
            k+=1
    ax1.set_ylim([-1.1, 1.1])
    ax1.legend()
    ax1.grid()
    ax1.set_title('Exp. Values_ control Qubit in:'+str(control_state))
    plt.show()

def expt1():
    global psi0
    psi0=instate([1,0,0],3,[1,0,0],3)
    hamiltonian(4.59329,0.304,4.65662,0.306,-0.0047)
    E=H0.eigenstates()
    E1=E[0]/(2.0*np.pi)
    print("Freq of Q1 when Q2 is in 0-state(00-->10):",(E1[1]-E1[0])," GHz")
    print("Freq of Q2 when Q1 is in 0-state(00-->01):",(E1[2]-E1[0])," GHz")
    print("Freq of Q1 when Q2 is in 1-state(01-->11):",(E1[5]-E1[2])," GHz")
    print("Freq of Q2 when Q1 is in 1-state(10-->11):",(E1[5]-E1[1])," GHz")
    wq1_0=(E1[1]-E1[0]); wq1_1=(E1[5]-E1[2])
    wq2_0=(E1[2]-E1[0]); wq2_1=(E1[5]-E1[1])
    w1_avg=(wq1_0 + wq1_1)/2
    w2_avg=(wq2_0 + wq2_1)/2
    output=main(0.01,0.0,wq1_0,wq2_0,0,0,0,100,10)
    print("expt1 initial state:")
    print(output[1].states[0])
    print("expt1 final state:")
    print(output[1].states[-1])
    plt.plot(tlist,output[0][0],label='Q1')
    plt.plot(tlist,output[0][1],label='Q2')
    plt.legend()
    plt.show()

def avg_value(state,op):
    avg=np.array([])
    for i in range(len(state)):
        temp=state[i].dag()*op*state[i]
        avg=np.append(avg,temp)
    return avg    

def merge_array(a,b):
    a=np.array(a); b=np.array(b)
    for i in range(len(b)-1):
        a=np.append(a,(b[i+1]))
    return a

def expt2():
    global tlist1,tlist2,states_run1,states_run2,psi0,Z1,Z2
    psi0=instate([1,0,0],3,[1,0,0],3)
    hamiltonian(4.59329,0.304,4.65662,0.306,-0.0047)
    E=H0.eigenstates()
    E1=E[0]/(2.0*np.pi)
    print("Freq of Q1 when Q2 is in 0-state(00-->10):",(E1[1]-E1[0])," GHz")
    print("Freq of Q2 when Q1 is in 0-state(00-->01):",(E1[2]-E1[0])," GHz")
    print("Freq of Q1 when Q2 is in 1-state(01-->11):",(E1[5]-E1[2])," GHz")
    print("Freq of Q2 when Q1 is in 1-state(10-->11):",(E1[5]-E1[1])," GHz")
    wq1_0=(E1[1]-E1[0]); wq1_1=(E1[5]-E1[2])
    wq2_0=(E1[2]-E1[0]); wq2_1=(E1[5]-E1[1])
    w1_avg=(wq1_0 + wq1_1)/2
    w2_avg=(wq2_0 + wq2_1)/2
    output_run1=main(0.01,0.0,w1_avg,w2_avg,0,0,0,50,10)
    tlist1=tlist
    states_run1=output_run1[1].states
    psi0=X2*output_run1[1].states[-1]
    output_run2=main(0.01,0.0,w1_avg,w2_avg,0,0,50,100,10)
    tlist2=tlist
    states_run2=output_run2[1].states
    
    run1_Z1=avg_value(states_run1,Z1)
    run1_Z2=avg_value(states_run1,Z2)
    
    run2_Z1=avg_value(states_run2,Z1)
    run2_Z2=avg_value(states_run2,Z2)
    
    tlist_all= merge_array(tlist1,tlist2)
    avg_Z1=merge_array(run1_Z1,run2_Z1)
    avg_Z2=merge_array(run1_Z2,run2_Z2)
    
    print("expt1 initial state:")
    print(states_run1[0])
    print("expt1 final state:")
    print(states_run2[-1])
    
    
    plt.plot(tlist_all,avg_Z1,label='Q1')
    plt.plot(tlist_all,avg_Z2,label='Q2')
    plt.legend()
    plt.show()
    
    
def expt3():
    global psi0
    psi0=instate([1,0,0],3,[1,0,0],3)
    hamiltonian(4.59329,0.304,4.65662,0.306,-0.047)#Coupling value is chaged
    E=H0.eigenstates()
    E1=E[0]/(2.0*np.pi)
    print("Freq of Q1 when Q2 is in 0-state(00-->10):",(E1[1]-E1[0])," GHz")
    print("Freq of Q2 when Q1 is in 0-state(00-->01):",(E1[2]-E1[0])," GHz")
    print("Freq of Q1 when Q2 is in 1-state(01-->11):",(E1[5]-E1[2])," GHz")
    print("Freq of Q2 when Q1 is in 1-state(10-->11):",(E1[5]-E1[1])," GHz")
    wq1_0=(E1[1]-E1[0]); wq1_1=(E1[5]-E1[2])
    wq2_0=(E1[2]-E1[0]); wq2_1=(E1[5]-E1[1])
    w1_avg=(wq1_0 + wq1_1)/2
    w2_avg=(wq2_0 + wq2_1)/2
    output=main(0.01,0.01,w1_avg,w2_avg,0,0,0,100,10)
    print("expt1 initial state:")
    print(output[1].states[0])
    print("expt1 final state:")
    print(output[1].states[-1])
    plt.plot(tlist,output[0][0],label='Q1')
    plt.plot(tlist,output[0][1],label='Q2')
    plt.legend()
    plt.show()



    