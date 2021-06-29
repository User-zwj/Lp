################################################################
### This file is used to generate Table 4-6, Fig 2-3 ###
################################################################

import os
import numpy as np
import numpy.polynomial.legendre as leg
from scipy.stats import beta
from scipy.stats import uniform
from scipy.integrate import odeint
from scipy.stats import gaussian_kde as kde
from matplotlib import pyplot as plt


####### Plot Formatting ######
plt.rc('lines', linewidth = 2)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('legend',fontsize=14)
plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['lines.markersize'] = 8
plt.rcParams['figure.figsize'] = (8.0, 6.0)

def Phi(n):
    '''Define L_n'''
    coeffs = [0]*(n+1)
    coeffs[n] = 1
    return coeffs

def inner2_leg(n):       
    return 2/(2*n+1)

def product3_leg(i,j,l):
    #compute \Phi_i*\Phi_j*\Phi_l
    return lambda x: leg.legval(x, leg.legmul(leg.legmul(Phi(i),Phi(j)),Phi(l))) 

def inner3_leg(i,j,l):
    '''
    compute <\Phi_i\Phi_j\Phi_l>    
    Set up Gauss-Legendra quadrature
    '''

    x, w=leg.leggauss(20) 
    inner=sum([product3_leg(i,j,l)(x[idx]) * w[idx] for idx in range(20)])         
    
    return inner  

def ode_system_leg(y, t, P):   
    '''P indicates highest order of Polynomial we use'''
    dydt = np.zeros(P+1) 
    for l in range(len(dydt)):
        dydt[l] = -(sum(sum(inner3_leg(i,j,l)*ki_leg[i]*y[j] for j in range(P+1)) for i in range(P+1)))/inner2_leg(l)
    return dydt

P=5
ki_leg = [0,1]+[0]*(P-1)
sol_leg = odeint(ode_system_leg, [1.0]+[0.0]*P, np.linspace(0,1,101), args=(P,)) 

def a(i):
    return sol_leg[:,i][50]
coef = np.array([a(0), a(1), a(2), a(3), a(4), a(5)])   #fixed

def Q(i,x):
    return leg.legval(x,coef[:(i+1)])

def Qexact(x):
    return np.exp(-x*0.5)

##################################
######## Forward Problem #########
# Assume
# $$ \lambda_1 \sim U([-1,1]) $$

######### Verify Assumption 1 ############
##### Generate data in Table 4 and 5 #####
def assumption1(n,J):
    np.random.seed(123456)
    initial_sample = np.random.uniform(-1,1,size = J)    
    pfprior_sample_n = Q(n,initial_sample)
    pfprior_dens_n = kde(pfprior_sample_n)
    x = np.linspace(-1,3,1000)
    
    return np.round(np.max(np.abs(np.gradient(pfprior_dens_n(x), x))), 2), np.round(np.max(pfprior_dens_n(x)),2)

size_J = [int(1E3), int(1E4), int(1E5)]
degree_n = [1, 2, 3, 4, 5]
Bound_matrix, Lip_Bound_matrix = np.zeros((3,5)), np.zeros((3,5))
for i in range(3):
    for j in range(5):
        n, J = degree_n[j], size_J[i]
        Lip_Bound_matrix[i,j] = assumption1(n, J)[0]
        Bound_matrix[i,j] = assumption1(n, J)[1]

###########################################
############### Table 4 #################
###########################################
print('Table 4')
print('Bound under certain n and J values')
print(Bound_matrix)

###########################################
############### Table 5 #################
###########################################
print('Table 5')
print('Lipschitz bound under certain n and J values')
print(Lip_Bound_matrix)


################## Verify Lemma 1 ##################
#Build $\pi_D^Q$ and $\pi_D^{Q,n}$, use 10,000 samples
N_kde = int(1E4)
N_mc = int(1E4)

np.random.seed(123456)
initial_sample = np.random.uniform(-1,1,size = N_kde)
pfprior_sample = Qexact(initial_sample)
pfprior_dens = kde(pfprior_sample)

def pfprior_dens_n(n,x):
    pfprior_sample_n = Q(n,initial_sample)
    pdf = kde(pfprior_sample_n)
    return pdf(x)

error_r_D = np.zeros((5,5))
np.random.seed(123456)            
qsample = np.random.uniform(0,2,N_mc)
for i in range(5):
    for j in range(5):
        error_r_D[i,j] = (np.mean((np.abs(pfprior_dens(qsample) - pfprior_dens_n(j+1,qsample)))**(i+1)))**(1/(i+1))

#### To make it cleaner, create Directory "images" to store all the figures ####
imagepath = os.path.join(os.getcwd(),"images")
os.makedirs(imagepath,exist_ok=True)

###########################################
######### The left plot of Fig 2 ##########
###########################################
fig = plt.figure()
plt.xlim([0,6])
marker = ['-D', '-o', '-v', '-s', '-.']
for i in range(5):
    plt.semilogy([1,2,3,4,5],error_r_D[i,:],marker[i],label='r = ' + np.str(i+1))    
plt.xlabel('Order of PCE (n)')
plt.ylabel('$L^r$'+' Error in Push-Forward on '+'$\mathcal{D}$')
plt.legend();
plt.savefig("images/Fig2(Left).png")

#################### Verify Theorem 3.1 ###############
##### Generate data for the right plot of Fig 2 #####
np.random.seed(123456)       
lamsample = np.random.uniform(-1,1,size = N_mc)

error_2 = np.zeros(5)
for i in range(5):
    error_2[i] = (np.mean((np.abs(pfprior_dens(Qexact(lamsample)) - pfprior_dens_n(i+1,Q(i+1,lamsample))))**2))**(1/2)     
    
# print('L^2 error on parameter space for Forward Problem')
# print(error_2)

############################################
######## The right plot of Fig 2 #########
############################################
fig = plt.figure()
plt.xlim([0,6])
plt.semilogy([1,2,3,4,5],error_2,'-s')   
plt.xlabel('Order of PCE (n)')
plt.ylabel('$L^2$'+' Error in Push-Forward on '+'$\Lambda$');
plt.savefig("images/Fig2(Right).png")


##################################
######## Inverse Problem #########
# Compute $\pi_{\Lambda}^u$ and $\pi_{\Lambda}^{u,n}$
# Observed pdf is $\pi_{\mathcal{D}} \sim Beta(4,4)$
# Guess is $\lambda_1\sim U([-1,1])$
def pdf_obs(x):
    return beta.pdf(x, a=4, b=4, loc=1, scale=0.25)

##### Verify Assumption 2 #####
def Meanr(n):
    pfprior_sample_n = Q(n,initial_sample)
    if n==0:
        r = pdf_obs(pfprior_sample)/pfprior_dens(pfprior_sample)
    else:
        r = pdf_obs(pfprior_sample_n)/pfprior_dens_n(n,pfprior_sample_n)
    return np.mean(r)
 

def pdf_update(n,x):
    if n==0:
        r = pdf_obs(pfprior_sample)/pfprior_dens(pfprior_sample)
        pdf = kde(initial_sample,weights=r)
    else:
        pfprior_sample_n = Q(n,initial_sample)
#         pfprior_dens_n = kde(pfprior_sample_n)
        r = pdf_obs(pfprior_sample_n)/pfprior_dens_n(n,pfprior_sample_n)
        pdf = kde(initial_sample,weights=r)
    return pdf(x)

Expect_r = np.zeros(6)
for i in range(6):
    Expect_r[i] = Meanr(i)

###########################################
############### Table 6 #################
###########################################
print('Table 6')
print('Expected ratio for verifying Assumption 2')
print(Expect_r[1:])    


######## Verify Theorem 4.2 #######
##### Generate data for Fig 3 #####
# Print out Monte Carlo Approximation of $\|\pi_{\Lambda}^{u,n}(\lambda)-\pi_{\Lambda}^u(\lambda)\|_{L^2(\Lambda)} $
np.random.seed(123456)     
lamsample = np.random.uniform(-1,1,size = N_mc)

error_update = np.zeros(5)
for i in range(5):
    error_update[i] = (np.mean((np.abs(pdf_update(0,lamsample) - pdf_update(i+1,lamsample)))**2))**(1/2)
    
# print('L^2 Error for Inverse Problem')
# print(error_update)  

###########################################
############### Figure 3 ################
###########################################
fig = plt.figure()
plt.xlim([0,6])
plt.semilogy([1,2,3,4,5],error_update,'-s')   
plt.xlabel('Order of PCE (n)')
plt.ylabel('$L^2$'+' Error in Update');
plt.savefig("images/Fig3.png")