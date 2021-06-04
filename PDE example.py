#Note(Important): Since this file would require data from Data directory, 
#you will need to first run GenerateData.py before you run this file.

########################################################
### This file is used to generate Table 4-5, Fig 3-4 ###
########################################################

import os
import scipy.io as sio  # for the i/o
import numpy as np
import numpy.polynomial.hermite_e as H
from math import factorial
from scipy.stats import norm
from scipy.stats import gaussian_kde as kde
from matplotlib import pyplot as plt

proc_size = 25   # this number is determined by the data file

mu1 = 0
mu2 = 0
sigma1 = 0.1
sigma2 = 0.1


# Get $Q_n(\lambda_1,\lambda_2)$ (need to compute coef $q_{ij}$)
def Hermite_2d(i, j, x, y):
    '''
    Phi_{i,j}(x,y) = Phi_i(x) * Phi_j(y)  (left: 2d; right: 1d)
    '''
    c = np.zeros((20, 20))
    c[i, j] = 1
    return H.hermeval2d(x, y, c)


Q_FEM_quad = np.zeros(int(400))
for i in range(proc_size):
    filename = os.path.join(os.getcwd(), "Data", "Q_FEM_quad_") + str(i) + ".mat"
    partial_data = sio.loadmat(filename)
    Q_FEM_quad += partial_data['Q_FEM'].reshape(int(400))


def Phi(n):
    '''define H_n'''
    coeffs = [0]*(n+1)
    coeffs[n] = 1
    return coeffs


def q(i, j):
    '''
    copmute coefficient q_{ij}
    Set up Gauss-Hermite quadrature, weighting function is exp^{-x^2}
    '''
    x, w = H.hermegauss(20)
    Q = sum([w[ldx]*sum([w[kdx] * Q_FEM_quad[ldx*20+kdx] * H.hermeval(x[kdx], Phi(i))
                         for kdx in range(20)])*H.hermeval(x[ldx], Phi(j)) for ldx in range(20)])
    q = Q/(2*np.pi*factorial(i)*factorial(j))

    return q


qij = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        qij[i, j] = q(i, j)


def Q(n, x, y):
    result = 0
    for i in range(n+1):
        for j in range(n+1):
            if i+j <= n:
                result += qij[i, j]*Hermite_2d(i, j, (x-mu1)/sigma1, (y-mu2)/sigma2)
    return result


def Qexact(x, y, a=0.4, b=0.6, c=0.4, d=0.6):
    sol = (np.cos(x*np.pi*a)-np.cos(x*np.pi*b)) * \
        (np.sin(y*np.pi*d)-np.sin(y*np.pi*c))/((b-a)*(d-c)*x*y*np.pi**2)
    return sol


##################################
######## Forward Problem #########
# Assume
# $$ \lambda_1 \sim N(\mu_1, \sigma_1^2) = N(0, 0.1^2) \ \ \ \lambda_2 \sim N(\mu_2, \sigma_2^2) = N(0, 0.1^2) $$
#

##### Generate data in Table 4 and 5 #####
def assumption1(n, J):
    np.random.seed(123456)
    lam1sample = np.random.normal(mu1, sigma1, J)
    lam2sample = np.random.normal(mu2, sigma2, J)
    pfprior_sample_n = Q(n, lam1sample, lam2sample)
    pfprior_dens_n = kde(pfprior_sample_n)
    x = np.linspace(-1, 1, 1000)
    return np.round(np.max(np.abs(np.gradient(pfprior_dens_n(x), x))), 2), np.round(np.max(pfprior_dens_n(x)), 2)


size_J = [int(1E3), int(1E4), int(1E5)]
degree_n = [1, 2, 3, 4, 5]
Bound_matrix, Lip_Bound_matrix = np.zeros((3, 5)), np.zeros((3, 5))
for i in range(3):
    for j in range(5):
        n, J = degree_n[j], size_J[i]
        Lip_Bound_matrix[i, j] = assumption1(n, J)[0]
        Bound_matrix[i, j] = assumption1(n, J)[1]

        
###########################################
################ Table 4 ##################
###########################################
print('Table 4')
print('Bound under certain n and J values')
print(Bound_matrix)

###########################################
################ Table 5 ##################
###########################################
print('Table 5')
print('Lipschitz bound under certain n and J values')
print(Lip_Bound_matrix)


##### Generate data for the left plot of Fig 3 #####
# Define push-forward densities
N_kde = int(1E4)
N_mc = int(1E4)
np.random.seed(123456)
lam1sample = np.random.normal(mu1, sigma1, N_kde)
lam2sample = np.random.normal(mu2, sigma2, N_kde)
pfprior_dens = kde(Qexact(lam1sample, lam2sample))

def pfprior_dens_n(n, x):
    pfprior_sample_n = Q(n, lam1sample, lam2sample)
    pdf = kde(pfprior_sample_n)
    return pdf(x)

# **Print out Monte Carlo Approximation of $	\|\pi_{\mathcal{D}}^Q(q)-\pi_{\mathcal{D}}^{Q_n}(q)\|_{L^r(\mathcal{D_c})} $ where $r>0$ and $D_c=[-1,1]$**
np.random.seed(123456)
qsample = np.random.uniform(-1, 1, N_mc)

def error_r_onD(r, n):
    diff = (np.mean((np.abs(pfprior_dens_n(n, qsample) - pfprior_dens(qsample)))**r))**(1/r)
    return diff

error_r_D = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        error_r_D[i, j] = error_r_onD(i+1, j+1)
        
# print('L^r error on data space for Forward Problem')
# print(error_r_D)

#### To make it cleaner, create Directory "images" to store all the figures ####
imagepath = os.path.join(os.getcwd(),"images")
os.makedirs(imagepath,exist_ok=True)

###########################################
######### The left plot of Fig 3 ##########
###########################################
fig = plt.figure(figsize=(8, 6))
plt.xlim([0, 6])
marker = ['-D', '-o', '-v', '-s', '-.']
for i in range(5):
    plt.semilogy([1, 2, 3, 4, 5], error_r_D[i, :], marker[i], label='r = ' + np.str(i+1))
plt.xticks(fontsize=14, rotation=0)
plt.xlabel('Order of PCE (n)', fontsize=20)
plt.yticks(fontsize=14, rotation=0)
plt.ylabel('$L^r$'+' Error in Push-Forward on '+'$\mathcal{D}$', fontsize=20)
plt.legend(prop={'size': 14})
plt.savefig("images/Fig3(Left).png")


##### Generate data for the right plot of Fig 3 #####
# Print out Monte Carlo Approximation of $	\|\pi_{\mathcal{D}}^Q(Q(\lambda))-\pi_{\mathcal{D}}^{Q_n}(Q_n(\lambda))\|_{L^2(\Lambda)} $

np.random.seed(123456)
lam1_seed = np.random.normal(mu1, sigma1, int(1E4))
lam2_seed = np.random.normal(mu2, sigma2, int(1E4))  # int(1E4) since Q_FEM size
error_2_Lam = np.zeros(5)
for i in range(5):
    pfprior_sample = Qexact(lam1_seed, lam2_seed)
    error_2_Lam[i] = (np.mean(
        (np.abs(pfprior_dens_n(i+1, Q(i+1, lam1_seed, lam2_seed)) - pfprior_dens(pfprior_sample)))**2))**(1/2)

# print('L^2 error on parameter space for Forward Problem')
# print(error_2_Lam)

############################################
######### The right plot of Fig 3 ##########
############################################
fig = plt.figure(figsize=(8, 6))
plt.xlim([0, 6])
plt.semilogy([1, 2, 3, 4, 5], error_2_Lam, '-s')  # , label='$L^2(\Lambda)$ error')
plt.xticks(fontsize=14, rotation=0)
plt.xlabel('Order of PCE (n)', fontsize=20)
plt.yticks(fontsize=14, rotation=0)
plt.ylabel('$L^2$'+' Error in Push-Forward on '+'$\Lambda$', fontsize=20)
plt.savefig("images/Fig3(Right).png")


##################################
######## Inverse Problem #########
# Compute $\pi_{\Lambda}^u$ and $\pi_{\Lambda}^{u,n}$
# Observed pdf is $\pi_{\mathcal{D}} \sim N(0.3,0.1^2)$
# Guess is $\lambda_1\sim N(0,0.1)$, $\lambda_2\sim N(0,0.1)$


def rejection_sampling(r):
    N = r.size  # size of proposal sample set
    # create random uniform weights to check r against
    check = np.random.uniform(low=0, high=1, size=N)
    M = np.max(r)
    new_r = r/M     # normalize weights
    idx = np.where(new_r >= check)[0]  # rejection criterion
    return idx


def pdf_obs(x):
    return norm.pdf(x, loc=0.1, scale=0.1)


##### Verify Assumption 2 #####

def Meanr(n):
    if n == 0:
        pfprior_sample = Qexact(lam1_seed, lam2_seed)
        r = pdf_obs(pfprior_sample)/pfprior_dens(pfprior_sample)
    else:
        pfprior_sample_n = Q(n, lam1_seed, lam2_seed)
        r = pdf_obs(pfprior_sample_n)/pfprior_dens_n(n, pfprior_sample_n)
    return np.mean(r)


Expect_r = np.zeros(6)
for i in range(6):
    Expect_r[i] = Meanr(i)
    
print('Expected ratio for verifying Assumption 2')
print(Expect_r[1:])


##### Load data for Fig 4 #####
# Print out Monte Carlo Approximation of $\|\pi_{\Lambda}^{u,n}(\lambda)-\pi_{\Lambda}^u(\lambda)\|_{L^2(\Lambda)} $
init_eval = np.zeros(int(1E4))
for i in range(int(1E4)):
    init_eval[i] = norm.pdf(lam1_seed[i], loc=0.1, scale=0.1) * \
        norm.pdf(lam2_seed[i], loc=0.1, scale=0.1)

r = np.zeros(int(1E4))
for i in range(proc_size):
    filename = os.path.join(os.getcwd(), "Data", "r_") + str(i) + ".mat"
    partial_data = sio.loadmat(filename)
    r += partial_data['r'].reshape(int(1E4))

rn = np.zeros((6, int(1E4)))
for i in range(6):
    for j in range(proc_size):
        filename = os.path.join(os.getcwd(), "Data", "r")  + str(i+1) + '_' + str(j) + ".mat"
        partial_data = sio.loadmat(filename)
        rn[i, :] += partial_data['r'].reshape(int(1E4))


error_Update = np.zeros(5)

for i in range(5):
    error_Update[i] = (np.mean(init_eval*(rn[i, :] - r)**2))**(1/2)

# print('L^2 Error for Inverse Problem')
# print(error_update)


###########################################
################ Figure 4 #################
###########################################
fig = plt.figure(figsize=(8, 6))
plt.xlim([0, 6])
plt.semilogy([1, 2, 3, 4, 5], error_Update, '-s')  # , label='$L^2(\Lambda)$ error')
plt.xticks(fontsize=14, rotation=0)
plt.xlabel('Order of PCE (n)', fontsize=20)
plt.yticks(fontsize=14, rotation=0)
plt.ylabel('$L^2$'+' Error in Update', fontsize=20)
plt.savefig("images/Fig4")
