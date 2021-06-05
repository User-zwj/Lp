############################################################
### This file is used to generate Table 1-3, Fig 1 ###
############################################################

import os
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt

####### Plot Formatting ######
plt.rc('lines', linewidth = 4)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('legend',fontsize=14)
plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['lines.markersize'] = 8
plt.rcParams['figure.figsize'] = (7.0, 5.0)

# A simple example in Python: The Spaces, QoI, and Densities
# $\Lambda=[-1,1]$.
# $Q(\lambda) = \lambda^p$ for $p=5$.
# $\mathcal{D} = Q(\Lambda) = [-1,1]$.
# $\pi_{\Lambda}^{init} \sim U([-1,1])$
# $\pi_{\mathcal{D}}^{obs} \sim N(0.25,0.1^2)$
# $\pi_{\mathcal{D}}^{Q(init)}$


def QoI(lam, p):
    '''Defing a QoI mapping function'''
    q = lam**p
    return q


def QoI_approx(lam, p, n):
    '''Defining a QoI approximation with n+2 knots for piecewise linear spline'''
    lam_knots = np.linspace(-1, 1, n+2)
    q_knots = QoI(lam_knots, p)
    q = np.interp(lam, lam_knots, q_knots)
    return q


# number of samples from prior and observed mean (mu) and st. dev (sigma)
N, mus, sigma = int(1E5), [0.5, 0.25, 1], 0.1
lam = np.random.uniform(low=-1, high=1, size=N)  # sample set of the prior

qvals_nonlinear = QoI(lam, 5)  # Evaluate lam^5 samples

# Estimate the push-forward density for the QoI
q_nonlinear_kde = kde(qvals_nonlinear)


##### Verify Assumption 1 #####
##### Generate data in Table 1 and 2 #####
def assumption1(n, J):
    np.random.seed(123456)
    x = np.linspace(-1, 1, 100)
    lam = np.random.uniform(low=-1, high=1, size=J)  # sample set of the init
    qvals_approx_nonlinear = QoI_approx(lam, 5, n)  # Evaluate lam^5 samples
    q_nonlinear_kde = kde(qvals_approx_nonlinear)
    return np.round(np.max(np.abs(np.gradient(q_nonlinear_kde(x), x))), 2), np.round(np.max(q_nonlinear_kde(x)), 2)


size_J = [int(1E3), int(1E4), int(1E5)]
degree_n = [1, 2, 4, 8, 16]
Bound_matrix, Lip_Bound_matrix = np.zeros((3, 5)), np.zeros((3, 5))
for i in range(3):
    for j in range(5):
        n, J = degree_n[j], size_J[i]
        Lip_Bound_matrix[i, j] = assumption1(n, J)[0]
        Bound_matrix[i, j] = assumption1(n, J)[1]
        
###########################################
############## Table 1 ##################
###########################################
print('Table 1')        
print('Bound under certain n and J values')
print(Bound_matrix)

###########################################
############## Table 2 ##################
###########################################
print('Table 2')
print('Lipschitz bound under certain n and J values')
print(Lip_Bound_matrix)


##### Verify Assumption 2 #####
# The expected r value
def Meanr(n, J, m):
    '''
    n: index of approximating mapping
    J: sample size of sample generated from parameter space
    m: index of mu
    '''
    np.random.seed(123456)
    lam = np.random.uniform(low=-1, high=1, size=J)  # sample set of the init
    qvals_approx_nonlinear = QoI_approx(lam, 5, n)  # Evaluate lam^5 samples
    q_nonlinear_kde = kde(qvals_approx_nonlinear)
    obs_vals_nonlinear = norm.pdf(qvals_approx_nonlinear, loc=mus[m], scale=sigma)
    r = np.divide(obs_vals_nonlinear, q_nonlinear_kde(qvals_approx_nonlinear))
    return np.round(np.mean(r), 2)


meanr_matrix = np.zeros((3, 5))
for i in range(3):
    for j in range(5):
        J = int(1E4)
        meanr_matrix[i, j] = Meanr(degree_n[j], J, i)

###########################################
############## Table 3 ##################
###########################################
print('Table 3')
print('Expected ratio for verifying Assumption 2')
print(meanr_matrix)


#### To make it cleaner, create Directory "images" to store all the figures ####
imagepath = os.path.join(os.getcwd(),"images")
os.makedirs(imagepath,exist_ok=True)

fig = plt.figure()

def plot_all(i):
    fig.clear()
    case = ['Case I', 'Case II', 'Case III']
    qplot = np.linspace(-1, 1, num=100)
    observed_plot = plt.plot(qplot, norm.pdf(
        qplot, loc=mus[i], scale=sigma), 'r-', label="$\pi_\mathcal{D}^{obs}$")
    pf_init_plot = plt.plot(qplot, 1/10*np.abs(qplot)**(-4/5), 'b-',
                             label="$\pi_\mathcal{D}^{Q(init)}$")
    pf_init_plot = plt.plot(qplot, q_nonlinear_kde(qplot), 'b--',
                             label="$\pi_{\mathcal{D},J}^{Q(init)}$")
    plt.xlim([-1, 1])
    plt.legend()
    plt.title(case[i])；
    loc = ['Left','Mid','Right']
    if loc[i]:
        filename = os.path.join(os.getcwd(), "images", "Fig1(%s).png"%(loc[i])) 
        plt.savefig(filename)
        
###########################################
####### The left plot of Fig 1 ##########
###########################################
plot_all(0)

#############################################
####### The middle plot of Fig 1 ##########
#############################################
plot_all(1)

############################################
####### The right plot of Fig 1 ##########
############################################
plot_all(2)
