#############################################################################################
### This file is used to generate Data needed for "PDE example.py" or "PDE example.ipynb" ###
#############################################################################################

import os
import numpy as np
import dolfin as fn
import scipy.io as sio
import numpy.polynomial.hermite_e as H 
from scipy.stats import gaussian_kde as kde
from scipy.stats import norm
from math import factorial 

## This is just set to be consistent with the number of files GenerateData_ParallelVersion.ipynb generates
proc_size = 25 

#### To make it cleaner, create Directory "Data" to store all the data ####
datapath = os.path.join(os.getcwd(),"Data")
os.makedirs(datapath,exist_ok=True)

def Compute_Q(proc_num, proc_max, mu1=0, mu2=0, sigma1=0.1, sigma2=0.1, gridx=50, gridy=50, p=1):
    num_quad = 20
    lambda1 = H.hermegauss(num_quad)[0]
    lambda2 = H.hermegauss(num_quad)[0]
    
    # Create the characteristic function class used to define the QoI
    class AvgCharFunc(fn.UserExpression):
        def __init__(self, region, **kwargs):
            self.a = region[0]
            self.b = region[1]
            self.c = region[2]
            self.d = region[3]
            super().__init__(**kwargs)
        def eval(self, v, x):
            v[0] = 0
            if (x[0] >= self.a) & (x[0] <= self.b) & (x[1] >= self.c) & (x[1] <= self.d):
                v[0] = 1./( (self.b-self.a) * (self.d-self.c) )
            return v
        def value_shape(self):
            return ()
        
    def QoI_FEM(lam1,lam2,pointa,pointb,gridx,gridy,p):
        aa = pointa[0]
        bb = pointb[0]
        cc = pointa[1]
        dd = pointb[1]

        mesh = fn.UnitSquareMesh(gridx, gridy)
        V = fn.FunctionSpace(mesh, "Lagrange", p)

        # Define diffusion tensor (here, just a scalar function) and parameters
        A = fn.Expression((('exp(lam1)','a'),
                    ('a','exp(lam2)')), a = fn.Constant(0.0), lam1 = lam1, lam2 = lam2, degree=3) 

        u_exact = fn.Expression("sin(lam1*pi*x[0])*cos(lam2*pi*x[1])", lam1 = lam1, lam2 = lam2, degree=2+p)

        # Define the mix of Neumann and Dirichlet BCs
        class LeftBoundary(fn.SubDomain):
            def inside(self, x, on_boundary):
                return (x[0] < fn.DOLFIN_EPS)
        class RightBoundary(fn.SubDomain):
            def inside(self, x, on_boundary):
                return (x[0] > 1.0 - fn.DOLFIN_EPS)
        class TopBoundary(fn.SubDomain):
            def inside(self, x, on_boundary):
                return (x[1] > 1.0 - fn.DOLFIN_EPS)
        class BottomBoundary(fn.SubDomain):
            def inside(self, x, on_boundary):
                return (x[1] < fn.DOLFIN_EPS)

        # Create a mesh function (mf) assigning an unsigned integer ('uint')
        # to each edge (which is a "Facet" in 2D)
        mf = fn.MeshFunction('size_t', mesh, 1)
        mf.set_all(0) # initialize the function to be zero
        # Setup the boundary classes that use Neumann boundary conditions
        NTB = TopBoundary() # instatiate
        NTB.mark(mf, 1) # set all values of the mf to be 1 on this boundary
        NBB = BottomBoundary()
        NBB.mark(mf, 2) # set all values of the mf to be 2 on this boundary
        NRB = RightBoundary()
        NRB.mark(mf, 3)

        # Define Dirichlet boundary conditions
        Gamma_0 = fn.DirichletBC(V, u_exact, LeftBoundary())
        bcs = [Gamma_0]

        # Define data necessary to approximate exact solution
        f = ( fn.exp(lam1)*(lam1*fn.pi)**2 + fn.exp(lam2)*(lam2*fn.pi)**2 ) * u_exact
        #g1:#pointing outward unit normal vector, pointing upaward (0,1)
        g1 = fn.Expression("-exp(lam2)*lam2*pi*sin(lam1*pi*x[0])*sin(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)
        #g2:pointing downward (0,1)
        g2 = fn.Expression("exp(lam2)*lam2*pi*sin(lam1*pi*x[0])*sin(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)
        g3 = fn.Expression("exp(lam1)*lam1*pi*cos(lam1*pi*x[0])*cos(lam2*pi*x[1])", lam1=lam1, lam2=lam2, degree=2+p)

        fn.ds = fn.ds(subdomain_data=mf)
        # Define variational problem
        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.inner(A*fn.grad(u), fn.grad(v))*fn.dx
        L = f*v*fn.dx + g1*v*fn.ds(1) + g2*v*fn.ds(2) + g3*v*fn.ds(3)  #note the 1, 2 and 3 correspond to the mf

        # Compute solution
        u = fn.Function(V)
        fn.solve(a == L, u, bcs)
        psi = AvgCharFunc([aa, bb, cc, dd], degree=0)
        Q = fn.assemble(fn.project(psi * u, V) * fn.dx)
        
        return Q

    Q_FEM = np.zeros(400)
    num_Q_per_proc = 400//proc_max
    if proc_num != proc_size -1:       
        for i in range(proc_num*num_Q_per_proc, (proc_num+1)*num_Q_per_proc):
            Q_FEM[i] = QoI_FEM(mu1+sigma1*lambda1[i%num_quad],mu2+sigma2*lambda2[i//num_quad],[0.4,0.4],[0.6,0.6],gridx,gridy,p)
    else:
        for i in range(proc_num*num_Q_per_proc,400):
            Q_FEM[i] = QoI_FEM(mu1+sigma1*lambda1[i%num_quad],mu2+sigma2*lambda2[i//num_quad],[0.4,0.4],[0.6,0.6],gridx,gridy,p)
    filename = os.path.join(os.getcwd(), "Data", "Q_FEM_quad_") + str(proc_num) + ".mat" 
    data_dict = {'Q_FEM': Q_FEM}
    sio.savemat(filename, data_dict)
    return

#########################################################
##### Generate datafiles Data/Q_FEM_quad_[0-24].mat #####
#########################################################
for i in range(proc_size):
    Compute_Q(i, proc_max=proc_size)
      
def r(nn, proc_num, proc_max):
    mu1 = 0
    mu2 = 0
    sigma1 = 0.1
    sigma2 = 0.1
    N_size = int(1E4)
    np.random.seed(123456)
    lam1 = np.random.normal(mu1,sigma1,N_size)
    lam2 = np.random.normal(mu2,sigma2,N_size)
    
    def Hermite_2d(i,j,x,y):
        c = np.zeros((20,20))
        c[i,j] = 1
        return H.hermeval2d(x, y, c)

    Q_FEM_quad = np.zeros(int(400))   #already include information of mu1, mu2, sigma1, sigma2
    for i in range(proc_size):         
        filename = os.path.join(os.getcwd(), "Data", "Q_FEM_quad_") + str(i) + '.mat' 
        partial_data = sio.loadmat(filename)
        Q_FEM_quad += partial_data['Q_FEM'].reshape(int(400))

    def Phi(n):
        #define H_n
        coeffs = [0]*(n+1)
        coeffs[n] = 1
        return coeffs

    def q(i,j):
        x, w=H.hermegauss(20)     
        Q=sum([w[ldx]*sum([w[kdx] * Q_FEM_quad[ldx*20+kdx] * H.hermeval(x[kdx],Phi(i)) for kdx in range(20)])*H.hermeval(x[ldx],Phi(j)) for ldx in range(20)])     
        q= Q/(2*np.pi*factorial(i)*factorial(j))

        return q

    qij = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            qij[i,j] = q(i,j)

    def Q(n,x,y):
        result = 0 
        for i in range(n+1):
            for j in range(n+1):
                if i+j <=n:
                    result += qij[i,j]*Hermite_2d(i,j,(x-mu1)/sigma1,(y-mu2)/sigma2)
        return result

    def Qexact(x,y,a=0.4,b=0.6,c=0.4,d=0.6):
        sol = (np.cos(x*np.pi*a)-np.cos(x*np.pi*b))*(np.sin(y*np.pi*d)-np.sin(y*np.pi*c))/((b-a)*(d-c)*x*y*np.pi**2)
        return sol
    qexact = Qexact(lam1,lam2)
    pfprior_dens = kde(qexact)

    def pfprior_dens_n(n,x):
        pfprior_sample_n =  Q(n,lam1,lam2)
        pdf = kde(pfprior_sample_n)
        return pdf(x)
     
    def pdf_obs(x):
        return norm.pdf(x, loc=0.3, scale=0.1)

    r = np.zeros(int(1E4))    
    num_r_per_proc = int(1E4)//proc_max
    if proc_num != proc_size -1: 
        for i in range(proc_num*num_r_per_proc, (proc_num+1)*num_r_per_proc):
            if nn == 0:
                r[i] = pdf_obs(qexact[i])/pfprior_dens(qexact[i])
            else:
                q = Q(nn,lam1[i],lam2[i])
                r[i] = pdf_obs(q)/pfprior_dens_n(nn,q)
    elif proc_num == proc_size-1:
        for i in range(proc_num*num_r_per_proc,int(1E4)):
            if nn == 0:
                r[i] = pdf_obs(qexact[i])/pfprior_dens(qexact[i])
            else:
                q = Q(nn,lam1[i],lam2[i])
                r[i] = pdf_obs(q)/pfprior_dens_n(nn,q)
    if nn == 0:
        filename = os.path.join(os.getcwd(),"Data","r_") + str(proc_num) + '.mat' 
    else:
        filename = os.path.join(os.getcwd(),"Data","r") + str(nn) + '_' + str(proc_num) + '.mat' 
    data_dict = {'r': r}
    sio.savemat(filename, data_dict)
    return


#########################################################
######### Generate datafiles Data/r_[0-24].mat ##########
######### Generate datafiles Data/r1_[0-24].mat #########
######### Generate datafiles Data/r2_[0-24].mat #########
######### Generate datafiles Data/r3_[0-24].mat #########
######### Generate datafiles Data/r4_[0-24].mat #########
######### Generate datafiles Data/r5_[0-24].mat #########
######### Generate datafiles Data/r6_[0-24].mat #########
#########################################################
nn = [0,1,2,3,4,5,6]
for i in range(len(nn)):
    for j in range(proc_size):
        r(nn[i], j, proc_max=proc_size)