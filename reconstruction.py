import numpy as np
from scipy.optimize import minimize
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import eye as sparseid
from numba import njit, prange
import h5py

rm = [] # Radon transform matrix
rm_transp = [] # Transposed Radon transform matrix
y = [] # projection values
cov_matrix = [] # Covariance matrix for likelihood function
inv_cov_matrix = [] # Inverse matrix for covariance matrix

N = 0 # Number of elements per one axe

# For Gaussian priors
c_prior = 0
c_bound = 0

# For log-transformation
c = 0

# For Cauchy priors
gamma = 0

@njit(parallel=True)
def index(x,y,z):
    return x+N*y+N**2*z

# Initializes Radon transform matrix from the .mat-file and also prints size of matrix in bytes
# Input params:
#   filename - name of .mat-file with RT matrix with format "filename.mat"
def set_rt_matrix(filename):
    global rm 
    #rm = sio.loadmat(filename)['rm']
    f = h5py.File(filename, 'r')
    rm = csc_matrix((f["rm"]["data"], f["rm"]["ir"], f["rm"]["jc"]))
    global rm_transp 
    rm_transp = csc_matrix.transpose(rm)
    print("Size of Radon transform matrix: ", rm.data.nbytes/1024/1024, " MB")
    f.close()

# Initializes projection values from the binary file, prints signal-to-noise ratio and optionally prints projections values and shape of array
# Input params:
#   filename - name of binary file with projection values
#   M - number of projections per one direction
#   K - number of directions
#   p - will projection values and its shape be printed or not
#   sigma_noise - standard deviation value of noise (required for signal-to-noise ratio calculation)
def set_projections(filename, M, K, p = 0, sigma_noise = 1):
    proj = []
    with open(filename) as f:
        for line in f:
            proj.append([float(x) for x in line.split()])
    proj = np.reshape(proj, (1, M*K))
    if (p != 0):
        print(proj.shape)
        print(proj)
    print("SNR = ", np.mean(proj)/sigma_noise)
    global y 
    y = proj


# ========== GAUSSIAN PRIORS  ==========
# Gaussian priors calculation in parallel loop and sum with likelihood value.
# Function is called from gaussian_logpost function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood value
#   x - current value of x vector 
# Outputt params:
#   res - real value equal to sum of priors values and likelihood
@njit(parallel=True)
def add_gaussian_priors(cur_res, x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):                
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    t = (x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])**2
                    res += c_prior * t
                else: #boundary point
                    t = x[i + N*j + N**2*k]**2
                    res += c_bound * t

    return res

# Log-posterior function calculation with Gaussian priors.
# Function is called from get_MAP_gaussian function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
# Output params:
#   res - real value equals to log_posterior function value
def gaussian_logpost(x):
    # likelihood:
    res = 0
    yAx = y - rm.dot(x) # (y-Ax)
    yAx_transp = np.transpose(yAx)
    t = np.matmul(inv_cov_matrix, yAx_transp) #s^-1(y-Ax)
    res = 1/2 * np.matmul(yAx, t)

    # priors
    res = add_gaussian_priors(res,x)
    return res

# Gaussian priors derivatives calculation in parallel loop.
# Function is called from gaussian_logpost_gradient function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood derivative value
#   x - current value of x vector 
# Output params:
#   res - vector of sums of prior derivatives values and likelihood derivatives values
@njit(parallel=True)
def add_gaussian_gradient_priors(cur_res,x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    t = 2*(x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k]) + 2*(x[i + N*j + N**2*k]-x[i + N*(j-1) + N**2*k]) + 2*(x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])
                    if (i+1 < N-1):
                        t -= 2*(x[i+1 + N*j + N**2*k] - x[i + N*j + N**2*k]) 
                    if (j+1 < N-1):
                        t -= 2*(x[i + N*(j+1) + N**2*k] - x[i + N*j + N**2*k]) 
                    if (k+1 < N-1):
                        t -= 2*(x[i + N*j + N**2*(k+1)] - x[i + N*j + N**2*k])
                    res[i+N*j+N**2*k] += c_prior * t
                    
                else:
                    res[i + N*j + N**2*k] += c_bound*2*x[i + N*j + N**2*k]
    return res

# Gaussian log-posterior function calculation.
# Function is called from getMAP_gaussian function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
def gaussian_logpost_gradient(x):
    #likelihood derivative
    yAx = y - rm.dot(x) # (y-Ax)
    yAx = np.transpose(yAx)
    syAX = np.matmul(inv_cov_matrix, yAx)
    res = -rm_transp.dot(syAX) 
    
    res = add_gaussian_gradient_priors(res,x)
    return res 

# Maximum a posteriori with Gaussian priors calculation
# Input params:
#   N_elem - N
#   M - number of projections per one direction
#   K - number of directions 
#   sigma - standard deviation value for likelihood
#   sigma_priors - standard deviation value for priors inside the domain
#   sigma_bound - standard deviation value for priors for boundary voxels
def get_MAP_gaussian(N_elem, M, K, sigma, sigma_priors, sigma_bound):
    if (rm == []):
        print("One have to initialize Radon transform matrix. Use setRTmatrix function")
        return 1
    if (y == []):
        print("One have to initialize projection values. Use setProjections function")
        return 2

    global N 
    N = N_elem
    init = np.ones(N**3)
    global cov_matrix
    cov_matrix = sigma**2 * np.eye(M*K)
    global inv_cov_matrix 
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    global c_prior 
    c_prior = 1/(sigma_priors**2)
    global c_bound 
    c_bound = 1/(sigma_bound**2)
    
    res = minimize(gaussian_logpost, init, method='L-BFGS-B', jac=gaussian_logpost_gradient, options={'disp': True})
    return res.x


# ========== GAUSSIAN PRIORS WITH LOG-TRANSFORMATION  ========== 
# Gaussian priors with log-transform calculation in parallel loop and sum with likelihood value.
# Function is called from gaussian_logpost_log function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood value
#   x - current value of x vector 
# Outputt params:
#   res - real value equal to sum of priors values and likelihood
@njit(parallel=True)
def add_gaussian_priors_log(cur_res, x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):                
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    t = (np.log(x[i+N*j+N**2*k] + c) - np.log(x[i-1+N*j+N**2*k] + c))**2 + (np.log(x[i+N*j+N**2*k] + c) - np.log(x[i+N*(j-1)+N**2*k] + c))**2 + (np.log(x[i+N*j+N**2*k] + c) - np.log(x[i+N*j+N**2*(k-1)] + c))**2
                    res += c_prior * t
                else: #boundary point
                    t = np.log(x[i + N*j + N**2*k]+c)**2
                    res += c_bound * t

    return res

# Log-posterior function calculation with log-transformed Gaussian priors.
# Function is called from get_MAP_gaussian_log function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
# Output params:
#   res - real value equals to log_posterior function value
def gaussian_logpost_log(x):
    # likelihood:
    res = 0
    yAx = y - rm.dot(x) # (y-Ax)
    yAx_transp = np.transpose(yAx)
    t = np.matmul(inv_cov_matrix, yAx_transp) #s^-1(y-Ax)
    res = 1/2 * np.matmul(yAx, t)

    # priors
    res = add_gaussian_priors_log(res,x)
    return res

# Gaussian log-transformed priors derivatives calculation in parallel loop.
# Function is called from gaussian_logpost_gradient function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood derivative value
#   x - current value of x vector 
# Output params:
#   res - vector of sums of prior derivatives values and likelihood derivatives values
@njit(parallel=True)
def add_gaussian_gradient_priors_log(cur_res,x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    #t = 2/(x[i+N*j+N**2*k]+c) * ( (np.log(x[i+N*j+N**2*k]+c) - np.log(x[i-1+N*j+N**2*k]+c)) + (np.log(x[i+N*j+N**2*k]+c) - np.log(x[i+N*(j-1)+N**2*k]+c)) + (np.log(x[i+N*j+N**2*k]+c) - np.log(x[i+N*j+N**2*(k-1)]+c)) )
                    t = 2 * (np.log(x[i+N*j+N**2*k] + c) - np.log(x[(i-1)+N*j+N**2*k] + c))/(x[i+N*j+N**2*k] + c)
                    t += 2 * (np.log(x[i+N*j+N**2*k] + c) - np.log(x[i+N*(j-1)+N**2*k] + c))/(x[i+N*j+N**2*k] + c)
                    t += 2 * (np.log(x[i+N*j+N**2*k] + c) - np.log(x[i+N*j+N**2*(k-1)] + c))/(x[i+N*j+N**2*k] + c)
                    if (i+1 < N-1):
                        #t -= 2/(x[i + N*j + N**2*k]+c) * ( np.log(x[i+1 + N*j + N**2*k]+c) - np.log(x[i + N*j + N**2*k]+c) ) 
                        t -= 2 * (np.log(x[(i+1)+N*j+N**2*k] + c) - np.log(x[i+N*j+N**2*k] + c)) / (x[i+N*j+N**2*k] + c)
                    if (j+1 < N-1):
                        #t -= 2/(x[i + N*j + N**2*k]+c) * ( np.log(x[i + N*(j+1) + N**2*k]+c) - np.log(x[i + N*j + N**2*k]+c) ) 
                        t -= 2 * (np.log(x[i+N*(j+1)+N**2*k] + c) - np.log(x[i+N*j+N**2*k] + c)) / (x[i+N*j+N**2*k] + c)
                    if (k+1 < N-1):
                        #t -= 2/(x[i + N*j + N**2*k]+c) * ( np.log(x[i + N*j + N**2*(k+1)]+c) - np.log(x[i + N*j + N**2*k]+c) ) 
                        t -= 2 * (np.log(x[i+N*j+N**2*(k+1)] + c) - np.log(x[i+N*j+N**2*k] + c)) / (x[i+N*j+N**2*k] + c)
                    res[i+N*j+N**2*k] += c_prior * t
                    
                else:
                    res[i + N*j + N**2*k] += c_bound* 2/(x[i + N*j + N**2*k]+c) * np.log(x[i + N*j + N**2*k]+c)
    return res

# Gaussian log-posterior gradient calculation.
# Function is called from get_MAP_gaussian_log function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
def gaussian_logpost_gradient_log(x):
    #likelihood derivative
    yAx = y - rm.dot(x) # (y-Ax)
    yAx = np.transpose(yAx)
    syAX = np.matmul(inv_cov_matrix, yAx)
    res = -rm_transp.dot(syAX) 
    
    res = add_gaussian_gradient_priors_log(res,x)
    return res 

# Maximum a posteriori with Gaussian log-transformed priors calculation
# Input params:
#   N_elem - N
#   M - number of projections per one direction
#   K - number of directions 
#   c_log - constant in logarithm to avoid zero value as its argument
#   sigma - standard deviation value for likelihood
#   sigma_priors - standard deviation value for priors inside the domain
#   sigma_bound - standard deviation value for priors for boundary voxels
def get_MAP_gaussian_log(N_elem, M, K, c_log, sigma, sigma_priors, sigma_bound):
    if (rm == []):
        print("One have to initialize Radon transform matrix. Use setRTmatrix function")
        return 1
    if (y == []):
        print("One have to initialize projection values. Use setProjections function")
        return 2

    global N 
    N = N_elem
    init = np.ones(N**3)
    global cov_matrix
    cov_matrix = sigma**2 * np.eye(M*K)
    global inv_cov_matrix 
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    global c_prior 
    c_prior = 1/(sigma_priors**2)
    global c_bound 
    c_bound = 1/(sigma_bound**2)
    global c
    c = c_log;
    
    res = minimize(gaussian_logpost_log, init, method='L-BFGS-B', jac=gaussian_logpost_gradient_log, options={'disp': True})
    return res.x



# ========== CAUCHY PRIORS  ==========
# Cauchy priors calculation in parallel loop and sum with likelihood value.
# Function is called from cauchy_logpost function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood value
#   x - current value of x vector 
# Outputt params:
#   res - real value equal to sum of priors values and likelihood
@njit(parallel=True)
def add_cauchy_priors(cur_res, x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):
                
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    #res -= np.log(gamma/((x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])**2 + gamma**2) * gamma/((x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])**2 + gamma**2) * gamma/((x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])**2 + gamma**2))
                    res -= np.log(gamma/((x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])**2 + gamma**2)**2)
                else:
                    res -= np.log(gamma/((x[i + N*j + N**2*k])**2 + gamma**2))
    return res

# Log-posterior function calculation with Cauchy priors.
# Function is called from get_MAP_cauchy function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
# Output params:
#   res - real value equals to log_posterior function value
def cauchy_logpost(x):
    # likelihood:
    res = 0
    yAx = y - rm.dot(x) # (y-Ax)
    yAx_transp = np.transpose(yAx)
    t = np.matmul(inv_cov_matrix, yAx_transp) #s^-1(y-Ax)
    res = 1/2 * np.matmul(yAx, t)

    # priors
    res = add_cauchy_priors(res, x)
    return res

# Cauchy priors derivatives calculation in parallel loop and sum with likelihood.
# Function is called from cauchy_logpost_gradient function. Shouldn't be called from another functions
# Input params:
#   cur_res - previously calculated likelihood derivative value
#   x - current value of x vector 
# Output params:
#   res - vector of sums of prior derivatives values and likelihood derivatives values
@njit(parallel=True)
def add_cauchy_gradient_priors(cur_res,x):
    res = cur_res
    for k in prange(N):
        for j in range(N):
            for i in range(N):
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    
                    #t = 2*(x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])/((x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])**2 + gamma**2)
                    #t += 2*(x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])/((x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])**2 + gamma**2)
                    #t += 2*(x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])/((x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])**2 + gamma**2)
                    t = 4*((x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k]) + (x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k]) + (x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)]))/((x[i + N*j + N**2*k] - x[i-1 + N*j + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*(j-1) + N**2*k])**2 + (x[i + N*j + N**2*k] - x[i + N*j + N**2*(k-1)])**2 + gamma**2)

                    if (i+1 < N-1):
                        #t -= 2*(x[i+1 + N*j + N**2*k] - x[i + N*j + N**2*k])/((x[i+1 + N*j + N**2*k] - x[i + N*j + N**2*k])**2 + gamma**2) 
                        t -= 4*(x[i+1 + N*j + N**2*k] - x[i + N*j + N**2*k])/((x[i+1 + N*j + N**2*k] - x[i+1 + N*(j-1) + N**2*k])**2 + (x[i+1 + N*j + N**2*k] - x[i+1 + N*j + N**2*(k-1)])**2 + (x[i+1 + N*j + N**2*k] - x[i + N*j + N**2*k])**2 + gamma**2)
                    if (j+1 < N-1):
                        #t -= 2*(x[i + N*(j+1) + N**2*k] - x[i + N*j + N**2*k])/((x[i + N*(j+1) + N**2*k] - x[i + N*j + N**2*k])**2 + gamma**2)  
                        t -= 4*(x[i + N*(j+1) + N**2*k] - x[i + N*j + N**2*k])/((x[i + N*(j+1) + N**2*k] - x[i-1 + N*(j+1) + N**2*k])**2 + (x[i + N*(j+1) + N**2*k] - x[i + N*(j+1) + N**2*(k-1)])**2 + (x[i + N*(j+1) + N**2*k] - x[i + N*j + N**2*k])**2 + gamma**2)
                    if (k+1 < N-1):
                        #t -= 2*(x[i + N*j + N**2*(k+1)] - x[i + N*j + N**2*k])/((x[i + N*j + N**2*(k+1)] - x[i + N*j + N**2*k])**2 + gamma**2) 
                        t -= 4*(x[i + N*j + N**2*(k+1)] - x[i + N*j + N**2*k])/((x[i + N*j + N**2*(k+1)] - x[i-1 + N*j + N**2*(k+1)])**2 + (x[i + N*j + N**2*(k+1)] - x[i + N*(j-1) + N**2*(k+1)])**2 + (x[i + N*j + N**2*(k+1)] - x[i + N*j + N**2*k])**2 + gamma**2)
                    res[i + N*j + N**2*k] += t
                    
                else:
                    res[i + N*j + N**2*k] += 2*x[i + N*j + N**2*k]/(x[i + N*j + N**2*k]**2+gamma**2)
    return res

# Cauchy log-posterior function calculation.
# Function is called from get_MAP_cauchy function. Shouldn't be called from another functions
# Input params:
#   x - current value of x vector 
def cauchy_logpost_gradient(x):
    #likelihood derivative
    yAx = y - rm.dot(x) # (y-Ax)
    yAx = np.transpose(yAx)
    syAX = np.matmul(inv_cov_matrix, yAx)
    res = -rm_transp.dot(syAX) 

    res = add_cauchy_gradient_priors(res,x)
    return res 

# Maximum a posteriori with Gaussian priors calculation
# Input params:
#   N_elem - N
#   M - number of projections per one direction
#   K - number of directions 
#   sigma - standard deviation value for likelihood
#   h - discretization step
#   lmbd - lambda value for Cauchy priors
def get_MAP_cauchy(N_elem, M, K, sigma, h, lmbd):
    if (rm == []):
        print("One have to initialize Radon transform matrix. Use setRTmatrix function")
        return 1
    if (y == []):
        print("One have to initialize projection values. Use setProjections function")
        return 2

    global gamma
    gamma = h*lmbd

    global N
    N = N_elem

    global cov_matrix
    cov_matrix = sigma * np.eye(M*K)

    global inv_cov_matrix 
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    init = np.ones(N**3)

    res = minimize(cauchy_logpost, init, method='L-BFGS-B', jac=cauchy_logpost_gradient, options={'disp': True})
    return res.x


def clearAll():
    del index
    del init
    del rm
    del rm_transp
    del y
    del cov_matrix
    del inv_cov_matrix
    del N
    del c_prior
    del c_bound

    
