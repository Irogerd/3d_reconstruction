import numpy as np
from scipy.optimize import minimize
import scipy.io as sio
from scipy.sparse import csc_matrix

index = lambda x,y,z : x + N*y + N**2*z
rm = []
rm_transp = []
y = []
cov_matrix = []
inv_cov_matrix = []
N = 0
c_prior = 0
c_bound = 0


def MAP_gaussian(x):
    
    # likelihood:
    res = 0
    yAx = y - rm.dot(x) # (y-Ax)
    yAx_transp = np.transpose(yAx)
    t = np.matmul(inv_cov_matrix, yAx_transp) #s^-1(y-Ax)
    res = 1/2 * np.matmul(yAx, t)
    # priors
    for k in range(N):
        for j in range(N):
            for i in range(N):
                
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    t = (x[index(i,j,k)] - x[index(i-1,j,k)])**2 + (x[index(i,j,k)] - x[index(i,j-1,k)])**2 + (x[index(i,j,k)] - x[index(i,j,k-1)])**2
                    res += c_prior * t
                else:
                    t = x[index(i,j,k)]**2
                    res += c_bound * t

    return res

def gradient_gaussian(x):
    
    #likelihood derivative
    yAx = y - rm.dot(x) # (y-Ax)
    yAx = np.transpose(yAx)
    syAX = np.matmul(inv_cov_matrix, yAx)
    res = -rm_transp.dot(syAX) 
    
    for k in range(N):
        for j in range(N):
            for i in range(N):
                # boundary point or not
                if (i > 0 and i < N-1 and j > 0 and j < N-1 and k > 0 and k < N-1): # inside of domain
                    t = 2*(x[index(i,j,k)] - x[index(i-1,j,k)]) + 2*(x[index(i,j,k)]-x[index(i,j-1,k)]) + 2*(x[index(i,j,k)] - x[index(i,j,k-1)])
                    if (i+1 < N-1):
                        t -= 2*(x[index(i+1,j,k)] - x[index(i,j,k)]) 
                    if (j+1 < N-1):
                        t -= 2*(x[index(i,j+1,k)] - x[index(i,j,k)]) 
                    if (k+1 < N-1):
                        t -= 2*(x[index(i,j,k+1)] - x[index(i,j,k)])
                    res[index(i,j,k)] += c_prior * t
                    
                else:
                    res[index(i,j,k)] += c_bound*2*x[index(i,j,k)]
    return res 
	
def setRTmatrix(filename):
	global rm 
	rm = sio.loadmat(filename)['rm']
	global rm_transp 
	rm_transp = csc_matrix.transpose(rm)
	print("Size of Radon transform matrix: ", rm.data.nbytes/1024/1024, " bytes")

def setProjections(filename, M, K, p = 0, sigma_noise = 1):
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

def getMAP_gaussian(N_elem, M, K, sigma, sigma_priors, sigma_bound):
	global N 
	N = N_elem
	init = np.ones(N**3)
	global cov_matrix
	cov_matrix = sigma * np.eye(M*K)
	global inv_cov_matrix 
	inv_cov_matrix = np.linalg.inv(cov_matrix)
	global c_prior 
	c_prior = 1/(sigma_priors**2)
	global c_bound 
	c_bound = 1/(sigma_bound**2)
	
	res = minimize(MAP_gaussian, init, method='L-BFGS-B', jac=gradient_gaussian, options={'disp': True})
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

	
