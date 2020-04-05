# 3d_reconstruction

Some functions for the reconstruction of 3D functions using probabilisic approach
## Dependencies
[The ASTRA Toolbox](https://www.astra-toolbox.com/)

## Functions
### - getBallData
  3D-array with zeros and ones generation. Ones are located in each point which satisfies (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 <= R^2
  
  Input params:
  
    N - number of elements in each dimention
    
    R - radius of ball
  
  Output params:
  
    data - 3D array NxNxN with generated values
    
    dt - discretization step
    
### - getAnalyticalIntegrals
  Plane integrals of the ball computation
  
  Input params:
  
    N - number of planes per the direction
    
    dt - discretisation step
    
    R - radius of ball
  
  Output params:
  
    res - array of integral values with size N
### - getVectorRepresentationOfAngle
  Calculation of angle vector representation which is required to ASTRA
  
  Input params:
  
    angle - (theta, phi) vector
  
  Output params:
    
    res - vector representation of vector with size = 12
### - getSinograms
  
  2D array of sinogram with ASTRA toolbox calculation
  
  Input params:
  
    data - 3D array NxNxN of data with (x,y,z) order
    
    N - number of elements in each dimention
    
    M - number of projectons per direction
    
    angles - 2D array N_anglesx12 of normal vector angles
    
    N_angles - number of angles
  
  Output params:
    
    sinograms - 3D array NxN_anglesxN of calculated sinograms
### - getRTmatrix
  
  3D Radon transform linear operator matrix calculation
  
  Input params:
  
    N - number of elements in each dimension
    
    M - number of projections per each direction
    
    angles - 2D array N_anglesx12 of normal vector angles
    
    N_angles - number of angles
  
  Output params:
  
  radon_matrix - 2D N_angles*N x N^3 array. 
  
### - getAstraReconstruction
  
  Reconstruction of original data based on ASTRA sinogram
  
  Input params:
    
    N - number of elements in each dimension of the original data
    
    sino_id - ID of sinogram in ASTRA memory 
    
    N_iter - number of reconstruction algorithm iterations
  
  Output params:
    
    data - reconstructed data
    
### - L
  
  Gaussian log-posterior function. 
  
  Here likelihood function is Gaussian and priors are also Gaussian
  
  Input params:
    
    matrix - matrix of linear operator
    
    x - generated element of chain
    
    y - calculated integrals (initial data) 
    
    sigma_lh - sigma of likelihood distribution
    
    sigma_noise - sigma of noise distribution
    
    sigma_priors - sigma of priors distribution
    
    N - numer of elements in each direction
    
    N_angles - number of angles
  
  Output params:
    
    res - log-posterior value 
    
### - MCMC_MH
  Metropolis-Hastings MCMC algorithm realization
  
  Input params:
    
    N_steps - number of steps
    
    N - numer of elements in each direction
    
    N_angles - number of angles
    
    N_burnin_period - number of burn-in elements
    
    prop_sigma - sigma of proposal distribution
    
    radon_matrix - matrix of Radon transform
    
    y - calculated integrals (initial data)
    
    init_value - initial element of chain

Output params:
    
    chain - Markov chain: 2D array (N_steps-N_burnin_period)xN^3
    
    ratio - rate of accepted chain elements 
    
### - printToFile
  Data to file in (x,y,z)-order printing
  
  Input params:
    
    filename - name of file with data
    
    data - data array (1D, 2D or 3D)
    
    Nx - number of elements with respect to Ox
    
    Ny - number of elements with respect to Oy
    
    Nz - number of elements with respect to Oz

Output params:
    
    flag - 1 - data is successfully printed, 0 - otherwise
