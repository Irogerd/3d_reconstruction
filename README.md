# 3d_reconstruction

Some functions for the reconstruction of 3D functions using probabilisic approach. Data preparation, Radon transform matrix calculation, projections calculations are implemented using Matlab. MAP estimation is implemented using Python.

## Matlab dependencies
- [The ASTRA Toolbox](https://www.astra-toolbox.com/)
## Python dependencies
- Numpy
- Scipy (minimize, io, sparse)
- Numba

## Python functions
### - set_rt_matrix
  Uploads Radon transform matrix which was precalculated using Matlab from .mat files and saves it in memory. One need to call this function before MAP estimation
  
  Input params:
  
    filename - name of .mat file. For instance, "RM15.mat"

### - set_projections
  Uploads projection values which was precalculated using Matlab from binary file. One need to call this function before MAP estimation

  Input params:
  
    filename - name of binary file with projection values    
    M - number of planes per one direction    
    K - number of directions    
    p = 0 - should projection values and shape of data be printed or not    
    sigma_noise = 1 - standard deviation of noise
    
### - get_MAP_gaussian
  Calculates MAP estimation using Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    sigma - standard deviation value for covariance matrix    
    sigma_priors - standard deviation value for gaussian priors    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    
  Output params
    
    X - one dimentional array (length = N^3) with reconstructed data
### get_MAP_cauchy
  Calculates MAP estimation using Cauchy priors
  
  Input params:
    
    N_elem - N
    M - number of projections per one direction
    K - number of directions 
    sigma - standard deviation value for likelihood
    h - discretization step
    lmbd - lambda value for Cauchy priors
    
  Output params
    
    X - one dimentional array (length = N^3) with reconstructed data
    
## Matlab functions
### - getBallData
  3D-array with zeros and ones generation. Ones are located in each point which satisfies (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 <= R^2
  
  Input params:
  
    N - number of elements in each dimension    
    R - radius of ball
  
  Output params:
  
    data - 3D array NxNxN with generated values    
    dt - discretization step
    
### - getComplexBallData
  3D-array with the ball which has ellipsoid, parallelepiped and two balls inside. 
  Input params:
  
    N - number of elements in each dimension    
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
    
### - getAngles
  Calculates set of angles which define directions. This set is Cartesian product of number of theta angles set located between 1 and 90 degrees and phi angles set located between 0 and 179 degrees
  
  Input params:
  
    N_theta - number of theta angles
    N_phi - number of phi angles
  
  Output params:
  
    astra_angles - set of angles in its the ASTRA Toolbox representation 
    deg_angles - set of angles in its degrees representation

### - getVectorRepresentationOfAngle
  Calculation of angle vector representation which is required to the ASTRA Toolbox
  
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
  
  3D Radon transform linear operator sparse matrix calculation
  
  Input params:
  
    N - number of elements in each dimension    
    M - number of projections per each direction    
    angles - 2D array N_anglesx12 of normal vector angles    
    N_angles - number of angles
  
  Output params:
  
    radon_matrix - 2D N_angles*M x N^3 sparse matrix 
  
### - getAstraReconstruction
  
  Reconstruction of original data based on ASTRA sinogram
  
  Input params:
    
    N - number of elements in each dimension of the original data    
    sino_id - ID of sinogram in ASTRA memory     
    N_iter - number of reconstruction algorithm iterations
  
  Output params:
    
    data - reconstructed data
    
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


## Examples
### How to prepare data using Matlab?
1. Initialize main values:
- N = number of elements per one axe (grid size)
- K = number of direction
- M = number of projections per one direction
2. Initialize directions using function getAngles(N_theta, N_phi), where N_theta and N_phi are numbers of theta and phi angles respectively
3. Generate data using getBallData or getComplexBallData functions
4a. Obtain projections with function getSinograms
4b. Add some noise to calculated projections if neccesary
5. Save projections to the file using printToFile
6. Obtain Radon transform matrix using function getRTmatrix (computational time depends on grid size, may requires a lot of time). It is important to note that this matrix must be saved using "save" Matlab' command. This matrix will be saved as .mat file

### How to reconstruct data using Python?
1. Import module "reconstruction"
2. Call function set_rt_matrix and set the matrix previously saved to .mat file
3. Call function set_projections and set calculated projections saved to binary file
4. Call get_MAP_gaussian (log-posterior with Gaussian priors) or get_MAP_cauchy (log-posterior with Cauchy priors) to calculate desired reconstruction
