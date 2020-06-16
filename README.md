# 3d_reconstruction

There are some functions for the reconstruction of the 3D function p=f(x,y,z) from its plane integration (formally, three-dimensional Radon transform application) vaules using probabilisic approach. Data preparation, Radon transform matrix calculation, projections calculations (forward problem) are implemented using MATLAB. MAP estimations (inverse problem) with different priors are implemented using Python.

## Dependencies
### Matlab
- [The ASTRA Toolbox](https://www.astra-toolbox.com/)
### Python
- Numpy
- Scipy (minimize, io, sparse)
- Numba (note that the code was tested using Numba 0.48, one need to be carefull using another versions)
- h5py

## Python functions
### - set_rt_matrix
  Uploads Radon transform matrix which was precalculated using Matlab from .mat files and saves it in memory. One need to call this function before MAP estimation
  
  Input params:
  
    filename - name of .mat file with Radon transform matrix. For instance, "RM15.mat"

### - set_projections
  Uploads projection values which was precalculated using Matlab from binary file. One need to call this function before MAP estimation

  Input params:
  
    filename - name of binary file with projection values    
    M - number of planes per one direction    
    K - number of directions    
    p = 0 - should projection values and shape of data be printed or not    
    sigma_noise = 1 - standard deviation of noise
    
### - get_MAP_gaussian_1
  Calculates MAP estimation using first order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    sigma - standard deviation value for covariance matrix    
    sigma_priors - standard deviation value for gaussian priors    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    isPos = 0 - boolean value which defines domain of the unknown. isPos = 1 => X_i >= 0 \forall i, X_i \in R otherwise. 
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data

### - get_MAP_gaussian_1_log
  Calculates MAP estimation using log-transformed first order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    c_log - constant in logarithm to avoid zero value as its argument
    sigma - standard deviation value for covariance matrix    
    sigma_priors - standard deviation value for gaussian priors    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data
    
### - get_MAP_gaussian_2
  Calculates MAP estimation using second order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    sigma - standard deviation value for covariance matrix    
    sigma_priors - standard deviation value for gaussian priors    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    isPos = 0 - boolean value which defines domain of the unknown. isPos = 1 => X_i >= 0 \forall i, X_i \in R otherwise. 
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data
    
### - get_MAP_gaussian_2_log
  Calculates MAP estimation using log-transformed second order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    c_log - constant in logarithm to avoid zero value as its argument
    sigma - standard deviation value for covariance matrix    
    sigma_priors - standard deviation value for gaussian priors    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data
    
### - get_MAP_gaussian_1_2
  Calculates MAP estimation using the sum of the first and the second order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    sigma - standard deviation value for covariance matrix    
    sigma_priors_1 - standard deviation value for the first order priors inside the domain
    sigma_priors_2 - standard deviation value  for the second order priors inside the domain    
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    isPos = 0 - boolean value which defines domain of the unknown. isPos = 1 => X_i >= 0 \forall i, X_i \in R otherwise. 
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data
    
### - get_MAP_gaussian_1_2_log
  Calculates MAP estimation using log-transformed second order differences Gaussian priors
  
  Input params:
    
    N_elem - number of elements of reconstructed data per each axe    
    M - number of planes per one direction    
    K - number of directions    
    c_log - constant in logarithm to avoid zero value as its argument
    sigma - standard deviation value for covariance matrix    
    sigma_priors_1 - standard deviation value for the first order priors inside the domain
    sigma_priors_2 - standard deviation value  for the second order priors inside the domain   
    sigma_bound - standard deviation value for boundary voxels for gaussian priors
    
  Output params
    
    X - one dimensional array (length = N^3) with reconstructed data


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
    
    X - one dimensional array (length = N^3) with reconstructed data

## Matlab functions
### - getBallData
  3D-array with zeros and ones generation. Ones are located in each point which satisfies (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 <= R^2
  
  Input params:
  
    N - number of elements in each dimension    
    R - radius of the ball
  
  Output params:
  
    data - 3D array NxNxN with generated values    
    dt - discretization step
    
### - getComplexBallData
  3D-array with the ball which has ellipsoid, parallelepiped and two balls inside. 
  Input params:
  
    N - number of elements in each dimension    
    R - radius of the ball
  
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
  Calculates set of angles which define directions. This set is Cartesian product of number of theta angles set located between 1 and 179 degrees and phi angles set located between 0 and 179 degrees
  
  Input params:
  
    N_theta - number of theta angles
    N_phi - number of phi angles
  
  Output params:
  
    astra_angles - set of angles in its the ASTRA Toolbox representation 
    deg_angles - set of angles in its degrees representation

### - getVectorRepresentationOfAngle
  Calculation of angle vector representation which is required to the ASTRA Toolbox (see its documentation)
  
  Input params:
  
    angle - (theta, phi) vector
  
  Output params:
    
    res - vector representation of vector with size = 12
### - getSinograms
  
  2D array of sinogram with ASTRA toolbox calculation
  
  Input params:
  
    data - 3D array NxNxN of data with (x,y,z) order    
    N - number of elements in each dimension    
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
## How to prepare data using Matlab?
1. Initialize main values:
- N = number of elements per one axe (grid size)
- K = number of direction
- M = number of projections per one direction
2. Initialize directions:
- you cound initialize directions manually using function getVectorRepresentationOfAngles
- or using function getAngles(N_theta, N_phi), where N_theta and N_phi are numbers of theta and phi angles respectively
3. Generate data using getBallData or getComplexBallData functions for the test cases or initialize by any other 3D data
4. Obtain projections with function getSinograms (and add some noise to calculated projections if necessary)
5. Save projections to the file using printToFile
6. Obtain Radon transform matrix using function getRTmatrix (computational time depends on grid size, may requires a lot of time). It is important to note that this matrix must be saved using "save" Matlab' command with version flag "-v7.3". This matrix will be saved as .mat file. The flag is required because the matrix would be too large.

### How to reconstruct data using Python?
1. Import module "reconstruction"
2. Call function set_rt_matrix and set the matrix previously saved to .mat file
3. Call function set_projections and set calculated projections previously saved to binary file
4. Call some of the reconstruction functions, for instance, get_MAP_gaussian_1(log-posterior with Cauchy priors) to calculate desired reconstruction
