%
%   3D Radon transform linear operator sparse matrix parallel calculation
%   Input params:
%       N               number of elements in each dimension
%       M               number of projections per each direction
%       angles          2D array N_anglesx12 of normal vector angles
%       N_angles        number of angles
%   Output params:
%       radon_matrix    2D N_angles*M x N^3 sparse matrix. 
%       
%       Action of calculated linear operator could be 
%       result = radon_matrix * data.' (data is N^3-array)
%

function radon_matrix = getRTmatrix(N, M, angles, N_angles)
    m = N_angles*M;
    n = N^3;
    num_nz = 1/N;
    radon_matrix = spalloc(m, n, round(m*n*num_nz));
    
    parfor z = 1:n   
        id_cube = zeros(1,n);
        id_cube(1,z) = 1;
        id_cube = reshape(id_cube, [N,N,N]);
        sino = reshape(getSinograms(id_cube, N, M, angles, N_angles), [M*N_angles,1]);
		radon_matrix(:, z) = sparse(sino); 
        if (mod(z, 1000000) == 0)
            disp(z);
        end
    end
end