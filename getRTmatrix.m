%
%   3D Radon transform linear operator sparse matrix calculation
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
    radon_matrix = spalloc(m, n, round(m*n*0.05));
    
    columns = 1;
    for z = 1:N
        for y = 1:N
            for x = 1:N
                id_cube = zeros(N,N,N);
                id_cube(x,y,z) = 1;
                sino = getSinograms(id_cube, N, M, angles, N_angles);
                radon_matrix(:,columns) = sparse(reshape(sino,[M*N_angles,1]));
                columns = columns + 1;  
            end
        end
    end
end