%
%   3D Radon transform linear operator matrix calculation
%   Input params:
%       N               number of elements in each dimention
%       M               number of projections per each direction
%       angles          2D array N_anglesx12 of normal vector angles
%       N_angles        number of angles
%   Output params:
%       radon_matrix    2D N_angles*M x N^3 array. 
%       
%       Action of calculated linear operator could be 
%       result = radon_matrix * data.' (data is N^3-array)
%

function radon_matrix = getRTmatrix(N, M, angles, N_angles)
    radon_matrix = zeros(N_angles*M, N*N*N);   
    columns = 1;
    for z = 1:N
        for y = 1:N
            for x = 1:N
                id_cube = zeros(N,N,N);
                id_cube(y,x,z) = 1;
                sino = getSinograms(id_cube, N, M, angles, N_angles);
                radon_matrix(:, columns) = reshape(sino,[M*N_angles,1]);
                columns = columns + 1;  
            end
        end
    end
end