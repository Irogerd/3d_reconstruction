%
%   3D Radon transform linear operator matrix calculation
%   Input params:
%       N               number of elements in each dimention
%       angles          2D array N_anglesx12 of normal vector angles
%       N_angles        number of angles
%   Output params:
%       radon_matrix    2D N_angles*N x N^3 array. 
%       
%       Action of calculated linear operator could be 
%       result = radon_matrix * data.' (data is N^3-array)
%

function radon_matrix = getRTmatrix(N, angles, N_angles)
%     vol_geom = astra_create_vol_geom(N, N, N);
%     proj_geom = astra_create_proj_geom('parallel3d_vec', N, N, angles);
    
    radon_matrix = zeros(N_angles*N, N*N*N);   
    columns = 1;
    for z = 1:N
        for y = 1:N
            for x = 1:N
                id_cube = zeros(N,N,N);
                id_cube(y,x,z) = 1;
                sino = getSinograms(id_cube, N, angles, N_angles);
                radon_matrix(:, columns) = reshape(sino,[N*N_angles,1]);
                columns = columns + 1;  
            end
        end
    end
end

%                 [sinoID, sino] = astra_create_sino3d_cuda(id_cube, proj_geom, vol_geom);
%                 astra_mex_data3d('delete', sinoID);
%                 row_counter = 1;
%                 for a_num = 1:N_angles
%                     for planes = 1:N
%                         radon_matrix(row_counter, counter) = sum(sinoID(:,a_num,planes));
%                         row_counter = row_counter + 1;
%                     end
%                 end
%                 counter = counter + 1;    