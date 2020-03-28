%
%   2D array of sinogram with ASTRA toolbox calculation
%   Input params:
%       data        3D array NxNxN of data with (x,y,z) order
%       N           number of elements in each dimention
%       angles      2D array N_anglesx12 of normal vector angles
%       N_angles    number of angles
%   Output params:
%       sinograms    3D array NxN_anglesxN of calculated sinograms
%
function [sinograms, sinoID] = getSinograms(data, N, angles, N_angles, isNeedID)
    vol_geom = astra_create_vol_geom(N, N, N);
    proj_geom = astra_create_proj_geom('parallel3d_vec', N, N, angles);
    [sinoID, sino] = astra_create_sino3d_cuda(data, proj_geom, vol_geom);
    
    sinograms = zeros(N, N_angles);
    for directions=1:N_angles
        for planes=1:N
            sinograms(planes, directions) = sum(sino(:, directions, planes));
        end
    end
    if (nargin == 4 || isNeedID == 0)        
        astra_mex_data3d('delete', sinoID);
    end
    
end