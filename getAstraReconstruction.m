%
%   Reconstruction of original data based on ASTRA sinogram
%   Input params:
%       N           number of reconstructed elements in each dimension of the
%                   original data
%       sino_id     ID of sinogram in ASTRA memory 
%       N_iter      number of reconstruction algorithm iterations
%   Output params:
%       data        reconstructed data
%

function data = getAstraReconstruction(sino_id, N, N_iter)
    vol_geom = astra_create_vol_geom(N, N, N);    
    rec_id = astra_mex_data3d('create', '-vol', vol_geom);
    
    cfg = astra_struct('SIRT3D_CUDA');
    cfg.ReconstructionDataId = rec_id;
    cfg.ProjectionDataId = sino_id;

    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('iterate', alg_id, N_iter);

    data = astra_mex_data3d('get', rec_id);
    astra_mex_algorithm('delete', alg_id);
    astra_mex_data3d('delete', rec_id);
    %astra_mex_data3d('delete', sino_id);
    
end