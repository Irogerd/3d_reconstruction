%
%   Data to file in (x,y,z)-order printing
%   Input params:
%       filename    name of file with data
%       data        data array (1D, 2D or 3D)
%       Nx          number of elements with respect to Ox
%       Ny          number of elements with respect to Oy
%       Nz          number of elements with respect to Oz
%   Output params:
%       flag        1 - data is successfully printed, 0 - otherwise
%
%
function flag = printToFile(filename, data, Nx, Ny, Nz)
    
    fID = fopen(filename, 'wt');
    if (nargin == 3)
        flag = 1;
        for x = 1:Nx
            fprintf(fID, '%.12f ', data(x));
        end
    end
    if (nargin == 4)
        flag = 1;
        for y = 1:Ny
            for x = 1:Nx
                fprintf(fID, '%.12f ', data(x,y));
            end
        end
    end
    if (nargin == 5)
        for z = 1:Nz
            for y = 1:Ny
                for x = 1:Nx
                    fprintf(fID, '%.12f ', data(x,y,z));
                end
            end
        end
    end
    if (nargin < 3 || nargin > 5)
        flag = 0;
        disp('printToFile: Incorrect number of arguments');
    end
    
    fclose(fID);    
end