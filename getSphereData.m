%
%   3D-array with zeros and ones generation. Ones are located in each point
%   which satisfies (x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 <= R^2
%   Input params:
%       N           number of elements in each dimention
%       R           radius of sphere
%       xyz_center  sphere center coordinate (x_c = y_c = z_c = xyz_cener)
%   Output params:
%       data    	3D array NxNxN with generated values
%
function data = getSphereData(N, R, xyz_center)
    data = zeros(N,N,N);
    for z = 1:xyz_center % here the idea of symmetry is used
        for y = 1:xyz_center
            for x = 1:xyz_center
                if ((x - xyz_center)^2 + (y - xyz_center)^2 + (z - xyz_center)^2 <= R^2)
                    data(x,y,z) = 1;
                    data(N-x+1,y,z) = 1;
                    data(x,N-y+1,z) = 1;
                    data(N-x+1,N-y+1,z) = 1;
                    data(x,y,N-z+1) = 1;
                    data(N-x+1,y,N-z+1) = 1;
                    data(x,N-y+1,N-z+1) = 1;
                    data(N-x+1,N-y+1,N-z+1) = 1;                    
                end
            end
        end
    end
end