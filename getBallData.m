%
%   3D-array with zeros and ones generation. Ones are located in each point
%   which satisfies x^2 + y^2 + z^2 < R^2
%   Input params:
%       N           number of elements in each dimention
%       R           radius of ball
%   Output params:
%       data    	3D array NxNxN with generated values
%       dt          discretization step
%
function [data, dt] = getBallData(N, R)

    data = zeros(N,N,N);
    discrete_center = floor((N+1)/2);
    dt = R/(N-discrete_center);
    for z = 1:N
        for y = 1:N
            for x = 1:N
                x_b = -R + (x-1)*dt;
                y_b = -R + (y-1)*dt;
                z_b = -R + (z-1)*dt;                
                if(x_b^2 + y_b^2 + z_b^2 <= R^2)
                    data(x,y,z) = 1;
                end
            end
        end
    end
end