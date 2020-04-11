%
%   3D-array with the ball which has ellipsoid, parallelepiped and two
%   balls inside. 
%   Input params:
%       N           number of elements in each dimention
%       R           radius of ball
%   Output params:
%       data    	3D array NxNxN with generated values
%       dt          discretization step
%

function [data, dt] = getComplexBallData(N, R)
    data = zeros(N,N,N);
    discrete_center_main_ball = floor((N+1)/2);
    dt = R/(N-discrete_center_main_ball);
    
    for z = 1:N
        for y = 1:N
            for x = 1:N
                % current corrdinates
                x_b = -R + (x-1)*dt;
                y_b = -R + (y-1)*dt;
                z_b = -R + (z-1)*dt;  
            
                % if coordinates are located inside the main ball
                if(x_b^2 + y_b^2 + z_b^2 <= R^2) 
                    data(x,y,z) = 1;
                    
                    %if coordinates are located inside the ellipsoid
                    if ((x_b+0.4*R)^2/(0.2*R)^2 + y_b^2/(0.4*R)^2 + (z_b+0.4*R)^2/(0.2*R)^2 <= 1)
                        data(x,y,z) = 0.35;
                    end
                    
                    % if coordinates are located inside the parallelepiped
                    if (x_b > -R*0.4 && x_b < R*0.4 && y_b > - R*0.5 && y_b < R*0.5 && z_b > -R*0.3 && z_b < R*0.3)
                        data(x,y,z) = 0.2;
                        
                        %if coordinates are located inside the left small
                        %ball
                        if((x_b-0.1*R)^2 + (y_b+0.2*R)^2 + z_b^2 <= (0.2*R)^2)
                            data(x,y,z) = 0.5;
                        end
                        %if coordinates are located inside the right
                        %small ball
                        if(x_b^2 + y_b^2 + z_b^2 <= (0.1*R)^2)
                            data(x,y,z) = 0.85;
                        end
                    end
                end
            end
        end
    end  
end



