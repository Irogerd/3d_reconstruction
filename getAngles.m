%
%   Calculates set of angles which define directions. This set is Cartesian
%   product of number of theta angles set located between 1 and 90 degrees
%   and phi angles set located between 0 and 179 degrees
%   Input params:
%       N_theta         number of theta angles
%       N_phi           number of phi angles
%   Output params:
%       astra_angles    set of angles in its the ASTRA Toolbox representation 
%       deg_angles      set of angles in its degrees representation
%

function [astra_angles, deg_angles] = getAngles(N_theta, N_phi)
    N = N_theta*N_phi;
    deg_angles = zeros(N,2);
    theta = linspace(1,90,N_theta);
    phi = linspace(0,179,N_phi);
    for i = 1:N_theta
        for j = 1:N_phi
            deg_angles(j + (i-1)*N_phi, :) = [theta(i) phi(j)];
        end
    end
    
    astra_angles = zeros(N_theta*N_phi,12);
    for i = 1:N
        astra_angles(i,:) = getVectorRepresentationOfAngle(deg_angles(i,:));
    end    

end