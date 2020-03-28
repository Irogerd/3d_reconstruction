%
%   Calculation of angle vector representation which is required to ASTRA
%   Input params:
%       angle       [theta, phi] vector
%   Output params:
%       res         vector representation of vector with size = 12 
%       
% How it works:
% - given normal vector one represents its description to Cartesian format;
% - perpendicular plane is defined and then vector mith maximal z-component
% in this plane is fined;
% - detector center is always (0,0,0);
% - vector u is the result of finded vector and normal vector cross product
% - vector v is given normal vector
% output format
% [dx, dy, dz, cx, cy, cz, ux, uy, uz, vx, vy, vz]
% [dx, dy, dz] - coordinates of x-ray direction vector
% [cx, cy, cz] - coordinates of detector center
% [ux, uy, uz] - the vector from detector pixel (0,0) to (0,1)
% [vx, vy, vz] - the vector from detector pixel (0,0) to (1,0)


function res = getVectorRepresentationOfAngle(angle)
    theta_normal = deg2rad(angle(1));
    phi_normal = deg2rad(angle(2));
    
    % define one of integration plane (x-ray direction is there)
    a = sin(theta_normal) * cos(phi_normal);
    b = sin(theta_normal) * sin(phi_normal);
    c = cos(theta_normal);
    normal = [a b c];
    
    plane_equation_wrt_z = @(x)(-(a*x(1) + b*x(2)) / c);
    g = @(x)(-plane_equation_wrt_z(x));
    
    % direction vector must has max z-component
    x0 = [0 0];
    [xmin, gmin] = fminsearch(g, [0, 0]);
    if xmin(1) ~= 0 && xmin(2) ~= 0 && gmin ~= 0
        direction_vector = [xmin(1) xmin(2) -gmin]/sqrt(xmin(1)*xmin(1) + xmin(2)*xmin(2) + gmin*gmin);
    else
        direction_vector = [1 0 0];
    end
    
    % detector's vectors are 1. given normal 2. cross product of given
    % normal and direction vector
    detector_vector = cross(normal, direction_vector);
    detector_vector = detector_vector / sqrt(detector_vector(1)*detector_vector(1) + detector_vector(2)*detector_vector(2) + detector_vector(3)*detector_vector(3));
    
    res = zeros(1,12);
    res(1:3) = direction_vector;
    res(7:9) = detector_vector;
    res(10:12) = normal;

    
end
