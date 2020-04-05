%
%   Plane integrals of the ball computation
%   Input params:
%       N           number of planes per the direction
%       dt          discretisation step
%       R           radius of the ball
%   Output params:
%       res         array of integral values with size N
%

function res = getAnalyticalIntegrals(N, dt, R)
    res = zeros(1,N);
    for i = 1:N
        d = -R + dt*(i-1);
        res(1,i) = pi*(R^2-d^2);
    end
    
end

