%
%   Gaussian log-posterior function. 
%   Here likelihood function is Gaussian and priors are also Gaussian
%   Input params:
%       matrix          matrix of linear operator
%       x               generated element of chain
%       y               calculated integrals (initial data) 
%       sigma_lh        sigma of likelihood distribution
%       sigma_noise     sigma of noise distribution
%       sigma_priors    sigma of priors distribution
%       N               numer of elements in each direction
%       N_angles        number of angles
%   Output params:
%       res             log-posterior value         
%

function res = L(matrix, x, y, sigma_lh, sigma_noise, sigma_priors, N, N_angles)
    
    index = @(x,y,z) x + N * (y - 1) + N^2 * (z - 1);
    
    dim = N*N_angles;
    cov_matrix = (sigma_lh + sigma_noise) * eye(dim);
    
    res = -log(1 / sqrt((2*pi)^dim * det(cov_matrix)));
    x = x.';
    y = y.';
    inv_matr = inv(cov_matrix);
    res = res + 1/2 * (matrix * x - y).' * inv_matr * (matrix * x - y);

    
    % piors calculation
    res = res - N^3 * log(1/sqrt(2*pi) * sigma_priors);   
    c = 1/(2*sigma_priors^2);
    for k=2:N
        for j=2:N
            for i=2:N
                t = (x(index(i,j,k)) - x(index(i-1,j,k)))^2 + (x(index(i,j,k)) - x(index(i,j-1,k)))^2 + (x(index(i,j,k)) - x(index(i,j,k-1)))^2;
                res = res + c*t;
            end
        end
    end
    
    for i=1:N
        for j=1:N
            res = res + c*x(index(i,j,1))^2;
            res = res + c*x(index(i,1,j))^2;
            res = res + c*x(index(1,i,j))^2;
        end
    end
    
end