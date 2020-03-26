function res = L(matrix, x, y, sigma, sigma_noise, N, N_angles)

    cov_matrix = (sigma + sigma_noise) * eye(N*N_angles);
    res = -log(1 / sqrt((2*pi)^(N*N_angles) * det(cov_matrix)));
    x = x.';
    inv_matr = inv(cov_matrix);
    res = res - 1/2 * (matrix * x - y).' * inv_matr * (matrix * x - y);
    
    for i=1:N^3
        res = res - log(normcdf(x(i),0.5,sigma));
        %res = res - log(0.5);
    end
    
end