%
%   Metropolis-Hastings MCMC algorithm realization
%   Input params:
%       N_steps             number of steps
%       N                   numer of elements in each direction
%       N_angles            number of angles
%       N_burnin_period     number of burn-in elements
%       prop_sigma          sigma of proposal distribution
%       radon_matrix        matrix of Radon transform
%       y                   calculated integrals (initial data)
%       init_value          initial element of chain
%   Output params:
%       chain               Markov chain: 2D array
%                           (N_steps-N_burnin_period)xN^3
%       ratio               rate of accepted chain elements                 
%

function [chain, ratio] = MCMC_MH(N_steps, N, N_angles, N_burnin_period, prop_sigma, radon_matrix, y, init_value)
    chain = zeros(N_steps, N^3);
    
    old_value = init_value; 
    chain(1,:) = init_value;    
    old_est = L(radon_matrix, old_value, y, 0.1, 0.1, 0.1, N, N_angles);
    %old_est = L(radon_matrix, old_value, y, 0.001, 0.01, 0.001, N, N_angles);
%     meanval = zeros(1, N^3);
%     for i = 1:N^3
%         meanval(1,i) = 0.04;
%     end
    ratio = 0;
    for i=2:N_steps   
        %new = old + x, where x ~ N([0,0,..0], prop_sigma * I)
        new_value = old_value + mvnrnd(zeros(1, N^3), prop_sigma * eye(N^3));
        %new_value = old_value + mvnrnd(meanval, prop_sigma * eye(N^3));
        
        new_est = L(radon_matrix, new_value, y, 0.1, 0.1, 0.1, N, N_angles);
        %new_est = L(radon_matrix, new_value, y, 0.001, 0.01, 0.001, N, N_angles);

        check = exp(old_est-new_est);
        if (rand < exp(old_est-new_est))
            chain(i,:) = new_value;
            old_value = new_value;
            old_est = new_est;
            ratio = ratio+1;
        else
            chain(i,:) = old_value;
        end
    end    
    chain = chain(N_burnin_period+1:N_steps,:);
    
end