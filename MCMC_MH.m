function [chain, ratio] = MCMC_MH(N_steps, N, N_angles, N_burnin_period, prop_sigma, radon_matrix, y, init_value)
    chain = zeros(N_steps, N^3);
    
    old_value = init_value; 
    chain(1,:) = init_value;    
    old_est = L(radon_matrix, old_value, y, 0.01, 0.1, N, N_angles);
    
    ratio = 0;
    for i=2:N_steps
        
       %new = old + x, where x ~ N([0,0,..0], prop_sigma * I)
        new_value = old_value + mvnrnd(zeros(1, N^3), prop_sigma * eye(N^3));
%         new_value = old_value;
%         for k = 1:N^3
%             if (rand < 0.5)
%                 t = 0;
%             else
%                 t = 1;
%             end
%             new_value(1,k) = mod(new_value(1,k) + t, 2);
%         end
        
        new_est = L(radon_matrix, new_value, y, 0.01, 0, N, N_angles);

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