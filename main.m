



N = 5; % number of elements
N_angles = 15; % number of angles

% calculate initial data for comparison
initial_data = getDataIdenticalSphere(5,2,3);
initial_data_vector = zeros(1,N^3);

counter = 1;
for z = 1:N
    for x = 1:N
        for y=1:N
            initial_data_vector(counter) = initial_data(x,y,z);
            counter = counter + 1;
        end
    end
end

% read Radon transform matrix from file
fileID = fopen('RM_5', 'r');
matrix_string = fscanf(fileID, '%f');
fclose(fileID);
radon_matrix = reshape(matrix_string, [N^3, N*N_angles]).';

% read calculated plane integrals from file
fileID_data = fopen('profiles_5','r');
data_vector = fscanf(fileID_data, '%f');
fclose(fileID_data);
data = reshape(data_vector, [N_angles, N]).';

clear x y z matrix_string fileID_data fileID counter;

% ================
%       MCMC
% ================

% initial value of chain
init = zeros(1, N^3);
for i = 1:N^3
    init(1,i) = 0.5;
end

MCMC_steps = 1000;
burnin_steps = 200;
t1 = now();
[chain, ratio] = MCMC_MH(MCMC_steps, N, N_angles, burnin_steps, 0.001, radon_matrix, data_vector, init);
t2 = now();
disp((t2-t1)*24*60*60);
disp(ratio/MCMC_steps);

res = mean(chain);
res = res / max(res);

% res_hypermatr = zeros(N,N,N);
% counter = 1;
% for z=1:N
%     for x=1:N
%         for y = 1:N
%             res_hypermatr(x,y,z) = res(counter);
%             counter = counter + 1;
%         end
%     end
% end

x_axe = zeros(1,N^3);
for i=1:N^3
    x_axe(1,i) = i;
end


fileID = fopen('reconstruction_5', 'r');

data_string = fscanf(fileID, '%f');
fclose(fileID);
res_astra = zeros(N, N,N);
counter = 1;
for i = 1:N
	for j = 1:N
        for k=1:N
           res_astra(i, j, k) = data_string(counter);
           counter = counter + 1;
        end
    end
end

res_astra_vector = zeros(1,N^3);
counter = 1;
for k = 1:N
	for i = 1:N
        for j=1:N
           res_astra_vector(counter) = res_astra(i, j, k);
           counter = counter + 1;
        end
    end
end

plot(x_axe, initial_data_vector,':g*', x_axe, res_astra_vector,'-.b.',  x_axe, res, '--r*')
legend('original data', 'astra reconstruction', 'stat reconstruction');

