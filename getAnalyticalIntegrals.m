function res = getAnalyticalIntegrals(N, dt, R)
%     res = zeros(N, N_angles);
%     dt = 2*L / N;
%     
%     for rho = 1:N
%         for angle = 1:N_angles
%             theta = deg2rad(angles(angle,1));
%             phi = deg2rad(angles(angle, 2));
%             
%             dist = -L + (rho-1)*dt;
%             res(rho, angle) = 4*(1 - dist*cos(phi)*sin(theta) - dist*sin(phi)*sin(theta) - dist*cos(theta));
%         end
%     end
    res = zeros(1,N);
    for i = 1:N
        d = -R + dt*(i-1);
        res(1,i) = pi*(R^2-d^2);
    end
    
end

