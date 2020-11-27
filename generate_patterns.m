clear;clc;

SNR_vals = -4:2:20; % dB units
SNR_vals = 16;
num_iterations = 3; % Number of images per failure state
M = 5; % one dimension of 2D array of elements
phi = 0; % phi direction of signal
theta = 0; % theta direction of signal
res = 2^8; % FFT size, select powers of 2, higher is more resolution
do_phase = 0;

for i = 1:length(SNR_vals)
    i_SNR = SNR_vals(i);
    folder_name = [int2str(i_SNR),'dB_M5_Size8_Amp']; % Folder name (will be located in current directory)
    % folder_name = 'Test';
    sim_array(folder_name, num_iterations, i_SNR, M, phi, theta, res, do_phase)
end

%% Simulate Array

function sim_array(folder_name, num_iterations, SNR, M, phi, theta, res, do_phase)

    mkdir(folder_name);
    
    nv = 0.001; % Set constant noise variance
    sv = 10^(SNR/10)*nv; % Adjust signal variance depending on SNR
    lambda = 1; % signal wavelength (arbitrary)
    d = lambda/2; % scale in terms of lambda/2, aliasing occurs when d>lambda/2
    
    % Create cosine to zero for boundary of real space
    [xx,yy] = meshgrid(linspace(-1,1,res),linspace(-1,1,res));
    cos_var = 1-sqrt(xx.^2+yy.^2);

    % Create array position grid
    [x_pos_grid,y_pos_grid] = meshgrid(1:M,1:M);
    
%     figure; hold on; grid on;
%     plot(x_pos_grid(:),y_pos_grid(:),'o')
%     plot(4,4,'rx','MarkerSize',16)
%     xlabel('x position ({\lambda}/2)')
%     ylabel('y position ({\lambda}/2)')
%     xlim([0,M+1])
%     ylim([0,M+1])
%     legend({'El. Locations','Ex. Dead El.'})
%     title('Array Architecture')

    % Steer elements to source location
    el_phs_kx = exp(-1i*2*pi/lambda*d*sind(phi)*cosd(theta).*x_pos_grid);
    el_phs_ky = exp(-1i*2*pi/lambda*d*sind(phi)*sind(theta).*y_pos_grid);
    el_phs = el_phs_kx.*el_phs_ky;

    % Creating array of indexes
    idx_array = zeros((M*M+1)*num_iterations,2);
    % Format:
    % Pairs of consecutve columns: i,j index of killed instance within an image
    % Rows: i,j indices of all killed elements for a certain image
    
    counter = 1;
    % Generate images with no failures
    for it = 1:num_iterations
        % Generate far field pattern
        norm_pattern = gen_pattern(el_phs, sv, nv, cos_var, M, res, 0, 0, do_phase);
        
        % Save image
        dir_name = sprintf('%s/image_00_%d.png',folder_name,counter);
        imwrite(norm_pattern,dir_name,'png');
        
        % Save failure state ([0,0] is no failures)
        idx_array(counter,:) = 0;
        
        counter = counter+1;
    end
    
    % Generate images with failures iterating through all possible states
    fail_state = 1;
    for i_idx = 1:M
        for j_idx = 1:M
            if length(int2str(fail_state))<2
                fail_state_str = ['0',int2str(fail_state)];
            else
                fail_state_str = int2str(fail_state);
            end
            for it = 1:num_iterations
                % Generate far field pattern
                norm_pattern = gen_pattern(el_phs, sv, nv, cos_var, M, res, i_idx, j_idx, do_phase);
                
                % Save image
                dir_name = sprintf('%s/image_%s_%d.png',folder_name,fail_state_str,counter);
                imwrite(norm_pattern,dir_name,'png');

                % Save failure state ([i,j] element index)
                idx_array(counter,:) = [i_idx, j_idx];

                counter = counter+1;
            end
            fail_state = fail_state+1;
        end
    end
    
    % xlswrite(sprintf('/%s/%s.xlsx',folder_name,folder_name),idx_array);
    
end

function norm_pattern = gen_pattern(el_phs, sv, nv, cos_var, M, res, i_idx, j_idx, do_phase)

    % Create signal with noise scaled by sv and add noise at each sensor
    % scaled by sn
%     signal = sqrt(sv/2)*(randn(1,1)+1i*randn(1,1));
    signal = sqrt(sv);
    noise = sqrt(nv/2)*(randn(M)+1i*randn(M));

    % Simulate element response for given source location and signal/noise
    el_resp = signal*el_phs+noise;

    % Kill element
    if i_idx ~= 0 && j_idx ~=0
        el_resp(i_idx,j_idx) = 0;
    end

    % Compute far field pattern
    pattern = cos_var.*ifftshift(ifft2(el_resp,res,res));
    norm_pattern = abs(pattern)/max(max(abs(pattern)));
    if do_phase
        norm_pattern = angle(pattern);
    end

end