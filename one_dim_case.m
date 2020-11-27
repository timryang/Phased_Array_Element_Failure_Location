%% One Dimension

M = 10;
theta = 0;
lambda = 1;
d = lambda/2;

xpos = -M/2:M/2;
k = 2*pi/lambda;
el_response = exp(-1i*k*d*sind(theta)*xpos);

theta_sweep = [-90:90]';
steer_v = exp(-1i*k*d*sind(theta_sweep)*xpos);

% el_response(7) = 0;
power = steer_v*el_response';

figure; hold on; grid on; plot(theta_sweep, 180/pi*angle(power))
figure; hold on; grid on; plot(theta_sweep,20*log10(abs(power))); ylim([-30,25])