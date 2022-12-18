clc; clear; close all;

% setting parameters
N       = 256; % signal length
n       = (0: N-1).'; 
K       = 3; % number of sinusoids (model order)
w       = 2*pi*[0.1; 0.23; 0.37]; % frequency
a       = 2*exp(1j*2*pi*rand(K, 1)); % amplitudes
base    = @(w) exp(1j*n*w.');

snrDb   = 20; % SNR (dB)
nu      = norm(a)^2*10^(-snrDb/10);
e       = sqrt(nu/2)*(randn(N, 1) + 1j*randn(N, 1)); % white Gaussian noise

x       = base(w)*a; % noise-free signal


h_bit   = 3; % quantization bit (2^h_bit)
h_max   = 2; % maximum threshold value
h_lv    = linspace(-h_max, h_max, 2^h_bit).';
h       = h_lv(randi(2^h_bit, [N, 1])) + ...
    1j*h_lv(randi(2^h_bit, [N, 1]));

y       = x + e;
z       = (real(y - h)>=0) + 1j*(imag(y - h)>=0);
z       = 2*z - (1+1j); % one-bit signal

% fft initial
Np = 3; % initial value of number of spectral peaks
ini = func_fft_ini(z, h_max, Np, 1);

tol.merge = 1e-14; % false alarm rate of merge
tol.prune = 1e-14; % false alarm rate of prune
tol.maxiter = 2e4; % max iteration times

% Quasi-NN using CG
out = func_1bls_pm(z, h, ini, 'DY', tol);


yfft    = fft(y, 1024);
% plot results
figure();
semilogy(0:1/1024:1-1/1024, abs(yfft)/N,':', 'LineWidth',1.5); hold on;  % infinite precision FFT
stem(w/2/pi, abs(a), 'LineWidth',1.5, 'MarkerSize',8); % ground truth
stem(out.freq/2/pi, abs(out.amp), 'LineWidth',1.5, 'MarkerSize',8, 'Marker','x'); % estimates
grid on; xlim([0,1]); ylim([0, 2.5]);
legend('Spectrum of infinite precision signal', 'Ground Truth', 'Estimation results of 1b Quasi-NN');

% plot loss curve
figure(); semilogy(out.loss, 'LineWidth',1.5); grid on;
