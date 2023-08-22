clc; clear; close all;
N       = 256;
n       = (0: N-1).';
K       = 3;
w       = 2*pi*[0.1; 0.23; 0.37];
a       = 2*exp(1j*2*pi*rand(K, 1));
base    = @(w) exp(1j*n*w.');

snrDb   = 20;
nu      = norm(a)^2*10^(-snrDb/10);
x       = base(w)*a;
e       = sqrt(nu/2)*(randn(N, 1) + 1j*randn(N, 1));

h_bit   = 3;
h_max   = 2;
h_lv    = linspace(-h_max, h_max, 2^h_bit).';
h       = h_lv(randi(2^h_bit, [N, 1])) + ...
    1j*h_lv(randi(2^h_bit, [N, 1]));

y       = x + e;
z       = (real(y - h)>=0) + 1j*(imag(y - h)>=0);
z       = 2*z - (1+1j);

yfft    = fft(y, 1024);
figure();
semilogy(0:1/1024:1-1/1024, abs(yfft)/N,':', 'LineWidth',1.5); hold on; 
stem(w/2/pi, abs(a), 'LineWidth',1.5, 'MarkerSize',8);

% fft initial
numPeak = 3;
[aini, wini] = func_fft_ini(h_max*z, numPeak, 1);
stem(wini/2/pi, abs(aini), 'LineWidth',1.5, 'MarkerSize',8);

nu = 0.1*h_max^2;
ini.amp = aini; 
ini.freq = wini;
ini.noise_var = nu;

tol.merge = 1e-14;
tol.prune = 1e-14;

out = func_1bls_pm(z, h, ini, 'DY', tol);

aest = out.amp;
west = out.freq;
[west, ind] = sort(west, 'ascend');
aest = aest(ind);

stem(out.freq/2/pi, abs(out.amp), 'LineWidth',1.5, 'MarkerSize',8, 'Marker','x'); 
figure(); semilogy(out.loss, 'LineWidth',1.5); grid on;
