function ini = func_fft_ini(z, h_max, Np, issplit)
% fast FFT-based initialization
% z -- one-bit signal
% h_max -- max threshold value
% Np -- initial value of number of spectral peaks
% issplit -- split flag
%            1: split
%            0: not split
% ini -- initial values

y       = h_max*z;
N       = length(y);
fftlen  = 8*N;
n       = (0: N-1).';
base    = @(w) exp(1j*n*w);

yfft    = 1/N*fft(y, fftlen);
[~, wfft] = findpeaks(abs(yfft), 'SortStr','descend');
wfft    = wfft(1:Np);

if issplit == 0 % not split
    wini = 2*pi*(wfft - 1)/fftlen;
    aini = zeros(Np, 1);
    for kk = 1:Np
        aini(kk) = base(wini(kk))\y;
    end
else % split
    wini = [];
    for kk = 1:Np
        tmp = (wfft(kk)+1).*(abs(yfft(wfft(kk)+1))>abs(yfft(wfft(kk)-1))) + ...
            (wfft(kk)-1).*(abs(yfft(wfft(kk)-1))>abs(yfft(wfft(kk)+1)));
        wini = [wini; wfft(kk); tmp];
    end
    wini = unique(wini, 'sorted'); % remove repeated values
    wini = 2*pi*(wini - 1)/fftlen;
    Kest = length(wini);
    aini = zeros(Kest, 1);
    for kk = 1:Kest
        aini(kk) = base(wini(kk))\y;
    end
end
ini.amp = aini; % initial amplitudes
ini.freq = wini; % initial frequencies
ini.noise_var = 0.1*h_max^2; % initial noise variance
end