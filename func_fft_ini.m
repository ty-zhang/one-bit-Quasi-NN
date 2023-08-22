function [aini, wini] = func_fft_ini(y, numPeak, issplit)
N       = length(y);
fftlen  = 8*N;
n       = (0: N-1).';
base    = @(w) exp(1j*n*w);

yfft    = 1/N*fft(y, fftlen);
[~, wfft] = findpeaks(abs(yfft), 'SortStr','descend');
wfft    = wfft(1:numPeak);

if issplit == 0
    wini = 2*pi*(wfft - 1)/fftlen;
    aini = zeros(numPeak, 1);
    for kk = 1:numPeak
        aini(kk) = base(wini(kk))\y;
    end
else
    wini = [];
    for kk = 1:numPeak
        tmp = (wfft(kk)+1).*(abs(yfft(wfft(kk)+1))>abs(yfft(wfft(kk)-1))) + ...
            (wfft(kk)-1).*(abs(yfft(wfft(kk)-1))>abs(yfft(wfft(kk)+1)));
        wini = [wini; wfft(kk); tmp];
    end
    wini = unique(wini, 'sorted');
    wini = 2*pi*(wini - 1)/fftlen;
    Kest = length(wini);
    aini = zeros(Kest, 1);
    for kk = 1:Kest
        aini(kk) = base(wini(kk))\y;
    end
end
end