function out = func_1bls_pm(z, h, ini, cg_method, tol)
% train Quasi-NN using CG method with merge and prune
% z -- one-bit signal
% h -- known threshold
% ini -- initial value
% cg_method -- choose CG update parameter
% tol -- network training parameters

ispm        = 1; % do merging and pruning or Not
loss        = [];
N           = length(z);

while ispm 
    out_in = func_1bls_cg(z, h, ini, cg_method, tol.maxiter);
    loss = [loss, out_in.loss];

    out_pm = node_merge_1b(out_in, N, tol.merge); % merge nodes
    out_pm = node_prune_1b(out_pm, N, tol.prune); % prune nodes

    if isempty(out_pm.freq)
        break;
    end

    ispm = (length(out_pm.freq) < length(out_in.freq));
    ini.amp = out_pm.amp;
    ini.freq = out_pm.freq;
    ini.noise_var = out_pm.noise_var;
end
out.amp = out_pm.amp;
out.freq = out_pm.freq;
out.noise_var = out_pm.noise_var;
out.loss = loss;
end

%% merge function
function out = node_merge_1b(ini, N, err_rate)
a = ini.amp;
w = ini.freq;
nu = ini.noise_var;
% merge network nodes
[w, order] = sort(w, 'ascend');
a = a(order);

ismerge = 1;
while ismerge
    if length(w) == 1
        ismerge = 0;
    else
        [a, w, nu, ismerge] = merge2nodes(a, w, nu, N, err_rate);
    end
end
out.amp = a;
out.freq = w;
out.noise_var = nu;
end

function [a, w, nu, ismerge] = merge2nodes(a, w, nu, N, err_rate)
n       = (0: N-1).';
base    = @(x) exp(1j*n*x);
M   = length(w);

% merde two closest nodes
baseEst = base(w.');
xEst    = baseEst*a;
ismerge = 0;

CovAmp = inv(baseEst'*baseEst);
criterion = finv(1 - err_rate, 2, 2*(N - M))/(N-M);
criterion = N*nu*criterion;

tmp = zeros(M, 1);
lambda = zeros(M, 1);
wAvg = zeros(M, 1);
aAvg = zeros(M, 1);
for ii = 1:M
    if ii == M
        seq = [1, M];
    else
        seq = ii:ii+1;
    end
    tmp1 = baseEst(:, seq)*a(seq);
    wAvg(ii) = mean(w(seq));
    wAvg(ii) = wrapTo2Pi(wAvg(ii));
    baseAvg = base(wAvg(ii));
    aAvg(ii) = sum(a(seq));

    tmp2 = aAvg(ii)*baseAvg;
    Covij = baseEst(:,seq)*CovAmp(seq, seq)*baseEst(:, seq)';
    lambda(ii) = real(trace(Covij) - baseAvg'*Covij*baseAvg/N);
    tmp(ii) = norm(tmp1 - tmp2)^2/lambda(ii);
end
[~, idx] = min(tmp);
tmp = tmp(idx);

if tmp <= criterion
    ismerge = 1;
    if idx == M
        w = [wAvg(idx); w(2:M-1)];
        a = [aAvg(idx); a(2:M-1)];
    else
    w = [wAvg(idx); w(1:idx-1); w(idx+2:end)];
    a = [aAvg(idx); a(1:idx-1); a(idx+2:end)];
    end
    nu = nu + norm(xEst - base(w.')*a)^2/N; % update noise variance
end
[w, order] = sort(w, 'ascend');
a = a(order);
end

%% prune function
function out = node_prune_1b(ini, N, err_rate)
a = ini.amp;
w = ini.freq;
nu = ini.noise_var;

isprune = 1;
while isprune
    [a, w, nu, isprune] = prune_one_node(a, w, nu, N, err_rate);
    if isempty(w)
        isprune = 0;
    end
end
[w, ord] = sort(w, 'ascend');
a = a(ord);

out.amp = a;
out.freq = w;
out.noise_var = nu;
end

function [a, w, nu, isprune] = prune_one_node(a, w, nu, N, err_rate)
M       = length(a);
n       = (0: N-1).';
isprune = 0;
base    = @(x) exp(1j*n*x);

criterion = N*nu*finv((1 - err_rate), 2, 2*(N-M))/(N-M); 
baseEst = base(w.');

CovAmp  = inv(baseEst'*baseEst);

tmp     = zeros(M, 1);
for ii  = 1:M
    Covi    = base(w(ii))*CovAmp(ii, ii)*base(w(ii))';
    lambda  = real(trace(Covi));
    tmp(ii) = norm(base(w(ii))*a(ii))^2;
    tmp(ii) = tmp(ii)/lambda;
end

[~, idx] = min(tmp);
tmp = tmp(idx);

if tmp <= criterion 
	nu = nu + abs(a(idx))^2; % update noise variance
    a = a([1:idx-1, idx+1:end]);
    w = w([1:idx-1, idx+1:end]);
    isprune = 1;
end
end