function out = func_1bls_cg(z, h, ini, cg_method, maxiter)
% train Quasi-NN using CG method without merge and prune
% z -- one-bit signal
% h -- known threshold
% ini -- initial value
% cg_method -- choose CG update parameter
% maxiter -- max iteration times
% out -- output

N       = length(z);
n       = (0: N-1).';
w       = ini.freq;
a       = ini.amp;
nu      = ini.noise_var;
lambda  = sqrt(2/nu);
a       = a*lambda;
K       = length(a);

x0      = [real(a); imag(a); w; lambda]; % initial value
L       = @(x)func_cost(x, z, h, n, K);
G       = @(x)func_grad(x, z, h, n, K);

% conjugate gradient
[x, loss] = func_conjugate_gradient(L, G, x0, cg_method, maxiter);

a = x(1:K)+1j*x(K+1:2*K);
w = x(2*K+1:3*K);
w = wrapTo2Pi(w);

lambda = x(end);
nu = 2/lambda^2;
a = a/lambda;

out.amp = a;
out.freq = w;
out.noise_var = nu;
out.loss = loss; % loss curve
end

%% cost function
function L = func_cost(x, z, h, n, K)
F       = @(x) - log(erfc(-x./sqrt(2))/2);
base    = @(w) exp(1j*n*w.');

aR      = x(1:K);
aI      = x(K+1:2*K);
a       = aR + 1j*aI;
w       = x(2*K+1:3*K);
lambda  = x(end);

zR = real(z); zI = imag(z);
hR = real(h); hI = imag(h);

y = base(w)*a;
R = zR.*(real(y) - lambda*hR);
I = zI.*(imag(y) - lambda*hI);

L = sum(F(R)) + sum(F(I));
end

%% gradient function
function G = func_grad(x, z, h, n, K)
gF      = @(x) - sqrt(2/pi)./erfcx(-x./sqrt(2));
base    = @(w) exp(1j*n*w.');
Dn      = repmat(n, 1, K);
N       = length(n);

aR      = x(1:K);
aI      = x(K+1:2*K);
a       = aR + 1j*aI;
w       = x(2*K+1:3*K);
lambda  = x(end);

zR = real(z); zI = imag(z);
hR = real(h); hI = imag(h);
DzR = repmat(zR, 1, K);
DzI = repmat(zI, 1, K);

A = base(w);
y = A*a;
R = zR.*(real(y) - lambda*hR);
I = zI.*(imag(y) - lambda*hI);

gFR = gF(R); gFI = gF(I);

% lambda gradient
gRL = -zR.*hR; gIL = -zI.*hI;
gL = gFR.'*gRL + gFI.'*gIL;

% aR aI w gradient
gRaR = DzR.*real(A);
gIaR = DzI.*imag(A);
gaR = gRaR.'*gFR + gIaR.'*gFI;

gRaI = -DzR.*imag(A);
gIaI = DzI.*real(A);
gaI = gRaI.'*gFR + gIaI.'*gFI;

tmp = 1j*Dn.*A.*repmat(a.',N,1);
gRw = DzR.*real( tmp );
gIw = DzI.*imag( tmp );
gw  = gRw.'*gFR + gIw.'*gFI;

G = [gaR; gaI; gw; gL];
end
