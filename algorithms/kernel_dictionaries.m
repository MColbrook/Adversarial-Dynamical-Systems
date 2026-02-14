function [G,K,L,PX,PY] = kernel_dictionaries(Xa,Ya,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% Xa and Ya: data matrices used in kernel_EDMD to form dictionary (columns
% correspond to instances of the state variable)

% OPTIONAL LABELLED INPUTS
% N: size of computed dictionary, default is number of data points for kernel EDMD
% type: kernel used, default is normalised Gaussian, "Laplacian" is for
% nomralised Laplacian, and numeric value (e.g., 20) is for polynomial
% kernel
% cut_off: stability parameter for SVD, default is 0

% OUTPUTS
% G K L matrices for kernelResDMD
% PSI matrices for ResDMD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Collect the optional inputs
p = inputParser;

addParameter(p,'N',size(Xa,2),@(x) x==floor(x))
addParameter(p,'type',"Gaussian");
addParameter(p,'cut_off',0,@(x) x>=0)
addParameter(p,'Xb',[],@isnumeric)
addParameter(p,'Yb',[],@isnumeric)
addParameter(p,'Y2',[],@isnumeric)

p.CaseSensitive = false;
parse(p,varargin{:})

% Apply kernel EDMD
if isnumeric(p.Results.type)
    d = mean(vecnorm(Xa));
    kernel_f = @(x,y) (y'*x/d^2+1).^(p.Results.type);
elseif p.Results.type=="Linear"
    kernel_f = @(x,y) y'*x;
elseif p.Results.type=="Laplacian"
    d = mean(vecnorm(Xa-mean(Xa,2)));
    if isa(Xa,'single') % safeguard against square root (but a little bit slower)
        kernel_f = @(x,y) exp(-pdist2(y',x')/d);
    else
        kernel_f = @(x,y) exp(-sqrt(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d);
    end
elseif p.Results.type=="Gaussian"
    d = 0.9*mean(vecnorm(Xa-mean(Xa,2)));
    kernel_f = @(x,y) exp(-(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d^2);
elseif p.Results.type=="Lorentzian"
    d = mean(vecnorm(Xa-mean(Xa,2)));
    kernel_f = @(x,y) (1+(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d^2).^(-1);
elseif p.Results.type=="poly_exp"
    d = mean(vecnorm(Xa-mean(Xa,2)));
    kernel_f = @(x,y) (exp(-(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d^2).*(y'*x/d^2+1).^20)+exp(-sqrt(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d).*(y'*x/d^2+1).^20;
elseif p.Results.type=="poly_exp2"
    d = mean(vecnorm(Xa-mean(Xa,2)));
    kernel_f = @(x,y) exp(-sqrt(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d).*(y'*x/d^2+1).^20;
end

G1 = kernel_f(Xa,Xa); G1 = (G1+G1')/2;
A1 = kernel_f(Ya,Xa)';
L1 = kernel_f(Ya,Ya);  L1 = (L1+L1')/2;

% Post processing

[U,D0] = eig(G1+norm(G1)*p.Results.cut_off*eye(size(G1)));
[~,I] = sort(diag(D0),'descend');
U = U(:,I); D0 = D0(I,I);
N = min(p.Results.N,length(find(diag(D0)>0)));
U = U(:,1:N); D0 = D0(1:N,1:N);
UU = U*sqrt(diag(1./diag(D0)));

% G = UU'*G1*UU;
G = eye(N);
K = UU'*A1*UU;
L = UU'*L1*UU;

PX = G1'*UU;
PY = A1*UU;

end
