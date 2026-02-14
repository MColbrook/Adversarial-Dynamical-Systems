clear
close all
addpath(genpath('./algorithms'))
addpath(genpath('./data_sets'))

rng(1) % set random seed for reproducability

%% Set parameters
M1=10^4; M2=5;  % number of data points
delta_t = 0.3;  % time step
ALPHA = 0.0;    % damping parameter
ODEFUN = @(t,y) [y(2);-ALPHA*y(2)+y(1)-y(1).^3];
options = odeset('RelTol',1e-12,'AbsTol',1e-12);
N = 500;        % size of dictionary
PHI = @(r) exp(-r*2);   % radial basis function used (others also work well with similar results)

%% Produce the data for the radial basis function centers
X = zeros(2,M1*M2);
Y=[];
for jj=1:M1
    Y0=(rand(2,1)-0.5)*4;
    [~,Y1]=ode45(ODEFUN,[0 0.000001 (1:(3+M2))*delta_t],Y0,options);
    Y1=Y1';
    X(:,(jj-1)*M2+1:jj*M2) = Y1(:,[1,3:M2+1]);
    Y(:,(jj-1)*M2+1:jj*M2) = Y1(:,3:M2+2);
end
[~,C] = kmeans([X';Y'],N,'MaxIter',500);    % centers for radial functions

%% Produce the trajectory data
X = zeros(2,M1*M2);
Y=[];
for jj=1:M1
    Y0=(rand(2,1)-0.5)*4;
    [~,Y1]=ode45(ODEFUN,[0 0.000001 (1:(3+M2))*delta_t],Y0,options);
    Y1=Y1';
    X(:,(jj-1)*M2+1:jj*M2) = Y1(:,[1,3:M2+1]);
    Y(:,(jj-1)*M2+1:jj*M2) = Y1(:,3:M2+2);
end
M = M1*M2;
d = mean(vecnorm(X-mean(X,2)));             % scaling for radial functions

PX = zeros(M,N); PY = zeros(M,N);           % dictionary evaluated at data points

for j = 1:N
    R = sqrt((X(1,:)-C(j,1)).^2+(X(2,:)-C(j,2)).^2);
    PX(:,j) = PHI(R(:)/d);
    R = sqrt((Y(1,:)-C(j,1)).^2+(Y(2,:)-C(j,2)).^2);
    PY(:,j) = PHI(R(:)/d);
end

%% Apply EDMD algorithm (which does not converge)
K = PX(1:M,:)\PY(1:M,:);
[V,LAM] = eig(K,'vector');
R = vecnorm(PY(1:M,:)*V-PX(1:M,:)*V*diag(LAM))./vecnorm(PX(1:M,:)*V); % residual errors of EDMD
er_EDMD = max(R);

%% Compute errors over a grid
x_pts = -1.5:0.05:1.5;    y_pts = -0.04:0.05:1.5; % use conjugate symmetry to only consider positive y
zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
DIST = KoopPseudoSpecQR(PX,PY,1/M,zpts,'Parallel','off');
if ALPHA == 0
    DIST = max(DIST,abs(1-abs(zpts)));
end

%% Run new algorithm for finding spectrum
spec = [];
for jj = 1:length(zpts)
    if DIST(jj)<1
        II = find(abs(zpts(jj)-zpts)<5*DIST(jj));
        RR = DIST(II);
        JJ = find(RR==min(RR));
        JJ = II(JJ(1));
        spec = [spec(:);JJ];
    end
end
spec = unique(spec);
er_NEW = max(DIST(spec));

%% Plot the pseudospectrum and EDMD eigenvalues

v=(10.^(-2:0.1:-0.5));
close all
DIST = reshape(DIST,length(y_pts),length(x_pts));

figure
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
hold on
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),-reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
cbh=colorbar;
cbh.Ticks=log10([0.005,0.01,0.1,1]);
cbh.TickLabels=[0,0.01,0.1,1];
clim(log10([min(v),max(v)]));
reset(gcf)
set(gca,'YDir','normal')
colormap inferno
hold on
plot(real(LAM),imag(LAM),'.b','markersize',14)

ax=gca; ax.FontSize=18;  axis equal tight;  axis([x_pts(1),x_pts(end),-y_pts(end),y_pts(end)])
xticks(-1.5:0.5:1.5)
xlabel('$\mathrm{Re}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\mathrm{Im}(\lambda)$','interpreter','latex','fontsize',18)
title(sprintf('$\\mathrm{Sp}_\\epsilon(\\mathcal{K})$'),'interpreter','latex','fontsize',18)

return
%% Convergence plot
Nvec = round(10.^(2:0.05:3.1));

[E1,E2] = duffing_err(Nvec,0);
[E3,E4] = duffing_err(Nvec,0.3);



