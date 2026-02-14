clear
close all

addpath(genpath('./datasets'))
addpath(genpath('./algorithms'))
rng(1) % set random seed for reproducibility

%% Set parameters for numerical integration of ODE
options = odeset('RelTol',1e-14,'AbsTol',1e-14);
SIGMA = 10;   BETA = 8/3;   RHO = 28;
ODEFUN = @(t,y) [SIGMA*(y(2)-y(1));y(1).*(RHO-y(3))-y(2);y(1).*y(2)-BETA*y(3)];

%% Set parameters for the example
N = 1200;    % number of delays (this is an upper bound, below we reduce at various stages)
M = 10^4;    % number of snapshots      
dt = 0.05;    % time step for trajectory sampling

%% Produce the data
M2 = 10^5; % for visualisation of functions we use more data points
Y0 = (rand(3,1)-0.5)*4; % initial point off the Lorenz attractor
[~,Y0] = ode45(ODEFUN,[0.000001 1, 100],Y0,options); Y0 = Y0(end,:)'; % new initial point on the Lorenz attractor
h = 1; % number of time steps for delay
[~,DATA] = ode45(ODEFUN,[0.000001 (1:((M2+h*(N+1))))*dt],Y0,options);

%% Use delay embedding as the dictionary
PX1=zeros(M2,N); PX1(:,1)=DATA(1:M2,1);
PX2=zeros(M2,N); PX2(:,1)=DATA(1:M2,2);
PX3=zeros(M2,N); PX3(:,1)=DATA(1:M2,3);
PY1=zeros(M2,N); PY1(:,1)=DATA((1:M2)+1,1);
PY2=zeros(M2,N); PY2(:,1)=DATA((1:M2)+1,2);
PY3=zeros(M2,N); PY3(:,1)=DATA((1:M2)+1,3);

for j=2:N
    PX1(:,j)=DATA((1:M2)+h*(j-1),1);
    PX2(:,j)=DATA((1:M2)+h*(j-1),2);
    PX3(:,j)=DATA((1:M2)+h*(j-1),3);
    PY1(:,j)=DATA((1:M2)+1+h*(j-1),1);
    PY2(:,j)=DATA((1:M2)+1+h*(j-1),2);
    PY3(:,j)=DATA((1:M2)+1+h*(j-1),3);
end

%%

N = 500;

PX = [PX1(1:M,1:N),PX2(1:M,1:N),PX3(1:M,1:N)];
PY = [PY1(1:M,1:N),PY2(1:M,1:N),PY3(1:M,1:N)];

%% Lorenz attractor
figure
plot3(PX1(1:M),PX2(1:M),PX3(1:M),'.b','markersize',15)
view(gca,[13.1786087602293 -1.28469255513244]);
xlabel('$x$','interpreter','latex','fontsize',18)
ylabel('$y$','interpreter','latex','fontsize',18)
zlabel('$z$','interpreter','latex','fontsize',18)
exportgraphics(gcf,'lorenz_attractor.png');

%%
figure
plot(PX1(1:1000),'k','linewidth',2)
axis off
exportgraphics(gcf,'lorenz_series.png');

%% Find errors
Nvec = 20:20:500;
[E1,E2] = lorenz_err(PX,PY,Nvec);

%%
figure
loglog(Nvec,E1,'b','linewidth',2,'markersize',20)
hold on
plot(Nvec,E2,'r:','linewidth',2,'markersize',20)
legend({'EDMD','proposed'},'fontsize',18,'interpreter','latex','location','best')
xlabel('$\frac{1}{3}\times$ Matrix Size','interpreter','latex','fontsize',18)
ylabel('Error','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=16;
exportgraphics(gcf,'lorenz_error.png');

return

%% Compute pseudospectra
N2 = 100;
PX = [PX1(1:M,1:N2),PX2(1:M,1:N2),PX3(1:M,1:N2)];
PY = [PY1(1:M,1:N2),PY2(1:M,1:N2),PY3(1:M,1:N2)];
x_pts = -1.5:0.02:1.5;    y_pts = -0.04:0.02:1.5; % use conjugate symmetry to only consider positive y
zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
DIST = KoopPseudoSpecQR(PX,PY,1/M,zpts,'Parallel','off');
DIST = max(DIST,abs(1-abs(zpts)));
LAM = eig(PX\PY);

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
exportgraphics(gcf,'lorenz_pseudospectra.png');
% title(sprintf('$\\mathrm{Sp}_\\epsilon(\\mathcal{K})$'),'interpreter','latex','fontsize',18)








function [E1,E2] = lorenz_err(PX,PY,Nvec)


E1 = zeros(1,length(Nvec));
E2 = zeros(1,length(Nvec));
M = length(PX(:,1));

ct = 1;

for N = Nvec
	N
    Id = 1:N;
    Id = [Id,Id+500,Id+500];

        
    %% Apply EDMD algorithm (which does not converge)
    K = PX(:,Id)\PY(:,Id);
    LAM = eig(K,'vector');
    DIST = KoopPseudoSpecQR(PX(:,Id),PY(:,Id),1/M,LAM,'Parallel','off');
    E1(ct) = max(DIST);
    zpts = LAM(:);
    
       
    %% Run new algorithm for finding spectrum
    spec = [];
    for jj = 1:length(zpts)
        if DIST(jj)<1
            II = find(abs(zpts(jj)-zpts)<5*DIST(jj));
            if ~isempty(II)
                RR = DIST(II);
                JJ = find(RR==min(RR));
                JJ = II(JJ(1));
                spec = [spec(:);JJ];
            end
        end
    end
    spec = unique(spec);
    E2(ct) = max(DIST(spec));
    ct = ct+1;
end


end

