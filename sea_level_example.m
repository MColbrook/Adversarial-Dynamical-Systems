clear
close all

addpath(genpath('./datasets'))
addpath(genpath('./algorithms'))
rng(0)

%% load data sets
load('DATA_north.mat')
load('LAND_north.mat')
load('nLAND_north.mat')
DATA=DATA_north;
LAND=LAND_north;
nLAND=nLAND_north;

%% Plot sea level

figure
u = DATA(:,1);
u = real(u*exp(1i*mean(angle(u))));
v = zeros(150*720,1)+NaN;
v(nLAND) = u(:);
v = reshape(v,[720,150]);
v=flip(v.');
imagesc(v,'AlphaData',~isnan(v))
colormap(coolwarm)
clim([mean(u(:))-2*std(u(:)) mean(u(:))+2*std(u(:))])
set(gca,'Color',[1,1,1]*0.6)
colorbar('southoutside')
% axis equal
axis tight
grid off
box on
set(gca,'xticklabel',{[]})
set(gca,'yticklabel',{[]})
title('Sea Surface Height','interpreter','latex','fontsize',16)
exportgraphics(gcf,'sea_system.png');



return
%%
[G,K,L,PX] = kernel_dictionaries(DATA(:,1:end-1),DATA(:,2:end),'type',"poly_exp2",'N',250);

%% Time series
colors = distinguishable_colors(5);
figure
for jj = 1:5
    plot(PX(1:200,jj)-mean(PX(1:200,jj)),'color',colors(jj,:),'linewidth',2)
    hold on
end
axis off
exportgraphics(gcf,'sea_surface_time_series.png');


%% compute psuedospectra
x_pts = -1.5:0.02:1.5;    y_pts = -0.04:0.02:1.5; % use conjugate symmetry to only consider positive y
zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
z_pts=kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    z_pts=z_pts(:);
DIST=KoopPseudoSpec(G,K,L,z_pts);
LAM = eig(G\K);

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
exportgraphics(gcf,'sea_surface_pseudospectra.png');
% title(sprintf('$\\mathrm{Sp}_\\epsilon(\\mathcal{K})$'),'interpreter','latex','fontsize',18)


%% Find errors
Nvec = 60:5:360;
[E1,E2] = sea_level_err(DATA,Nvec);


%%
figure
semilogy(Nvec,E1,'b','linewidth',2,'markersize',20)
hold on
plot(Nvec,E2,'r:','linewidth',2,'markersize',20)
legend({'EDMD','proposed'},'fontsize',18,'interpreter','latex','location','best')
xlabel('Matrix Size','interpreter','latex','fontsize',18)
ylabel('Error','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=16;
% ylim([0.01,1])
exportgraphics(gcf,'sea_surface_error.png');


function [E1,E2] = sea_level_err(DATA,Nvec)


E1 = zeros(1,length(Nvec));
E2 = zeros(1,length(Nvec));

ct = 1;

for N = Nvec
	N
    [G,K,L,~,~] = kernel_dictionaries(DATA(:,1:end-1),DATA(:,2:end),'type',"poly_exp2",'N',N);%kResDMD(DATA(:,1:end-1),DATA(:,2:end),'type',"Laplacian",'N',N);
        
    %% Apply EDMD algorithm (which does not converge)
    % LAM = eig(K,'vector');

    [V,LAM,W] = eig(K,'vector'); W = conj(W);
    % R = real(sqrt(dot(V,L*V+V*diag(abs(LAM)).^2-K'*V*diag(LAM)-K*V*diag(conj(LAM)))./dot(V,V)));
    
    

    zpts = LAM(:);

    DIST = KoopPseudoSpec(G,K,L,zpts);
    E1(ct) = max(DIST);

    zpts = exp(1i*(0:0.01:1)*pi);
    % zpts = [zpts(:);LAM(:)];

    DIST = KoopPseudoSpec(G,K,L,zpts);

    
    %% Run new algorithm for finding spectrum
    spec = [];
    for jj = 1:length(zpts)
        if DIST(jj)<1
            II = find(abs(zpts(jj)-zpts)<2*DIST(jj));
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

