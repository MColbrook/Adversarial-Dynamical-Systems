clear
close all

load('heart_data.mat')
PX = H(:,1:end-1); PX = PX';
PY = H(:,2:end); PY = PY';

return
%% Find errors
Nvec = 10:10:500;
[E1,E2] = rossler_err(PX,PY,Nvec);

%%
figure
loglog(Nvec,E1,'b','linewidth',2,'markersize',20)
hold on
plot(Nvec,E2,'r:','linewidth',2,'markersize',20)
legend({'EDMD','proposed'},'fontsize',18,'interpreter','latex','location','best')
xlabel('Matrix Size','interpreter','latex','fontsize',18)
ylabel('Error','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=16;
exportgraphics(gcf,'ECG_error.png');

%% Time series
figure
plot(PX(1:2000),'k','linewidth',2)
axis off
exportgraphics(gcf,'ECG_series.png');

%% Compute pseudospectra
N2 = 50;
M = length(PX(:,1));
x_pts = -1.5:0.02:1.5;    y_pts = -0.04:0.02:1.5; % use conjugate symmetry to only consider positive y
zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
DIST = KoopPseudoSpecQR(PX(:,1:N2),PY(:,1:N2),1/M,zpts,'Parallel','off');
LAM = eig(PX(:,1:N2)\PY(:,1:N2));

%% Plot the pseudospectrum and EDMD eigenvalues
v=(10.^(-2:0.1:-0.5));
close all
DIST = reshape(DIST,length(y_pts),length(x_pts));

figure
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
hold on
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),-reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
cbh=colorbar;
cbh.Ticks=log10([0.001,0.01,0.1,1]);
cbh.TickLabels=[0.001,0.01,0.1,1];
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
exportgraphics(gcf,'ECG_pseudospectra.png');
% title(sprintf('$\\mathrm{Sp}_\\epsilon(\\mathcal{K})$'),'interpreter','latex','fontsize',18)














function [E1,E2] = rossler_err(PX,PY,Nvec)


E1 = zeros(1,length(Nvec));
E2 = zeros(1,length(Nvec));
M = length(PX(:,1));

ct = 1;

for N = Nvec
	N
    Id = 1:N;
        
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

