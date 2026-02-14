clear
close all

addpath(genpath('./datasets'))
addpath(genpath('./algorithms'))

load('Vorticity_data.mat')

%% Perform EDMD with POD modes

M = 24*5;
ind = (1:M);
X = VORT(:,ind); 
Y = VORT(:,ind+1);
[U,~,~] = svd(X,'econ');
PX = X'*U;
PY = Y'*U;
return
% %% Time series
% 
% colors = distinguishable_colors(5);
% figure
% for jj = 1:5
%     plot(PX(:,jj)-mean(PX(:,jj)),'color',colors(jj,:),'linewidth',2)
%     hold on
% end
% axis off
% exportgraphics(gcf,'cyclinder_time_series.png');
% return

%% Plot vorticity field

C = zeros(800*200,1)+NaN;
    C(II)=real(VORT(:,1));
C = reshape(C,[800,200]);    

vv=-0.025:0.05:1.25;

a = prctile(C(~isnan(C)),99.9);
b = prctile(C(~isnan(C)),0.1);
C(~isnan(C)) = max(C(~isnan(C)),b);
C(~isnan(C)) = min(C(~isnan(C)),a);

c=(a+b)/2;
figure
[~,h]=contourf(Xgrid,Ygrid+0.06,C,vv*(a-b)+b,'edgecolor','k');
h.LineWidth = 0.2;
colormap(magma)
clim([min(VORT(:,1)),max(VORT(:,1))]/5)

% axis equal
hold on
fill(1.1/2*cos(0:0.01:2*pi),1.1/2*sin(0:0.01:2*pi),'w','edgecolor','none')
plot(1.1/2*cos(0:0.01:2*pi),1.1/2*sin(0:0.01:2*pi),'k','linewidth',1)
xlim([-1,8])
title('Vorticity Field','interpreter','latex','fontsize',16)
ylim([-2,2])
% xlabel('$x/D$','interpreter','latex','fontsize',18)
% ylabel('$y/D$','interpreter','latex','fontsize',18)
hold off
axis off
ax=gca; ax.FontSize=18;
exportgraphics(gcf,'cyclinder_system.png');

%% Find errors
Nvec = 1:50;
[E1,E2] = cylinder_err(PX,PY,Nvec);

%%
figure
semilogy(Nvec,E1,'b','linewidth',2,'markersize',20)
hold on
plot(Nvec,E2,'r:','linewidth',2,'markersize',20)
legend({'EDMD','proposed'},'fontsize',18,'interpreter','latex','location','northeast')
xlabel('Matrix Size','interpreter','latex','fontsize',18)
ylabel('Error','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=16;
exportgraphics(gcf,'cyclinder_error.png');

return

%% Compute pseudospectra
x_pts = -1.5:0.02:1.5;    y_pts = -0.04:0.02:1.5; % use conjugate symmetry to only consider positive y
zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
DIST = KoopPseudoSpecQR(PX(:,1:50),PY(:,1:50),1/M,zpts,'Parallel','off');
DIST = max(DIST,abs(1-abs(zpts)));
LAM = eig(PX(:,1:50)\PY(:,1:50));

%% Plot the pseudospectrum and EDMD eigenvalues
v=(10.^(-2:0.1:-0.5));
close all
DIST = reshape(DIST,length(y_pts),length(x_pts));

figure
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
hold on
contourf(reshape(real(zpts),length(y_pts),length(x_pts)),-reshape(imag(zpts),length(y_pts),length(x_pts)),log10(max(min(v),real(DIST))),log10(v));
cbh=colorbar;
cbh.Ticks=log10([0.01,0.1,1]);
cbh.TickLabels=[0.01,0.1,1];
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
exportgraphics(gcf,'cyclinder_pseudospectra.png');
% title(sprintf('$\\mathrm{Sp}_\\epsilon(\\mathcal{K})$'),'interpreter','latex','fontsize',18)



function [E1,E2] = cylinder_err(PX,PY,Nvec)


E1 = zeros(1,length(Nvec));
E2 = zeros(1,length(Nvec));
M = length(PX(:,1));

ct = 1;

for N = Nvec

    K = PX(:,1:N)\PY(:,1:N);
    LAM = eig(K,'vector');
    LAM = LAM(real(LAM)>0);
    R = KoopPseudoSpecQR(PX(:,1:N),PY(:,1:N),1/M,LAM,'Parallel','off');
    R = max(R,abs(1-abs(LAM)));
    E1(ct) = max(R);

    zpts = LAM(:);
    zpts = zpts(abs(abs(zpts)-1)<0.05);

    DIST = KoopPseudoSpecQR(PX(:,1:N),PY(:,1:N),1/M,zpts,'Parallel','off');
    DIST = max(DIST,abs(1-abs(zpts)));
    
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

