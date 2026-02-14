clear
close all
addpath(genpath('./algorithms'))
addpath(genpath('./data_sets'))


%% Load the data

load('ICE_DATA.mat')
%%
delays = 6; % number of time delays
X = DATA(:,delays:end-1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end


%% Run the algorithm

[~,K,L,PX,PY] = kernel_dictionaries(X(:,1:end-1),X(:,2:end),'type',"Gaussian");
[W,LAM,W2] = eig(K,'vector');

R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2))); % error bounds
[~,I] = sort(R,'ascend');
W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);

PXr = PX*W; PYr = PY*W;

N = knee_pt(R(I(max(1,length(I)-40):end-10)))+max(1,length(I)-40)-1;

figure
loglog(R,'.-','linewidth',2,'markersize',16)
hold on;
plot(N,R(N),'x','linewidth',2,'markersize',20)
xlabel('Number','interpreter','latex','fontsize',14)
title('Eigenpair Error','interpreter','latex','fontsize',14)
grid on
ax=gca; ax.FontSize=18;

%% Error bounds of EDMD eigenvalues

[~,I] = sort(R,'descend');
W = W(:,I); LAM = LAM(I)/max(abs(LAM)); W2 = W2(:,I); R = R(I);

figure
n2 = length(LAM);
scatter(angle(LAM(end-n2+1:end)),log(abs(LAM(end-n2+1:end))),1000*R(end-n2+1:end),R(end-n2+1:end),'filled','MarkerEdgeColor','k','LineWidth',0.01);
n2 = 17;
hold on
scatter(angle(LAM(end-n2+1:end)),log(abs(LAM(end-n2+1:end))),1000*R(end-n2+1:end),'b','filled','MarkerEdgeColor','y','LineWidth',0.02);
set(gca, 'GridColor', 'b')
hold on
load('cmap.mat')
colormap(cmap2); colorbar
xlabel('$\mathrm{arg}(\lambda)$','interpreter','latex','fontsize',18)
ylabel('$\log(|\lambda|)$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
xlim([-pi,pi])
ylim([-0.015,0.002])
clim([0,0.15])
grid on
box on

return
%% Koopman modes

[~,I] = sort(R,'ascend');
W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);
Phi = transpose((PX*W)\(X(1:97877,1:end-1)'));

c = vecnorm(Phi);
d = vecnorm(c.*(abs(LAM).^(0:100000))');
d = d.^2/(97877*504);
d = 1./d;

for jj =1:17
    figure
    u = real(Phi(1:size(DATA,1),jj));
    u = real(u*exp(1i*mean(angle(u))));
    
    v = zeros(432*432,1)+NaN;
    v(nLAND) = u(:);
    v = reshape(v,[432,432]);
    imagesc(v,'AlphaData',~isnan(v))
    colormap(coolwarm)
    set(gca,'Color',[1,1,1]*0.4)
    clim([-std(u),std(u)]*4+mean(u));
    title(sprintf('$|\\lambda|=%.3f,\\mathrm{arg}(\\lambda)/\\pi=%.3f,r=%.3f$',abs(LAM(jj)/max(abs(LAM))),angle(LAM(jj))/pi,R(jj)),'interpreter','latex','fontsize',15)
    axis equal
    axis tight
    grid on
    set(gca,'xticklabel',{[]})
    set(gca,'yticklabel',{[]})
    % exportgraphics(gcf,sprintf('sea_ice_gauss_%d.png',jj),'Resolution',300);

    pause(0.1)
    % close all
end


%% Forecast experiment
clear
close all

load('ICE_DATA.mat')


delays = 6; % number of time delays
INDX = (2005-1979)*12;
for jj = 13:511
    sigma2(jj) = sum((DATA(ACT{mod(jj-1,12)+1},jj)-mean(DATA(ACT{mod(jj-1,12)+1},jj-12:(-12):delays),2)).^2,1);
end
sigma2 = mean(sigma2(INDX+(1:12*11)));


X = DATA(:,delays:end-1); 
N0 = size(X,1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end

LAG = 3; % number of years we forecast for

er1 = zeros(11*11,LAG*12); er2 = er1; er3 = er1; er4 = er1;
ct = 1;

for yr = 2005:2015
    yr
    for mt = 1:12
        mt
        
        INDX = (yr-1979)*12+mt-delays; M = INDX-1;
        
        I2 = (mod(M,12)*0+1):M;
        X2 = X(:,[I2,I2(end)+1]); % training data
        mm = zeros(delays*N0,12);
        for jj = 1:12
            mm(:,jj) = mean(X2(:,jj:12:end),2);
        end
        Xper = repmat(mm,1,ceil(size(X2,2)/12)+LAG);
        X3 = X2 - Xper(:,1:size(X2,2));

        real_data = X(1:N0,INDX+(1:12*LAG)) - Xper(1:N0,length(I2)+1+(1:12*LAG));
        real_data2 = X(1:N0,INDX+(1:12*LAG));
        real_data3 = repmat(X(1:N0,INDX+1-12:INDX),1,ceil(size(real_data2,2)/12));
        real_data3 = real_data3(:,1:size(real_data2,2));

        % error bound approach
        [~,K,L,PXs,PYs] = kernel_dictionaries(X3(:,1:end-1),X3(:,(1+1):end),'type','Lorentzian');
        [W,LAM,W2] = eig(K,'vector');
        R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2)));
        [~,I] = sort(R,'ascend');
        N = knee_pt(R(I));
        PXr = PXs*W(:,I(1:N));
        PYr = PYs*W(:,I(1:N));

        c = ([PXr(1,:);PYr])\transpose(X3(1:N0,1:end));
        y1 = (real(transpose(transpose(PYr(end,:)).*(LAM(I(1:N)).^(1:12*LAG)))*c)');

        for tt = 1:size(y1,2)
            id = mod(mt+tt-2,12)+1;
            er1(ct,tt) = sum(abs(y1(ACT{id},tt)-real_data(ACT{id},tt)).^2,1)/sigma2;
        end
        
        % Compare to DMD
        
        [U,S,~] = svd(X3(:,1:end),'econ');
        
        clear Er
        S = diag(S);
        r = knee_pt(S)+22;
                   
        PXs = X3(:,1:end-1)'*U(:,1:r);
        PYs = X3(:,2:end)'*U(:,1:r);
        K = PXs\PYs;
        [W,LAM,W2] = eig(K,'vector');
        PXr = PXs*W; PYr = PYs*W;
        
        c = ([PXr(1,:);PYr])\transpose(X3(1:N0,1:end));
        y2 = (real(transpose(transpose(PYr(end,:)).*(LAM.^(1:12*LAG)))*c)');

        for tt = 1:size(y2,2)
            id = mod(mt+tt-2,12)+1;
            er2(ct,tt) = sum(abs(y2(ACT{id},tt)-real_data(ACT{id},tt)).^2,1)/sigma2;
            er3(ct,tt) = sum(abs(real_data(ACT{id},tt)).^2,1)/sigma2;
            er4(ct,tt) = sum(abs(real_data2(ACT{id},tt)-real_data3(ACT{id},tt)).^2,1)/sigma2;
        end
        ct = ct+1;
    end


end

figure
plot(mean(er2(1:ct-1,:),1),'-','linewidth',2,'color',colors(4,:))
hold on
plot(mean(er1(1:ct-1,:),1),'-','linewidth',2,'color',colors(3,:))
plot(mean(er3(1:ct-1,:),1),':','linewidth',2,'color','k')
plot(mean(er4(1:ct-1,:),1),':','linewidth',2,'color','r')
grid on

xlabel('Lead Time (Months)','interpreter','latex','fontsize',14)
ylabel('Forecast Error','interpreter','latex','fontsize',14)
legend({'DMD','Proposed Method','Periodic','Persistence'},'interpreter','latex','fontsize',12,'location','best')
xlim([0,12*LAG+1])
xlim([0,12*LAG+1])
hold off


%% Plot the sea ice extent

clear
close all
addpath(genpath('./algorithms'))
addpath(genpath('./data_sets'))

load('ICE_DATA.mat')

%%
delays = 6; % number of time delays
X = DATA(:,delays:end-1);
for jj = delays-1:(-1):1
    X = [X;DATA(:,jj:end-(delays-jj+1))];
end

% Run the algorithm for 10 year blocks
for block = 1:4
    It = (1:12*10) + (block-1)*10*12 +6;
    
    [~,K,L,PX,PY] = kernel_dictionaries(X(:,It),X(:,It+1),'type',"Gaussian");
    
    [W,LAM,W2] = eig(K,'vector');
    
    R = (sqrt(real(diag(W2'*L*W2)./diag(W2'*W2)-abs(LAM).^2))); 
    [~,I] = sort(R,'ascend');
    W = W(:,I); LAM = LAM(I); W2 = W2(:,I); R = R(I);
    
    PXr = PX*W; PYr = PY*W;
    Phi = transpose((PX*W)\(X(1:97877,It)'));
    R(1:2)
    
    for jj =1:2
        figure
        u = real(Phi(1:size(DATA,1),jj));
        u = abs(u*exp(1i*mean(angle(u))));
        
        v = zeros(432*432,1)+NaN;
        v(nLAND) = u(:);
        v = reshape(v,[432,432]);
        imagesc(v,'AlphaData',~isnan(v))
        colormap(brighten(coolwarm,0.1))
        % colormap default
        set(gca,'Color',[1,1,1]*0.5)
        clim([0,max(u)*1]);
        axis equal
        axis tight
        grid on
        set(gca,'xticklabel',{[]})
        set(gca,'yticklabel',{[]})
    end
end

t = DATA; clear DATA
t(t<=15)=0; t(t>15)=1;
t=sum(t,1)*625;

f=figure;
plot(t,'linewidth',2)
hold on
t2=movmean(t,12);
t2(1:6)="NaN"; t2(end-5:end)="NaN";
plot(t2,'r','linewidth',3)

xticks(13:12*5:600)
xticklabels(1980:5:2030)
xlim([1,length(t)])
title('Sea Ice Extent','interpreter','latex','fontsize',14)
ylabel('km$^2$','interpreter','latex','fontsize',14)
grid on
f.Position=[360  160  700.0000  350];





