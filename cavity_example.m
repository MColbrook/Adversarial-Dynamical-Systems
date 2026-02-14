clear
close all

%% Apply the RAGE theorem

% load the data. Data can be downloaded here: https://ucsb.app.box.com/s/tbtdij1r4z5qt7o4cbgmolx28do98xci/folder/46152793169
load('cavity_KE.mat')

% subtract the mean (eigenvalue 1 with constant eigenfunction)
KE1 = KE1 - mean(KE1);
KE2 = KE2 - mean(KE2);
KE3 = KE3 - mean(KE3);
KE4 = KE4 - mean(KE4);

N = 20; % total number of delay embeddings
h = 10; % relative time step for delay embedding
M = 10000; % number of data points
nvec = [1,10,20]; % second limit
L = 10000; % first limit


%% Use delay embedding to build the Koopman matrices

PX1 = zeros(20000-1-h*N,N); PX2 = PX1; PX3 = PX1; PX4 = PX1;
for j=1:N
    PX1(:,j)=KE1((1:(20000-1-h*N))+h*(j-1));
    PX2(:,j)=KE2((1:(20000-1-h*N))+h*(j-1));
    PX3(:,j)=KE3((1:(20000-1-h*N))+h*(j-1));
    PX4(:,j)=KE4((1:(20000-1-h*N))+h*(j-1));
end

%% Apply the RAGE theorem
[PX1,~] = qr(PX1(1:M,:),'econ');
[PX2,~] = qr(PX2(1:M,:),'econ');
[PX3,~] = qr(PX3(1:M,:),'econ');
[PX4,~] = qr(PX4(1:M,:),'econ');

C = zeros(4,max(nvec),L+1);

c1 = erg_prod(KE1(1:M),KE1(1:M)); % normalise the flow
c2 = erg_prod(KE2(1:M),KE2(1:M));
c3 = erg_prod(KE3(1:M),KE3(1:M));
c4 = erg_prod(KE4(1:M),KE4(1:M));

G1 = zeros(M,L+1); G2 = G1; G3 = G1; G4 = G1;
for jj = 0:L % flow jj time steps forward
    G1(:,jj+1)=KE1((1:M)+jj)/sqrt(c1);
    G2(:,jj+1)=KE2((1:M)+jj)/sqrt(c2);
    G3(:,jj+1)=KE3((1:M)+jj)/sqrt(c3);
    G4(:,jj+1)=KE4((1:M)+jj)/sqrt(c4);
end
G1 = PX1'*G1; G2 = PX2'*G2; G3 = PX3'*G3; G4 = PX4'*G4;

for nn = nvec
    g = PX1(:,1:nn)*G1(1:nn,:);
    C(1,nn,:)=dot(g,g)/size(g,1);
    g = PX2(:,1:nn)*G2(1:nn,:);
    C(2,nn,:)=dot(g,g)/size(g,1);
    g = PX3(:,1:nn)*G3(1:nn,:);
    C(3,nn,:)=dot(g,g)/size(g,1);
    g = PX4(:,1:nn)*G4(1:nn,:);
    C(4,nn,:)=dot(g,g)/size(g,1);
end

P = cumsum(min(C,1),3);

%% Plot the results
% close all
figure
loglog(1:L+1,squeeze(P(1,nvec,:))./(1:L+1),'linewidth',2)
legend({sprintf('$n=%d$',nvec(1)),sprintf('$n=%d$',nvec(2)),sprintf('$n=%d$',nvec(3))},'fontsize',16,'interpreter','latex','location','southwest')
xlabel('$L$','interpreter','latex','fontsize',18)
title('$Re=13000$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
ylim([0.1,1])
grid minor
% exportgraphics(gcf,'RAGE1.png','ContentType','vector','BackgroundColor','none')

figure
loglog(1:L+1,squeeze(P(2,nvec,:))./(1:L+1),'linewidth',2)
legend({sprintf('$n=%d$',nvec(1)),sprintf('$n=%d$',nvec(2)),sprintf('$n=%d$',nvec(3))},'fontsize',16,'interpreter','latex','location','southwest')
xlabel('$L$','interpreter','latex','fontsize',18)
title('$Re=16000$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
ylim([0.1,1])
grid minor
% exportgraphics(gcf,'RAGE2.png','ContentType','vector','BackgroundColor','none')

figure
loglog(1:L+1,squeeze(P(3,nvec,:))./(1:L+1),'linewidth',2)
legend({sprintf('$n=%d$',nvec(1)),sprintf('$n=%d$',nvec(2)),sprintf('$n=%d$',nvec(3))},'fontsize',16,'interpreter','latex','location','southwest')
xlabel('$L$','interpreter','latex','fontsize',18)
title('$Re=19000$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
ylim([0.1,1])
grid minor
% exportgraphics(gcf,'RAGE3.png','ContentType','vector','BackgroundColor','none')

figure
loglog(1:L+1,squeeze(P(4,nvec,:))./(1:L+1),'linewidth',2)
legend({sprintf('$n=%d$',nvec(1)),sprintf('$n=%d$',nvec(2)),sprintf('$n=%d$',nvec(3))},'fontsize',16,'interpreter','latex','location','southwest')
xlabel('$L$','interpreter','latex','fontsize',18)
title('$Re=30000$','interpreter','latex','fontsize',18)
ax=gca; ax.FontSize=18;
ylim([0.1,1])
grid minor
% exportgraphics(gcf,'RAGE4.png','ContentType','vector','BackgroundColor','none')



function z = erg_prod(x,y)
 z = (y(:)'*x(:))/length(x);
end

