function [E1,E2] = duffing_err(Nvec,ALPHA)

%% Set parameters
M1=10^4; M2=5;  % number of data points
delta_t = 0.3;  % time step
ODEFUN = @(t,y) [y(2);-ALPHA*y(2)+y(1)-y(1).^3];
options = odeset('RelTol',1e-12,'AbsTol',1e-12);
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

X2 = X;
Y2 = Y;

X = zeros(2,M1*M2);
Y=[];
for jj=1:M1
    Y0=(rand(2,1)-0.5)*4;
    [~,Y1]=ode45(ODEFUN,[0 0.000001 (1:(3+M2))*delta_t],Y0,options);
    Y1=Y1';
    X(:,(jj-1)*M2+1:jj*M2) = Y1(:,[1,3:M2+1]);
    Y(:,(jj-1)*M2+1:jj*M2) = Y1(:,3:M2+2);
end

X3 = zeros(2,M1*M2);
Y3=[];
for jj=1:M1
    Y0=(rand(2,1)-0.5)*4;
    [~,Y1]=ode45(ODEFUN,[0 0.000001 (1:(3+M2))*delta_t],Y0,options);
    Y1=Y1';
    X3(:,(jj-1)*M2+1:jj*M2) = Y1(:,[1,3:M2+1]);
    Y3(:,(jj-1)*M2+1:jj*M2) = Y1(:,3:M2+2);
end

M = M1*M2;
d = mean(vecnorm(X-mean(X,2)));             % scaling for radial functions

E1 = zeros(1,length(Nvec));
E2 = zeros(1,length(Nvec));

ct = 1;

for N = Nvec
	N

    [~,C] = kmeans([X2';Y2'],N,'MaxIter',500);    % centers for radial functions
    
    %% Produce the trajectory data
    
    
    PX = zeros(M,N); PY = zeros(M,N);           % dictionary evaluated at data points
    PX2 = PX; PY2 = PY;
    
    for j = 1:N
        R = sqrt((X(1,:)-C(j,1)).^2+(X(2,:)-C(j,2)).^2);
        PX(:,j) = PHI(R(:)/d);
        R = sqrt((Y(1,:)-C(j,1)).^2+(Y(2,:)-C(j,2)).^2);
        PY(:,j) = PHI(R(:)/d);

        R = sqrt((X3(1,:)-C(j,1)).^2+(X3(2,:)-C(j,2)).^2);
        PX2(:,j) = PHI(R(:)/d);
        R = sqrt((Y3(1,:)-C(j,1)).^2+(Y3(2,:)-C(j,2)).^2);
        PY2(:,j) = PHI(R(:)/d);
    end
    
    %% Apply EDMD algorithm (which does not converge)
    K = PX(1:M,:)\PY(1:M,:);
    [V,LAM] = eig(K,'vector');
    R = vecnorm(PY2(1:M,:)*V-PX2(1:M,:)*V*diag(LAM))./vecnorm(PX2(1:M,:)*V); % residual errors of EDMD
    E1(ct) = max(R);

    
    %% Compute errors over a grid
    x_pts = -0.5:0.02:1;    y_pts = 0:0.02:1; % use conjugate symmetry to only consider positive y
    zpts = kron(x_pts,ones(length(y_pts),1))+1i*kron(ones(1,length(x_pts)),y_pts(:));    zpts = zpts(:);
    DIST = KoopPseudoSpecQR(PX,PY,1/M,zpts,'Parallel','on');
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
    E2(ct) = max(DIST(spec));
    ct = ct+1;
end


end