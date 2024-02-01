function res = cts2Ktrans_GPM(Cts,aif,time)

nt = length(Cts);

%cts_vec = reshape(Cts, [numVox,nt]);

C = Cts(:) ./ aif(:);

xx = zeros(size(aif));


for i = 2:length(aif)
    xx(i) = trapz(time(1:i),aif(1:i))/aif(i);  
end


%xx = cumsum(aif);

xx(isnan(xx)) = 0;

[~,peak] = max(aif);

xx(1:peak-1) = [];
C(1:peak-1) = [];

xx = xx(:);

A = [ones(length(xx),1),xx]; % A = nt x 2 (1-Ktrans, 2-Vp)
%A = [ones(nt,1),xx]; % A = nt x 2 (1-Ktrans, 2-Vp)
%C = permute(C,[2,1]); % C => nt x numVox

X = pinv(A)*C;
% opts1=  optimset('display','off');
% X = lsqnonneg(A,C,opts1);


Vp = X(1); %Vp
Kt = X(2); %PS
res = [Vp,Kt];