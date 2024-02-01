%grundtruth,predict, result
load Tumor_testData_10.mat;
load Testset_Result_Vit_epoch0500(1).mat ;


sel_parMap = test_ParMap; % ground truth parameters
clear vp ps
for i = 1:size(sel_parMap,4)
    if mod(sel_parMap(2,2,end,i),1)==0
        vp(i) = squeeze(sel_parMap(2,2,2,i));
        ps(i) = squeeze(sel_parMap(2,2,3,i));
    else
        vp(i) = squeeze(sel_parMap(2,2,1,i));
        ps(i) = squeeze(sel_parMap(2,2,3,i));
    end
end
vp = vp(:,1:5000);
ps = ps(:,1:5000);
%% est. Parameter with prediction and compare

data = squeeze(test_Data(5,:,:,:)); % dim=[nt x #Patch]
data = data(:,1:5000); 
target = test_CIFs(:,1:5000); % dim=[nt x #Patch]% from the dataset
t = linspace(0,10*60,size(data,1));
% change this to prediction from network
prediction = result.'; % dim=[nt x #Patch]（need paar to paar）

clear par_targ par_pred
for i = 1:size(data,2)
    cts = data(:,i);
    par_targ(i,:) = cts2Ktrans_GPM(cts,target(:,i),t);
    par_pred(i,:) = cts2Ktrans_GPM(cts,prediction(:,i),t);
end

%% Excluding patches w/ zero

logIdx = (vp==0) | (ps==0);
vp(logIdx) = [];
ps(logIdx) = [];
par_targ(logIdx,:)=[];
par_pred(logIdx,:)=[];


%% Piece-wise error
vp_error_targ = abs(vp(:)-par_targ(:,1))./vp(:);
ps_error_targ = abs(ps(:)-par_targ(:,2))./ps(:);

vp_error_pred = abs(vp(:)-par_pred(:,1))./vp(:);
ps_error_pred = abs(ps(:)-par_pred(:,2))./ps(:);



%% Display

close all;
figure;
subplot(2,1,1);
bar(1,median(vp_error_targ),'red')
hold on;
er1 = errorbar(1,median(vp_error_targ),quantile(vp_error_targ,0.25),quantile(vp_error_targ,0.75),'Color','red');
bar(2,median(vp_error_pred),'blue')
er2 = errorbar(2,median(vp_error_pred),quantile(vp_error_pred,0.25),quantile(vp_error_pred,0.75),'Color','blue');
set(gca,'XTick',1:2,'Fontsize',13,'FontWeight','bold');
set(gca,'XTickLabel',{''})
ylabel('Vp Error (%)')

subplot(2,1,2);
bar(1,median(ps_error_targ),'red')
hold on;
er1 = errorbar(1,median(ps_error_targ),quantile(ps_error_targ,0.25),quantile(ps_error_targ,0.75),'Color','red');
bar(2,median(ps_error_pred),'blue')
er2 = errorbar(2,median(ps_error_pred),quantile(ps_error_pred,0.25),quantile(ps_error_pred,0.75),'Color','blue');
set(gca,'XTick',1:2,'Fontsize',13,'FontWeight','bold');
set(gca,'XTickLabel',{'GT-CIF','pred-CIF'})
ylabel('PS Error (%)')




