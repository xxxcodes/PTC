% For Peer-review
% Reproducing the reulsts of Prototype Theory Classifier paper


if ~isfolder('datasets')
    disp('You have not downloaded the datasets yet. Please choose one of the following options:');
    disp('1) Run the script "DownloadDatasets.m", which will automatically download the datasets from the repositories and process them into the required format for the Reproduce.m script.');
    disp('2) Download the datasets in MATLAB-ready format using the following link. Then, create a folder named "datasets" and extract the contents of the ZIP file into that folder.');
    disp('   https://www.dropbox.com/scl/fi/d2xj0vycgtxcamkga8n2e/datasets.zip?rlkey=n6ig3azeyw1vuc9vi0kwbcv12&st=x6vwb30o&dl=0');
    return
end

%% Training for PTC, NCC, SNCC and LASSO

clear
load datasets

for n=1:height(datasets)
    clearex('n','datasets');

load (['datasets/',datasets.Name{n}])
disp(['Dataset ',num2str(n),':',datasets.Name{n},'...']);
% for NCC and SNCC
[X_train_N,c,s]=normalize(X_train);
[X_test_N,c,s]=normalize(X_test,'center',c,'scale',s);

%PTC Model
disp('Training PTC ... ');
tStart = cputime;
PTC_Mdl=fitPTC(X_train,y_train);
cputime_train_PTC = cputime - tStart;

PTC_numFeatures=numel(PTC_Mdl.PrototypeFeatureIDs);

tStart = cputime;
y_pred_PTC=PTC_Mdl.My(knnsearch(PTC_Mdl.MX,X_test(:,PTC_Mdl.PrototypeFeatureIDs),'K',1));
cputime_test_PTC = cputime - tStart;

%NCC
disp('Training NCC ... ');
tStart = cputime;
NCC_Mdl=fitNCC(X_train_N,y_train);
cputime_train_NCC = cputime - tStart;
tStart = cputime;
y_pred_NCC = predNCC(NCC_Mdl, X_test_N);
cputime_test_NCC = cputime - tStart;
NCC_numFeatures=size(X_train,2);

%SNCC
disp('Training SNCC ... ');
rng(42)
cv = cvpartition(size(X_train, 1), 'HoldOut', 0.2);
TrainTrainIdx = training(cv);
TrainValidIdx = test(cv);
X_train_train = X_train(TrainTrainIdx, :);
y_train_train=y_train(TrainTrainIdx);
X_train_valid = X_train(TrainValidIdx, :);
y_train_valid=y_train(TrainValidIdx);
[X_train_train_N,c,s]=normalize(X_train_train);
[X_train_valid_N,c,s]=normalize(X_train_valid,'center',c,'scale',s);

tStart = cputime;
for TopF=size(X_train,2):-1:1
    R=size(X_train,2)-TopF;
    [theta_plus_L2, theta_minus_L2] = fitSNCC(X_train_train_N', y_train_train, R);
    y_pred_val_SNCC = predSNCC(X_train_valid_N', theta_plus_L2, theta_minus_L2,2);
    AccVal_SNCC(TopF)=mean(y_pred_val_SNCC==y_train_valid);
end
[~,bestTopF]=max(AccVal_SNCC);
bestR=size(X_train,2)-bestTopF;

[theta_plus_L2, theta_minus_L2] = fitSNCC(X_train_N', y_train, bestR);
cputime_train_SNCC = cputime - tStart;

SNCC_Mdl.theta_plus_L2=theta_plus_L2;
SNCC_Mdl.theta_minus_L2=theta_minus_L2;
SNCC_Mdl.center_sparse=abs(theta_plus_L2-theta_minus_L2);
SNCC_Mdl.relevant_features=find(SNCC_Mdl.center_sparse~=0);
SNCC_Mdl.R=R;

SNCC_numFeatures=numel(SNCC_Mdl.relevant_features);


tStart = cputime;
y_pred_SNCC = predSNCC(X_test_N', SNCC_Mdl.theta_plus_L2, SNCC_Mdl.theta_minus_L2,2);
cputime_test_SNCC = cputime - tStart;

%LASSO
disp('Training LASSO ... ');


tStart = cputime;
rng(42)
lambdas = logspace(-4, 0, 100);
[BAll, FitInfoAll] = lassoglm(X_train, y_train, 'binomial', 'Lambda', lambdas,'Standardize',true);
lambda = FitInfoAll.Lambda;
deviance = FitInfoAll.Deviance;

clear BIC;
% Compute BIC for each lambda
n_samples = size(X_train, 1);
p = size(X_train, 2);
BIC = zeros(size(lambda));
for i = 1:length(lambda)
    % Number of non-zero coefficients
    numParams(i) = sum(BAll(:, i) ~= 0);
    % Compute BIC: BIC = n*log(deviance/n) + k*log(n)
    BIC(i) = n_samples * log(deviance(i) / n_samples) + numParams(i) * log(n_samples);
end
[minBIC, idxMinBIC] = min(BIC);
best_lambda = lambda(idxMinBIC);
B=BAll(:,idxMinBIC);
Intercept=FitInfoAll.Intercept(idxMinBIC);
LASSO_numFeatures=numParams(idxMinBIC);
cputime_train = cputime - tStart;

tStart = cputime;
y_pred_test_prob = glmval([Intercept; B], X_test, 'logit');
y_pred_LASSO = y_pred_test_prob > 0.5;
cputime_test_LASSO = cputime - tStart;

save(['results/',num2str(n)]);


end

%% Process The Results for Generating Fig. 3 in the paper
clear
load datasets

for n=1:height(datasets)
load(['results/',num2str(n)]);
   if n>=14 & n<=18
       posclass=0;
   else
       posclass=1;
   end
   p=size(X_train,2);
   D_PTC=100*PTC_numFeatures/p;
   D_NCC=100*NCC_numFeatures/p;
   D_SNCC=100*SNCC_numFeatures/p;
   D_LASSO=100*LASSO_numFeatures/p;
 
    [recall_PTC,bAcc_PTC,F1_PTC]=eval_perf(double(y_test),double(y_pred_PTC),posclass);
    [recall_NCC,bAcc_NCC,F1_NCC]=eval_perf(double(y_test),double(y_pred_NCC),posclass);
    [recall_SNCC,bAcc_SNCC,F1_SNCC]=eval_perf(double(y_test),double(y_pred_SNCC),posclass);
    [recall_LASSO,bAcc_LASSO,F1_LASSO]=eval_perf(double(y_test),double(y_pred_LASSO),posclass);


    D_vec=[D_PTC,D_NCC,D_SNCC,D_LASSO];
    D_u=unique(D_vec);
    D_u=sort(D_u,'asc');
    for j=1:numel(D_vec)
        D_rank(n,j)=find(D_u==D_vec(j));
    end
   	recall_vec=[recall_PTC,recall_NCC,recall_SNCC,recall_LASSO];
    recall_u=unique(recall_vec);
    recall_u=sort(recall_u,'descend');
    for j=1:numel(recall_vec)
        recall_rank(n,j)=find(recall_u==recall_vec(j));
    end

    bAcc_vec=[bAcc_PTC,bAcc_NCC,bAcc_SNCC,bAcc_LASSO];
    bAcc_u=unique(bAcc_vec);
    bAcc_u=sort(bAcc_u,'descend');
    for j=1:numel(bAcc_vec)
        bAcc_rank(n,j)=find(bAcc_u==bAcc_vec(j));
    end

    F1_vec=[F1_PTC,F1_NCC,F1_SNCC,F1_LASSO];
    F1_u=unique(F1_vec);
    F1_u=sort(F1_u,'descend');
    for j=1:numel(F1_vec)
        F1_rank(n,j)=find(F1_u==F1_vec(j));
    end

    cputime_train_vec=[cputime_train_PTC,cputime_train_NCC,cputime_train_SNCC,cputime_train];
    cputime_train_u=unique(cputime_train_vec);
    cputime_train_u=sort(cputime_train_u,'asc');
    for j=1:numel(cputime_train_vec)
        cputime_train_rank(n,j)=find(cputime_train_u==cputime_train_vec(j));
    end


    row(n,:)=[D_PTC D_NCC D_SNCC D_LASSO 100*recall_PTC,100*recall_NCC,100*recall_SNCC 100*recall_LASSO, 100*bAcc_PTC,100*bAcc_NCC 100*bAcc_SNCC,100*bAcc_LASSO, 100*F1_PTC 100*F1_NCC,100*F1_SNCC,100*F1_LASSO cputime_train_PTC cputime_train_NCC cputime_train_SNCC cputime_train];
end
cnt=size(row,1);
ranks=horzcat(D_rank,recall_rank,bAcc_rank,F1_rank,cputime_train_rank);

result=array2table(row,"VariableNames",{'D_PTC','D_NCC','D_SNCC','D_LASSO','Recall_PTC','Recall_NCC','Recall_SNCC','Recall_LASSO','bAcc_PTC','bAcc_NCC','bAcc_SNCC','bAcc_LASSO','F1_PTC','F1_NCC','F1_SNCC','F1_LASSO','TrainCPU_PTC','TrainCPU_NCC','TrainCPU_SNCC','TrainCPU_LASSO'});
result=horzcat(datasets(1:cnt,1:4),result);

win_D_PTCvsNCC=sum(result.D_PTC<=result.D_NCC);
win_D_PTCvsSNCC=sum(result.D_PTC<=result.D_SNCC);
win_D_PTCvsLASSO=sum(result.D_PTC<=result.D_LASSO);
loose_D_PTCvsNCC=sum(result.D_PTC>=result.D_NCC);
loose_D_PTCvsSNCC=sum(result.D_PTC>=result.D_SNCC);
loose_D_PTCvsLASSO=sum(result.D_PTC>=result.D_LASSO);

delta_D_PTCvsNCC=mean(result.D_PTC)-mean(result.D_NCC);
delta_D_PTCvsSNCC=mean(result.D_PTC)-mean(result.D_SNCC);
delta_D_PTCvsLASSO=mean(result.D_PTC)-mean(result.D_LASSO);

win_Recall_PTCvsNCC=sum(result.Recall_PTC>=result.Recall_NCC);
win_Recall_PTCvsSNCC=sum(result.Recall_PTC>=result.Recall_SNCC);
win_Recall_PTCvsLASSO=sum(result.Recall_PTC>=result.Recall_LASSO);
loose_Recall_PTCvsNCC=sum(result.Recall_PTC<=result.Recall_NCC);
loose_Recall_PTCvsSNCC=sum(result.Recall_PTC<=result.Recall_SNCC);
loose_Recall_PTCvsLASSO=sum(result.Recall_PTC<=result.Recall_LASSO);

delta_Recall_PTCvsNCC=mean(result.Recall_PTC)-mean(result.Recall_NCC);
delta_Recall_PTCvsSNCC=mean(result.Recall_PTC)-mean(result.Recall_SNCC);
delta_Recall_PTCvsLASSO=mean(result.Recall_PTC)-mean(result.Recall_LASSO);

win_bAcc_PTCvsNCC=sum(result.bAcc_PTC>=result.bAcc_NCC);
win_bAcc_PTCvsSNCC=sum(result.bAcc_PTC>=result.bAcc_SNCC);
win_bAcc_PTCvsLASSO=sum(result.bAcc_PTC>=result.bAcc_LASSO);
loose_bAcc_PTCvsNCC=sum(result.bAcc_PTC<=result.bAcc_NCC);
loose_bAcc_PTCvsSNCC=sum(result.bAcc_PTC<=result.bAcc_SNCC);
loose_bAcc_PTCvsLASSO=sum(result.bAcc_PTC<=result.bAcc_LASSO);

delta_bAcc_PTCvsNCC=mean(result.bAcc_PTC)-mean(result.bAcc_NCC);
delta_bAcc_PTCvsSNCC=mean(result.bAcc_PTC)-mean(result.bAcc_SNCC);
delta_bAcc_PTCvsLASSO=mean(result.bAcc_PTC)-mean(result.bAcc_LASSO);

win_F1_PTCvsNCC=sum(result.F1_PTC>=result.F1_NCC);
win_F1_PTCvsSNCC=sum(result.F1_PTC>=result.F1_SNCC);
win_F1_PTCvsLASSO=sum(result.F1_PTC>=result.F1_LASSO);
loose_F1_PTCvsNCC=sum(result.F1_PTC<=result.F1_NCC);
loose_F1_PTCvsSNCC=sum(result.F1_PTC<=result.F1_SNCC);
loose_F1_PTCvsLASSO=sum(result.F1_PTC<=result.F1_LASSO);

delta_F1_PTCvsNCC=mean(result.F1_PTC)-mean(result.F1_NCC);
delta_F1_PTCvsSNCC=mean(result.F1_PTC)-mean(result.F1_SNCC);
delta_F1_PTCvsLASSO=mean(result.F1_PTC)-mean(result.F1_LASSO);


win_TrainCPU_PTCvsNCC=sum(result.TrainCPU_PTC<=result.TrainCPU_NCC);
win_TrainCPU_PTCvsSNCC=sum(result.TrainCPU_PTC<=result.TrainCPU_SNCC);
win_TrainCPU_PTCvsLASSO=sum(result.TrainCPU_PTC<=result.TrainCPU_LASSO);
loose_TrainCPU_PTCvsNCC=sum(result.TrainCPU_PTC>=result.TrainCPU_NCC);
loose_TrainCPU_PTCvsSNCC=sum(result.TrainCPU_PTC>=result.TrainCPU_SNCC);
loose_TrainCPU_PTCvsLASSO=sum(result.TrainCPU_PTC>=result.TrainCPU_LASSO);

delta_TrainCPU_PTCvsNCC=mean(result.TrainCPU_PTC)-mean(result.TrainCPU_NCC);
delta_TrainCPU_PTCvsSNCC=mean(result.TrainCPU_PTC)-mean(result.TrainCPU_SNCC);
delta_TrainCPU_PTCvsLASSO=mean(result.TrainCPU_PTC)-mean(result.TrainCPU_LASSO);



win_overall_TPC_NCC=sum(result.D_PTC<=result.D_NCC&result.F1_PTC>=result.F1_NCC&result.TrainCPU_PTC<=result.TrainCPU_NCC)
win_overall_TPC_SNCC=sum(result.D_PTC<=result.D_SNCC&result.F1_PTC>=result.F1_SNCC&result.TrainCPU_PTC<=result.TrainCPU_SNCC)
win_overall_TPC_LASSO=sum(result.D_PTC<=result.D_LASSO&result.F1_PTC>=result.F1_LASSO&result.TrainCPU_PTC<=result.TrainCPU_LASSO)
loose_overall_TPC_NCC=sum(result.D_PTC>=result.D_NCC&result.F1_PTC>=result.F1_NCC&result.TrainCPU_PTC>=result.TrainCPU_NCC)
loose_overall_TPC_SNCC=sum(result.D_PTC>=result.D_SNCC&result.F1_PTC>=result.F1_SNCC&result.TrainCPU_PTC>=result.TrainCPU_SNCC)
loose_overall_TPC_LASSO=sum(result.D_PTC>=result.D_LASSO&result.F1_PTC>=result.F1_LASSO&result.TrainCPU_PTC>=result.TrainCPU_LASSO)

result.D_NCC(cnt+3)=100*win_D_PTCvsNCC/cnt;
result.D_SNCC(cnt+3)=100*win_D_PTCvsSNCC/cnt;
result.D_LASSO(cnt+3)=100*win_D_PTCvsLASSO/cnt;
result.D_NCC(cnt+4)=100*loose_D_PTCvsNCC/cnt;
result.D_SNCC(cnt+4)=100*loose_D_PTCvsSNCC/cnt;
result.D_LASSO(cnt+4)=100*loose_D_PTCvsLASSO/cnt;
result.D_NCC(cnt+5)=delta_D_PTCvsNCC;
result.D_SNCC(cnt+5)=delta_D_PTCvsSNCC;
result.D_LASSO(cnt+5)=delta_D_PTCvsLASSO;

result.Recall_NCC(cnt+3)=100*win_Recall_PTCvsNCC/cnt;
result.Recall_SNCC(cnt+3)=100*win_Recall_PTCvsSNCC/cnt;
result.Recall_LASSO(cnt+3)=100*win_Recall_PTCvsLASSO/cnt;
result.Recall_NCC(cnt+4)=100*loose_Recall_PTCvsNCC/cnt;
result.Recall_SNCC(cnt+4)=100*loose_Recall_PTCvsSNCC/cnt;
result.Recall_LASSO(cnt+4)=100*loose_Recall_PTCvsLASSO/cnt;
result.Recall_NCC(cnt+5)=delta_Recall_PTCvsNCC;
result.Recall_SNCC(cnt+5)=delta_Recall_PTCvsSNCC;
result.Recall_LASSO(cnt+5)=delta_Recall_PTCvsLASSO;

result.bAcc_NCC(cnt+3)=100*win_bAcc_PTCvsNCC/cnt;
result.bAcc_SNCC(cnt+3)=100*win_bAcc_PTCvsSNCC/cnt;
result.bAcc_LASSO(cnt+3)=100*win_bAcc_PTCvsLASSO/cnt;
result.bAcc_NCC(cnt+4)=100*loose_bAcc_PTCvsNCC/cnt;
result.bAcc_SNCC(cnt+4)=100*loose_bAcc_PTCvsSNCC/cnt;
result.bAcc_LASSO(cnt+4)=100*loose_bAcc_PTCvsLASSO/cnt;
result.bAcc_NCC(cnt+5)=delta_bAcc_PTCvsNCC;
result.bAcc_SNCC(cnt+5)=delta_bAcc_PTCvsSNCC;
result.bAcc_LASSO(cnt+5)=delta_bAcc_PTCvsLASSO;

result.F1_NCC(cnt+3)=100*win_F1_PTCvsNCC/cnt;
result.F1_SNCC(cnt+3)=100*win_F1_PTCvsSNCC/cnt;
result.F1_LASSO(cnt+3)=100*win_F1_PTCvsLASSO/cnt;
result.F1_NCC(cnt+4)=100*loose_F1_PTCvsNCC/cnt;
result.F1_SNCC(cnt+4)=100*loose_F1_PTCvsSNCC/cnt;
result.F1_LASSO(cnt+4)=100*loose_F1_PTCvsLASSO/cnt;
result.F1_NCC(cnt+5)=delta_F1_PTCvsNCC;
result.F1_SNCC(cnt+5)=delta_F1_PTCvsSNCC;
result.F1_LASSO(cnt+5)=delta_F1_PTCvsLASSO;

result.TrainCPU_NCC(cnt+3)=100*win_TrainCPU_PTCvsNCC/cnt;
result.TrainCPU_SNCC(cnt+3)=100*win_TrainCPU_PTCvsSNCC/cnt;
result.TrainCPU_LASSO(cnt+3)=100*win_TrainCPU_PTCvsLASSO/cnt;
result.TrainCPU_NCC(cnt+4)=100*loose_TrainCPU_PTCvsNCC/cnt;
result.TrainCPU_SNCC(cnt+4)=100*loose_TrainCPU_PTCvsSNCC/cnt;
result.TrainCPU_LASSO(cnt+4)=100*loose_TrainCPU_PTCvsLASSO/cnt;
result.TrainCPU_NCC(cnt+5)=delta_TrainCPU_PTCvsNCC;
result.TrainCPU_SNCC(cnt+5)=delta_TrainCPU_PTCvsSNCC;
result.TrainCPU_LASSO(cnt+5)=delta_TrainCPU_PTCvsLASSO;

result(cnt+1,5:24)=array2table(mean(row));
result(cnt+2,5:24)=array2table(mean(ranks));
result(cnt+1,1)={'*Average*'};
result(cnt+2,1)={'*Rank*'};
result(cnt+3,1)={'*PTC winning ratio(%)*'};
result(cnt+4,1)={'*PTC loosing ratio (%)*'};
result(cnt+5,1)={'*Delta (PTC vs. ...)*'};
result(cnt+6,1)={'*P-value (T-Test)*'};

[~, D_pv_NCC, ~, ~] = ttest2(result.D_PTC ,result.D_NCC);
[~, D_pv_SNCC, ~, ~] = ttest2(result.D_PTC ,result.D_SNCC);
[~, D_pv_LASSO, ~, ~] = ttest2(result.D_PTC ,result.D_LASSO);
[~, Recall_pv_NCC, ~, ~] = ttest2(result.Recall_PTC ,result.Recall_NCC);
[~, Recall_pv_SNCC, ~, ~] = ttest2(result.Recall_PTC ,result.Recall_SNCC);
[~, Recall_pv_LASSO, ~, ~] = ttest2(result.Recall_PTC ,result.Recall_LASSO);
[~, bAcc_pv_NCC, ~, ~] = ttest2(result.bAcc_PTC ,result.bAcc_NCC);
[~, bAcc_pv_SNCC, ~, ~] = ttest2(result.bAcc_PTC ,result.bAcc_SNCC);
[~, bAcc_pv_LASSO, ~, ~] = ttest2(result.bAcc_PTC ,result.bAcc_LASSO);
[~, F1_pv_NCC, ~, ~] = ttest2(result.F1_PTC ,result.F1_NCC);
[~, F1_pv_SNCC, ~, ~] = ttest2(result.F1_PTC ,result.F1_SNCC);
[~, F1_pv_LASSO, ~, ~] = ttest2(result.F1_PTC ,result.F1_LASSO);
[~, TrainCPU_pv_NCC, ~, ~] = ttest2(result.TrainCPU_PTC ,result.TrainCPU_NCC);
[~, TrainCPU_pv_SNCC, ~, ~] = ttest2(result.TrainCPU_PTC ,result.TrainCPU_SNCC);
[~, TrainCPU_pv_LASSO, ~, ~] = ttest2(result.TrainCPU_PTC ,result.TrainCPU_LASSO);

result(cnt+6,5:24)=array2table([0 D_pv_NCC D_pv_SNCC D_pv_LASSO 0 Recall_pv_NCC Recall_pv_SNCC Recall_pv_LASSO 0 bAcc_pv_NCC bAcc_pv_SNCC bAcc_pv_LASSO 0 F1_pv_NCC F1_pv_SNCC F1_pv_LASSO 0 TrainCPU_pv_NCC TrainCPU_pv_SNCC TrainCPU_pv_LASSO]);


writetable(result,'result.xlsx') % Fig. 3 in the paper



%% Appendix - Fig. 20 

clear
load datasets

clear col
for n=1:height(datasets)
    clearex('n','datasets','col');
    load(['results/',num2str(n)]);

    col{n,1}=PTC_Mdl.PrototypeSampleIDs;
    col{n,2}=PTC_Mdl.PrototypeFeatureIDs;
    col{n,3} = 100*mean(y_pred_PTC == y_test);
    col{n,4} = 100*mean(y_pred_LASSO == y_test);
    col{n,5}=PTC_numFeatures;
    col{n,6}=LASSO_numFeatures;
    col{n,7}=NCC_numFeatures;

end
PTC_Models = cell2table(col,'VariableNames',{'PrototypeSampleIDs','CoreFeatureIDs','TestAccuracy','LASSOTestAccuracy','NumFeaturesPTC','NumFeaturesLASSO','NumFeatursTotal'});
writetable(PTC_Models,'PTC_Models.xlsx')


%% Fig. 4 in the paper (MNIST case study)

clear
load MINIST01

% PTC
PTCMdl=fitPTC(X_train,y_train);
y_test_pred=Mdl.My(knnsearch(PTCMdl.MX,X_test(:,Mdl.PrototypeFeatureIDs),'K',1));
AccTestPTC=mean(y_test_pred==y_test);
PTCnumFeatures=numel(PTCMdl.PrototypeFeatureIDs);

rng(42)
lambdas = logspace(-4, 0, 100);
[BAll, FitInfoAll] = lassoglm(X_train, y_train, 'binomial', 'Lambda', lambdas,'Standardize',true);

for i = 1:length(FitInfoAll.Lambda)
    numNonZero = sum(BAll(:, i) ~= 0);

    if numNonZero >= 1 && numNonZero <= PTCnumFeatures
        fprintf('Lambda: %f, Non-zero features: %d\n', FitInfoAll.Lambda(i), numNonZero);
        selectedIntercept=FitInfoAll.Intercept(i)
        selectedB = BAll(:, i); 
        selectedLambda = FitInfoAll.Lambda(i); 
        break; 
    end
end


lambda = FitInfoAll.Lambda;
deviance = FitInfoAll.Deviance;

n_samples = size(X_train, 1);
p = size(X_train, 2);
BIC = zeros(size(lambda));
for i = 1:length(lambda)
    numParams(i) = sum(BAll(:, i) ~= 0);
    BIC(i) = n_samples * log(deviance(i) / n_samples) + numParams(i) * log(n_samples);
end
[minBIC, idxMinBIC] = min(BIC);

%BIC-otimized lambda
best_lambda = lambda(idxMinBIC);
B=BAll(:,idxMinBIC);
Intercept=FitInfoAll.Intercept(idxMinBIC);
CardinalityRatio=numParams(idxMinBIC)/p;

y_pred_test_prob = glmval([Intercept; B], X_test, 'logit');
y_pred_test = y_pred_test_prob > 0.5;
AccTest_LASSO = mean(y_pred_test == y_test);

LASSO3_Features=find(selectedB~=0)
weights=selectedB(LASSO3_Features);
selectedIntercept

% Infer virtual prototypes from the LASSO Model 
class_1 = X_train(y_train == 1, :);  
class_0 = X_train(y_train == 0, :);  
centroid_1 = mean(class_1, 1); 
centroid_0 = mean(class_0, 1);  
centroid_1 = centroid_1 .* weights;
centroid_0 = centroid_0 .* weights;
centroid_1=centroid_1(LASSO3_Features);
centroid_0=centroid_0(LASSO3_Features);
centroids_LASSO3=vertcat(centroid_0',centroid_1');

y_pred_test_prob_3 = glmval([selectedIntercept; selectedB], X_test, 'logit');
y_pred_test_3 = y_pred_test_prob_3 > 0.5;
AccTest_LASSO_3 = mean(y_pred_test_3 == y_test);

% NCC
NCC_Mdl=fitNCC(X_train,y_train);
y_pred_test_NCC = predNCC(NCC_Mdl, X_test);
AccTest_NCC=mean(y_test==y_pred_test_NCC);
NCCCentroids=NCC_Mdl.centroids

%SNCC
for R=1:size(X_train,2)
    [theta_plus, theta_minus] = fitSNCC(X_train', y_train, R);
    center_sparse=abs(theta_plus-theta_minus)';
    num_features(R)=sum(center_sparse~=0);
end
desired_Rs=find(num_features<=PTCnumFeatures & num_features>=1);
desired_R=desired_Rs(1);
[theta_plus, theta_minus] = fitSNCC(X_train', y_train, desired_R);
center_sparse=abs(theta_plus-theta_minus)';
CenterSparseFeatures=find(center_sparse~=0);
y_pred_test_SNCC = predSNCC(X_test', theta_plus, theta_minus,2);
AccTest_SNCC = mean(y_test == y_pred_test_SNCC);
