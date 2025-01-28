% For Peer-review
% Demo of brute-force approach with iris dataset
% (Fig. 5,6,7 in the appendix)
%
% Submitted to ICML 2025



clear
load fisheriris
y=grp2idx(species)-1;
X=meas;
ids=find(y==0|y==1);
y=y(ids);
X=X(ids,:);
[N,M]=size(X);
rng(42);
indices = randperm(N);
numTestSamples = round(0.2 * N);
trainIdx = indices(numTestSamples+1:end);
testIdx = indices(1:numTestSamples);
X_train = X(trainIdx, :);
X_test = X(testIdx, :);
y_train = y(trainIdx, :);
y_test = y(testIdx, :);

Mdl=NaivePrototype(X_train,y_train);
y_test_NL=Mdl.My(knnsearch(Mdl.MX,X_test(:,Mdl.PrototypeFeatureIDs),'K',1));
acc_test=sum(y_test_NL==y_test)/numel(y_test);
Model_1NN= fitcknn(X_train, y_train, 'NumNeighbors', 1);
y_test_1NN = predict(Model_1NN, X_test);
acc_test_1NN=sum(y_test_1NN==y_test)/numel(y_test);

X_train_w_id=horzcat([1:size(X_train,1)]',X_train);