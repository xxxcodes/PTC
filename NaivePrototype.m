% For Peer-review
% Brute-force algorithm for sparse prototype learning
% Note: Not robust to curse of dimensionality and noisy features. Not scalable: O(N^3 2^p)


function Mdl=NaivePrototype(X_train,y_train)

    set = 1:size(X_train,2);
    subsets = cell(1, length(set));
    for i = 1:length(set)
        subsets{i} = nchoosek(set, i);
    end
    idn=find(y_train==0);
    idp=find(y_train==1);
    k=0;
    for i=1:numel(subsets)
        for j=1:size(subsets{i},1)
            sfids=subsets{i}(j,:);
            for s=1:size(X_train,1)
                curr_y=y_train(s);
                if curr_y==0
                    nn_neg=knnsearch(X_train(idn,sfids),X_train(s,sfids),'K',2);
                    nn_neg=nn_neg(end);
                    nn_neg=idn(nn_neg);
                    nn_pos=knnsearch(X_train(idp,sfids),X_train(s,sfids),'K',1);
                    nn_pos=idp(nn_pos);
                else
                    nn_neg=knnsearch(X_train(idn,sfids),X_train(s,sfids),'K',1);
                    nn_neg=idn(nn_neg);
                    nn_pos=knnsearch(X_train(idp,sfids),X_train(s,sfids),'K',2);
                    nn_pos=nn_pos(end);
                    nn_pos=idp(nn_pos);
                end
                yt=y_train([nn_neg,nn_pos]);
                yhat=yt(knnsearch(X_train([nn_neg,nn_pos],sfids),X_train(:,sfids),'K',1));
                k=k+1;
                err(k) = sum(yhat~=y_train)/numel(y_train);
                svs(k,1)=nn_neg;
                svs(k,2)=nn_pos;
                nn_features{k}=sfids;
            end
        end
    end
    [minerr,bestk]=min(err);
    Mdl.PrototypeSampleIDs=svs(bestk,1:2);
    Mdl.PrototypeFeatureIDs=nn_features{bestk};
    Mdl.Error=minerr;
    Mdl.Subsets=subsets;
    Mdl.L=0;
    Mdl.MX=X_train(Mdl.PrototypeSampleIDs,Mdl.PrototypeFeatureIDs);
    Mdl.My=y_train(svs(bestk,1:2));
