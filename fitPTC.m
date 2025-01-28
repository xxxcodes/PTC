% For Peer-review
%
% ****** Proposed Method ****
%
% Implemenation of "Prototype Theory Classifier (PTC)"


function [Mdl]=fitPTC(X,y)
[n,p]=size(X);
M=1:p;
C_best=M; %new
ids0=find(y==0);
ids1=find(y==1);
L=0;
op=0;
mm=0;
e_best=inf;
while(true)
    L=L+1;
    Mdl0=LSH(X(ids0,M));
    Mdl1=LSH(X(ids1,M));
    for i=1:n
       s=LSH(X(ids0,M),Mdl0,X(i,M));
       if s~=-1
            s=ids0(s);
            o=LSH(X(ids1,M),Mdl1,X(i,M));
            if o~=-1
                o=ids1(o);
                if y(i)==1 tmp_o=o; o=s; s=tmp_o; end
                vs=abs(X(s,M)-X(i,M));
                vo=abs(X(o,M)-X(i,M));
                v=vo-vs;
                C=M(find(v>0));
                if numel(C)>1
                    yt=y([s,o]);
                    op=op+1;
                    yhat=yt(knnsearch(X([s;o],C),X(:,C),'K',1));
                    e = sum(yhat ~= y);
                    if e<e_best
                        mm=mm+1;
						C_best=C; s_best=s; o_best=o;  e_best=e; L_best=L; best_pivot=i; last_e_best=e_best;
                    end
                end
            end
       end
    end
    if numel(C_best)==numel(M)
        break;
    else
        M=C_best;
        e_best=inf;
    end
end

if op==0 %new
    Mdl=[];
    return
end
disp(sprintf('Best Prototype=[%i(class %i),%i(class %i)], BestError=%.4f, Features=[%s]',s_best,y(s_best),o_best,y(o_best),last_e_best/n,num2str(M,'%i ')));
Mdl.PrototypeSampleIDs=[s_best o_best];
Mdl.PrototypeFeatureIDs=M;
Mdl.BestPivotSampleID=best_pivot;
Mdl.Error=last_e_best/n;
Mdl.Acc=1-Mdl.Error;
Mdl.Layers=L_best;
Mdl.MX=X([s_best o_best],M);
Mdl.My=y([s_best o_best]);
Mdl.OperationSave=op/(L*n);
if y(Mdl.PrototypeSampleIDs(1))~=0
    Mdl.My=y(flip(Mdl.PrototypeSampleIDs));
    Mdl.PrototypeSampleIDs=flip(Mdl.PrototypeSampleIDs);
    tmp=Mdl.MX(1,:);
    Mdl.MX(1,:)=Mdl.MX(2,:);
    Mdl.MX(2,:)=tmp;
end

end

% Implementation of LSH
function  Mdl=LSH(X,Mdl,query_point)

if nargin==1
    [n,m]=size(X);
    Mdl.num_hash_functions = 10;
    Mdl.hash_size = 100;
	rng(42);
    Mdl.hash_functions = randn(m, Mdl.num_hash_functions);
    hash_codes = sign(X * Mdl.hash_functions);
    Mdl.hash_tables = cell(Mdl.hash_size, 1);
    for i = 1:n
        hash_code = hash_codes(i, :);
        hash_index = mod(sum(hash_code), Mdl.hash_size) + 1;
        if isempty(Mdl.hash_tables{hash_index})
            Mdl.hash_tables{hash_index} = i;
        else
            Mdl.hash_tables{hash_index} = [Mdl.hash_tables{hash_index}, i];
        end
    end

else

    query_hash_code = sign(query_point * Mdl.hash_functions);
    query_hash_index = mod(sum(query_hash_code), Mdl.hash_size) + 1;
    candidate_neighbors = Mdl.hash_tables{query_hash_index};
    
    best_distance = inf;
    best_neighbor = -1;
    for i = 1:length(candidate_neighbors)
        candidate_point = X(candidate_neighbors(i), :);
        distance = norm(candidate_point - query_point);
        if distance < best_distance & distance~=0
            best_distance = distance;
            best_neighbor = candidate_neighbors(i);
        end
    end    
    
    Mdl=best_neighbor;

end

end
