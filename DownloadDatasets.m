%%  Microarray datasets Benchmark by Kıvanç Güçkıran
% https://github.com/kivancguckiran/microarray-data/
clear


if exist('datasets', 'dir') ~= 7
    mkdir('datasets');
end

datasets={'alon','borovecki','chin','chowdary','golub','gordon','gravier','pomeroy','shipp','singh','subramanian','tian','west'};
datasets_DOIs={...
    'https://doi.org/10.1073/pnas.96.12.6745',...
    'https://doi.org/10.1073/pnas.0504921102',...
    'https://doi.org/10.1016/j.ccr.2006.10.009',...
    'https://doi.org/10.2353%2Fjmoldx.2006.050056',...
    'https://doi.org/10.1126/science.286.5439.531',...
    'https://aacrjournals.org/cancerres/article/62/17/4963/509160/Translation-of-Microarray-Data-into-Clinically',...
    'http://onlinelibrary.wiley.com/doi/10.1002/gcc.20820/abstract',...
    'https://doi.org/10.1038/415436a',...
    'https://doi.org/10.1038/nm0102-68',...
    'https://doi.org/10.1016/S1535-6108(02)00030-2',...
    'https://doi.org/10.1073%2Fpnas.0506580102',...
    'https://www.doi.org/10.1056/NEJMoa030847',...
    'https://doi.org/10.1073/pnas.201162998',...
    }
dataset_url='https://github.com/kivancguckiran/microarray-data/';
for i=1:numel(datasets)
    disp(['Downloading ',datasets{i}]);
    websave(['datasets/',datasets{i},'.tar.gz'],['https://github.com/kivancguckiran/microarray-data/raw/refs/heads/master/csv/',datasets{i},'.tar.gz']);
end
for i=1:numel(datasets)
    disp(['Unzipping datasets/',datasets{i},'.tar.gz ...']);
    untar(['datasets/',datasets{i},'.tar.gz'],['datasets/']);
end

for i=1:numel(datasets)
    dataset_DOI=datasets_DOIs{i};
    disp(['Import CSV... ',datasets{i}]);
    X=readtable(['datasets/',datasets{i},'_inputs.csv']);
    y=readmatrix(['datasets/',datasets{i},'_outputs.csv']);
    y=y-1;
    y_meta={'negative','positive'};
    save(['datasets/',datasets{i},'.mat'],'X','y','y_meta','dataset_url','dataset_DOI');
end

for i=1:numel(datasets)
    disp(['Delete Non-mat files... ',datasets{i}]);
    delete(['datasets/',datasets{i},'.tar.gz']);
    delete(['datasets/',datasets{i},'_inputs.csv']);
    delete(['datasets/',datasets{i},'_outputs.csv']);
end

%% OpenML binary datasets with n<<p condition
clear
dataset_urls={...
'https://www.openml.org/data/download/54011/OVA_Breast.arff',...
'https://www.openml.org/data/download/54044/OVA_Colon.arff',...
'https://www.openml.org/data/download/54017/OVA_Kidney.arff',...
'https://www.openml.org/data/download/54013/OVA_Lung.arff',...
'https://www.openml.org/data/download/54022/OVA_Omentum.arff',...
'https://api.openml.org/data/download/22112159/dataset',...
'https://www.openml.org/data/download/22045089/dataset',...
'https://api.openml.org/data/download/22112161/dataset',...
'https://www.openml.org/data/download/1390177/phpCLGrjq',...
'https://www.openml.org/data/download/1593757/phpEZ030X',...
'https://www.openml.org/data/download/1593758/phpdo58hj',...
'https://www.openml.org/data/download/21379038/file22f047870b9.arff'}

dataset_names={...
'OVA_Breast',...
'OVA_Colon',...
'OVA_Kidney',...
'OVA_Lung',...
'OVA_Omentum',...
'Ovarian',...
'PCam',...
'SMK',...
'anthracyclineTaxaneChemotherapy',...
'dbworld-bodies-stemmed',...
'dbworld-bodies',...
'isolet'};

% Downloading
for i = 1:length(dataset_names)
    filename = [dataset_names{i},'.arff'];
    filepath = fullfile('datasets/', filename);
    try
        websave(filepath, dataset_urls{i});
        disp(['File downloaded and saved as: ', filepath]);
    catch
        warning(['Failed to download file from URL: ', url]);
    end
end

% Processing
for i = 1:length(dataset_names)
    filename = [dataset_names{i},'.arff'];
    filepath = fullfile('datasets/', filename);
    disp(['Processing ', filepath, '...']);
    dataset=dataset_names{i};

    [dataTable,varcat]=arff2table(filepath);
    if isempty(dataTable)
        disp('empty');
        continue;
    end

    ClassCol=dataTable.Properties.VariableNames{end};
    ColNames=lower(dataTable.Properties.VariableNames);
    columnName = {'class', 'target','label','y','pclass','overall_diagnosis','result','click'};
    matchingColumns = find(ismember(ColNames{1}, columnName));
    if ~isempty(matchingColumns)
        ClassCol=dataTable.Properties.VariableNames{1};
       varcat(matchingColumns(1))=[];
    end
    uq=unique(table2array(dataTable(:,ClassCol)));
    for k=1:2
        if ~iscell(uq)
            y_meta{k}=[ClassCol,' ',num2str(uq(k))];
        else
            y_meta{k}=[ClassCol,' ',uq{k}];
        end
    end
    if iscell(uq)
        y=strcmp(table2array(dataTable(:,ClassCol)),uq{2});
    else
        y=table2array(dataTable(:,ClassCol))==uq(2);
    end

    clsscol=grp2idx(table2array(dataTable(:,ClassCol)));
    dataTable(:, ClassCol) = [];
    dataTable.Class=y;
    varcat(end+1)=1;

    columnName = {'ID', 'id','date','id_ref'};
    matchingColumns = find(ismember(lower(dataTable.Properties.VariableNames), lower(columnName)));
    if ~isempty(matchingColumns)
        dataTable(:,matchingColumns)=[];
        varcat(matchingColumns)=[];
    end
    clear MD;
    for j=1:width(dataTable)-1
        if varcat(j)==1
            MD(j)=numel(unique(dataTable(:,1)));
            %eval(['dataTable.',VariableNames{j},'=categorical(dataTable.',VariableNames{i},')']);
        else
            MD(j)=1;
        end
    end
    sumMD=sum(MD);
    %%disp(['i=',num2str(i), ' dataset=',dataset,' DO=',num2str(width(dataTable)-1),' D-OHE=',num2str(sumMD)]);
    if sumMD-width(dataTable)>20000
        continue;
    end
    if ~isempty(find(varcat(1:end-1)==1))
    if sumMD-width(dataTable)>0
        newTable=dataTable;
        for j=1:width(dataTable)-1
            if varcat(j)==1
                tmp=dummyvar(categorical(table2array(dataTable(:,j))));
                perfix=dataTable.Properties.VariableNames{j};
                newTable(:,perfix)=[];
                for k=1:size(tmp,2)
                    ColName=[perfix, '_',num2str(k)];
                    eval(['newTable.',ColName,'=tmp(:,k);']);
                end
            end
        end
        newTable = movevars(newTable, 'Class', 'After', newTable.Properties.VariableNames{end});
        dataTable=newTable;
        %disp(['-->',' Check D-OHE=',num2str(width(dataTable))]);
    end    

    end
    if sumMD-width(dataTable)<=20000
        k=k+1;
        X=dataTable(:,1:end-1);
        y=table2array(dataTable(:,end));
        dataset_url=dataset_urls{i};
        dataset_DOI=dataset_urls{i};
        save(sprintf('datasets/%s.mat',dataset),'X','y','y_meta','dataset_url','dataset_DOI');
        datasets{k}=dataset;
    end
end

delete('datasets/*.arff');

%% UCI Lung Cancer Datasets (binary and n<<p)
clear
zipfilename='datasets/lungcancer.zip';
websave(zipfilename, 'https://archive.ics.uci.edu/static/public/62/lung+cancer.zip');
unzip(zipfilename,'datasets/');
filename='datasets/lung-cancer.data';
data = readtable(filename, 'FileType', 'text');
X = data(:, 2:end);
y = data.Var1 == 1;  % Assuming class 1 as positive class
y_meta={'lung cancer type 1','lung cancer types 2 or 3'};
dataset_url='https://archive.ics.uci.edu/dataset/62/lung+cancer';
dataset_DOI='https://doi.org/10.1016/0031-3203(91)90074-F';
save('datasets/lung-cancer-UCI.mat','X','y','y_meta','dataset_url','dataset_DOI');
delete('datasets/lungcancer.zip');
delete('datasets/lung-cancer.data');
delete('datasets/lung-cancer.names');
delete('datasets/index');




%% Split to Train/Test if not available
load datasets
for n=25:height(datasets)
    clearex('n','datasets');
   
disp([num2str(n),':',datasets.Name{n}])
load (['datasets/',datasets.Name{n}]);
 if exist('dataset_url')
        dataset_URL=dataset_url;
 end
if istable(X)
    ftlbls=X.Properties.VariableNames;
    X=table2array(X);
end

automatic_split=0;
if exist('X_train')==0
    automatic_split=1;
    rng(42); cv = cvpartition(size(X,1), 'HoldOut', 0.2); X_train = X(training(cv), :); X_test = X(test(cv), :); y_train = y(training(cv), :); y_test = y(test(cv), :);
else
    if istable(X_train)
        X_train=table2array(X_train);
        X_test=table2array(X_test);
    end
end

missing_mask = isnan(X_train);
missing_train=0;
if ~isempty(find(missing_mask))
    missing_train=1;
    for i = 1:size(X, 2)  % Iterate over columns
        column_mean = mean(X(~missing_mask(:,i), i), 'omitnan');  % Mean of non-NaN values
        X_train(missing_mask(:,i), i) = column_mean;  % Replace NaNs with column mean
    end
end
missing_mask = isnan(X_test);
missing_test=0;
if ~isempty(find(missing_mask))
    missing_test=1;
    for i = 1:size(X, 2)  % Iterate over columns
        column_mean = mean(X(~missing_mask(:,i), i), 'omitnan');  % Mean of non-NaN values
        X_test(missing_mask(:,i), i) = column_mean;  % Replace NaNs with column mean
    end
end

save(['datasets/',datasets.Name{n}],'X','y','X_train','y_train','X_test','y_test','y_meta','dataset_URL','dataset_DOI','ftlbls','automatic_split','missing_train','missing_test');

end

%% dataset stats

clear
files = dir(fullfile(['datasets/*.mat']));
[~, sortedIdx] = sort([files.datenum],'asc');
files = files(sortedIdx);
for k=1:numel(files)
        dataset=files(k).name;
        dataset=strrep(dataset,'.mat','');
        datasets{k,1}=dataset;
        load(['datasets/',dataset,'.mat']);
        datasets{k,2}=size(X,1);
        datasets{k,3}=size(X,2);
        class0=numel(find(y==0));
        class1=numel(find(y==1));
        datasets{k,4}=max(class0,class1)/datasets{k,2};
        datasets{k,5}=automatic_split;
        datasets{k,6}=missing_train;
        datasets{k,7}=missing_test;
        datasets{k,8}=y_meta;
        datasets{k,9}=dataset_DOI;
        datasets{k,10}=dataset_URL;
end

datasets=cell2table(datasets);
datasets.Properties.VariableNames={'Name','n','p','MjClasss','AutoSplit','MissingTrain','MissingTest','Labels','DOI','URL'};
%datasets = [datasets(2:end, :); datasets(1, :)];
%datasets = [datasets(2:end, :); datasets(1, :)];
%datasets = [datasets([1:23,25:end], :); datasets(24, :)];

save('datasets.mat','datasets');