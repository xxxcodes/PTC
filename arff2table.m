% For Peer-review
% Covert ARFF file to mat


function [dataTable,varcat]=arff2table(arff_filelocation)

content = fileread(arff_filelocation);
content = regexprep(content, '\t', ' ');
tmp = extractBetween(lower(content),'@relation','@data');
tmpa = strsplit(lower(tmp{1}), '@attribute');
tmpa(1)=[];
varnames=[];
for i=1:numel(tmpa)
    tmpa{i}=strtrim(tmpa{i});
    tmpab{i} = strsplit(tmpa{i}, ' ');
    if strfind(tmpa{i}, '{')
        varcat(i)=1;
    else
        varcat(i)=0;
    end
    for j=1:2
        tmpab{i}{j}=strtrim(tmpab{i}{j});
    end
end
varnames=[];
for j=1:numel(tmpab)
    varnames=[varnames,strrep(tmpab{j}{1},"'","")];
    varnames=strrep(varnames,':','');
    if j<numel(tmpab)
        varnames=[varnames,','];
    end
end
tmp = strsplit(lower(content),'@data');
tmp=tmp{2};
if strfind(tmp, '?')
    dataTable=[];
    varcat=[];
    return;
end
if strfind(tmp, '{')
    dataTable=[];
    varcat=[];
    return;
end

csv_filelocation = strrep(arff_filelocation, '.arff', '.csv');

fileID = fopen(csv_filelocation, 'w');
fprintf(fileID, '%s%s', varnames,tmp);
fclose(fileID);

dataTable=readtable (csv_filelocation);
delete(csv_filelocation);
