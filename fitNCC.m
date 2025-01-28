% For Peer-review
% Implemenation of "Nearest Centroid Classifier (NCC) - Train"

function model = fitNCC(X, Y)
 
    classLabels = unique(Y);
    nClasses = length(classLabels);
    [nSamples, nFeatures] = size(X);
    
    centroids = zeros(nClasses, nFeatures);
    
    for i = 1:nClasses
        classData = X(Y == classLabels(i), :);  
        centroids(i, :) = mean(classData, 1);   
    end
    
    model.centroids = centroids;
    model.classLabels = classLabels;
end
