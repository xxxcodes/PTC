% For Peer-review
% Implemenation of "Nearest Centroid Classifier (NCC) - Test"

function predictedLabels = predNCC(model, X_test)

    nTestSamples = size(X_test, 1);
    nClasses = size(model.centroids, 1);
    distances = zeros(nTestSamples, nClasses);
    
    for i = 1:nClasses
        distances(:, i) = sum((X_test - model.centroids(i, :)).^2, 2);  % Euclidean distance
    end
    
    [~, minIdx] = min(distances, [], 2);  
    predictedLabels = model.classLabels(minIdx);
end
