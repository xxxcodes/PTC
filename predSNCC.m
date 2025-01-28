% For Peer-review
% Implemenation of "Sparse Nearest Centroid Classifier (SNCC) - Test"
% Algorithm 1 in: 
% Giuseppe C. Calafiore, Giulia Fracastoro, Sparse ℓ1- and ℓ2-Center Classifiers
% IEEE Transactions on Neural Networks and Learning Systems
% Volume: 33, Issue: 3, 2022


function predictions = predSNCC(X_test, theta_plus, theta_minus,L)

dist_to_plus = vecnorm(X_test - theta_plus, L, 1);
dist_to_minus = vecnorm(X_test - theta_minus, L, 1);

predictions = ones(size(X_test, 2),1);
predictions(dist_to_plus > dist_to_minus) = 0;

end