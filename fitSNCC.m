% For Peer-review
% Implemenation of "Sparse Nearest Centroid Classifier (SNCC) - Train"
% Algorithm 1 in: 
% Giuseppe C. Calafiore, Giulia Fracastoro, Sparse ℓ1- and ℓ2-Center Classifiers
% IEEE Transactions on Neural Networks and Learning Systems
% Volume: 33, Issue: 3, 2022


function [theta_plus, theta_minus] = fitSNCC(X, y, k)

indices_plus = find(y == 1);
indices_minus = find(y == 0);

n_plus = length(indices_plus);
n_minus = length(indices_minus);

x_plus = mean(X(:, indices_plus), 2); 
x_minus = mean(X(:, indices_minus), 2); 

x_tilde = (x_plus + x_minus) / 2;
delta = x_plus - x_minus;

[~, sorted_indices] = sort(abs(delta), 'descend');
D = sorted_indices(1:k); 

E = setdiff(1:length(delta), D);

x_tilde_D = zeros(size(x_tilde));
x_tilde_D(D) = x_tilde(D); 

x_tilde_E = zeros(size(x_tilde));
x_tilde_E(E) = x_tilde(E); 

theta_plus = x_tilde_D + x_tilde_E;
theta_minus = x_tilde_D - x_tilde_E;

end
