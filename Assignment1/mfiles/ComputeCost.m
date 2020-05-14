% Compute cost
function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
reg_term = lambda*sum(sum(W.*W));
error_term = sum(-log(sum(Y.*P, 1)))/size(Y, 2);
J = reg_term + error_term;
end