function score = plda_eval(mdl, X, Y)
if ~isequal(mdl.mu, 0)
    X = bsxfun(@minus, X, mdl.mu);
    Y = bsxfun(@minus, Y, mdl.mu);
end
X_norm = sum(X .* (mdl.Q * X), 1);
Y_norm = sum(Y .* (mdl.Q * Y), 1);
score = bsxfun(@plus, bsxfun(@plus, 2*X'*mdl.P*Y, X_norm'), Y_norm);