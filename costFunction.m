function [J, grad] = costFunction(theta, X, y)

%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.



m = length(y);

J = 0;
grad = zeros(size(theta));

term1 = -1 .* (y .* log(sigmoid(X * theta)));
term2 = 1 .* ((1-y) .* log((1-sigmoid(X * theta))));

J = sum(term1 - term2) / m;

grad = (X' * (sigmoid(X * theta) - y)) * (1/m);


end