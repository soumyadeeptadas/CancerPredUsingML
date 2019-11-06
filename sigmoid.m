function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.



g = zeros(size(z));


denominator =  1 + exp( -z);
g = 1 ./ denominator;


end