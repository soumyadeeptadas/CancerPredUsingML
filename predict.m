function [p F1] = predict(theta, X, y)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta. Also return the F1 score computed from
%the predictions

m = size(X, 1); 

p = zeros(m, 1);



p = round(sigmoid(X * theta));

predictions = (p >= 0.5);
tp = sum(predictions == 1 & y == 1);
fp = sum(predictions == 1 & y == 0);
fn = sum(predictions == 0 & y == 1);

prec = tp / (tp + fp);
rec = tp / (tp + fn);

F1 = (2 * prec * rec) / (prec + rec);


end