function plotData(X, y)
 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.


figure; hold on;

% plotting the positive and negative examples on a 2D plot, 
% using the option 'k+' for the positive and 'ko' for the negative examples.

pos = find(y==1);
neg = find(y==0);

plot(X(pos,1), X(pos,2), 'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1), X(neg,2), 'ko','MarkerFaceColor','y','MarkerSize',7);



hold off;

end