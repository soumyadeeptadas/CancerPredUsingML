%% Implementing Logistic Regression, on Wisconsin Breast Cancer Dataset


clear ; close all; clc
fprintf('\n \n=====================Cancer Prediction System==========================\n\n')
%% Load Data
%  The first 9 columns contains the extracted features and the 10th column
%  contains the label. 
%Label 1 is malignant and 0 is benign.

%%Loading Cross Validation Data

data = load('data_cross_validation.csv');
Xcv = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); ycv = data(:, 10);

%%  Loading test data 

data = load('data_test.csv');
Xtest = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); ytest = data(:, 10);



%% Loading training data

data = load('data_train.csv');
X = data(:, [1, 2, 3, 4, 5, 6, 7 ,8 ,9]); y = data(:, 10);




pause;

%% Adding cross validation data to training data

X = [X;Xcv];
y = [y;ycv];

%% PCA for data visualization 

[Xtest, mu, sigma] = featureNormalize(Xtest);

% Feature Normalization
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Project the data onto K = 2 dimension
K = 2;
Z = projectData(X_norm, U, K);

%  Plot the normalized dataset (returned from pca)

plotData(Z, y);
xlabel('Feature projection - 1');
ylabel('Feature projection - 2');
legend('Malign', 'Benign');
title('Breast cancer - cell malignancy data');


pause;

%% Compute Cost and Gradient 

% implementing the cost and gradient for logistic regression in costFunction.m

X = X_norm;

%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

%fprintf('\nCost at initial theta (zeros): %f\n', cost);
%fprintf('Gradient at initial theta (zeros): \n');
%fprintf(' %f \n', grad);


pause;

%% Optimizing using fminunc 

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

%fprintf('\nCost at theta found by fminunc: %f\n', cost);
%fprintf('theta: \n');
%fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, Z, y);

hold on;
xlabel('Feature projection - 1');
ylabel('Feature projection - 2');
title('Breast cancer - cell malignancy hypothesis');
hold off;

pause;





[p, F1] = predict(theta, X, y);
fprintf('\nTrain Accuracy: %f\nF1 score: %f\n\n', mean(double(p == y)) * 100, F1);

[mtest, ntest] = size(Xtest);
Xtest = [ones(mtest, 1) Xtest];

fprintf('----------------------------------------------------------------------------')

%randomly selects a row from dataset and predicts
fprintf('\n Randomly Selected Row: \n')
rowNumber = randperm(size(Xtest,1),1)
fprintf('\n Randomly Selected Row features after Normalisation: \n')
B = Xtest(rowNumber,:)
prob=sigmoid(B*theta);
if(prob>=0.65)
 printf(' Tumour condition: MALIGNANT \n\n')
 hold on
else
 printf(' Tumour condition: BENIGN \n\n')
 hold on
end
fprintf('----------------------------------------------------------------------------')

%predicting from the user input
printf('\n \n Enter the Patient Tumour features data: \n\n');
I(1,1)=1;
I(1,2)=input('Enter Clump Thickness: ');
I(1,3)=input('Enter Uniformity of Cell Size: ');
I(1,4)=input('Enter the Uniformity of Cell Shape: ');
I(1,5)=input('Enter the Marginal Adhesion: ');
I(1,6)=input('Enter the Single Epithelial Cell Size:');
I(1,7)=input('Enter the Bare Nuclei:');
I(1,8)=input('Enter the Bland Chromatin:');
I(1,9)=input('Enter the Normal Nucleoli:');
I(1,10)=input('Enter the Mitoses:');
I;
%feature normalise user input vector I
[I, mu, sigma] = featureNormalize(I);
prob=sigmoid(I*theta);
%boundary sigmoid value kept as 0.65
if(prob>=0.65)
 printf('\n Patient tumour condition: MALIGNANT \n\n')
 hold on
else
 printf('\n Patient tumour condition: BENIGN \n\n')
 hold on
end
% Compute accuracy on test set
[ptest, F1test] = predict(theta, Xtest, ytest);
fprintf('Test Accuracy: %f\nF1 score: %f\n', mean(double(ptest == ytest)) * 100, F1test);




%Sample Values for testing:

%[1 1.10E+00 -2.07E+00 1.27E+00 9.84E-01 1.57E+00 3.28E+00 2.65E+00 2.53E+00 2.22E+00]
%This value should be 1

%[1 1.06E+00 -1.41E+00 9.32E-01 9.59E-01 -1.28E+00 -7.99E-01 -5.57E-01 -1.84E-01 -2.16E+00]
%This value should be 0