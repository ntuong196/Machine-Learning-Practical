%% CAB420 Assignment 1
%
% Authors: 
% - Minh Nhat Tuong - Nguyen, n9776001
% - 
% 
%
%% 1. Features, Classes, and Linear Regression
% Clean up
clc
clear
close all

disp('1. Features, Classes, and Linear Regression');

% (a) Plot the training data in a scatter plot.

% Load training dataset
% replace \ to load file on windows
mTrain = load('data/mTrainData.txt');

% Separate features
Xtr = mTrain(: ,1); % X = single feature
Ytr = mTrain(: ,2); % Y = target value
whos

% Plot training data
figure('name', 'Training Data');
plot (Xtr, Ytr, 'bo'); 
xlabel('x');
ylabel('y');
title('Plotting all training data points');
legend('Training data');

% (b) Create a linear regression learner using the above functions. 
% Plot it on the same plot as the training data.

linXtr = polyx(xtr, 1);
linLearner = linearReg(linXtr, ytr);
xline = [0:.01:2]'; % Transpose
yline = predict(linLearner, polyx(xline, 1));
plot(xline, yline);
legend('Training data', 'Linear predictor');
xlabel('x');
ylabel('y');

% (c) Create plots with the data and a higher-order polynomial (3, 5, 7, 9, 11, 13).
Xtr = polyx(xtr, 5);
learner_quintic = linearReg (Xtr , ytr);
yline = predict (learner_quintic , polyx(xline ,5)); % assuming quintic features
figure('name', 'Quintic linear predictor');
plot ( xline , yline ,'ro ');
hold on
plot (xtr, ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quintic linear predictor');

% (d) Calculate the mean squared error associated with each of your learned 
%     models on the training data.
% Quadratic
Xtr = polyx(xtr, 2);
yhat = predict(learner_quadratic, Xtr);
mseQuadTrain = immse(yhat, ytr);
fprintf('The MSE for the quadratic linear predictor on training data was: %.2f\n', mseQuadTrain);
% Quintic
Xtr = polyx(xtr, 5);
yhat = predict(learner_quintic, Xtr);
mseQinTrain = immse(yhat, ytr);
fprintf('The MSE for the quintic linear predictor on training data was: %.2f\n', mseQinTrain);

% (e,f,g) Calculate the MSE for each model on the test data (in mTestData.txt).
% Compare the obtained MAE values with the MSE values obtained above.

mTest = load('data/mcycleTest.txt');
ytest = mTest(: ,1); xtest = mTest(: ,2);
% Quadratic
Xtest = polyx(xtest, 2);
yhat = predict(learner_quadratic, Xtest);
mseQuadTest = immse(yhat, ytest);
fprintf('The MSE for the quadratic linear predictor on test data was: %.2f\n', mseQuadTest);
% Quintic
Xtest = polyx(xtest, 5);
yhat = predict(learner_quintic, Xtest);
mseQuinTest = immse(yhat, ytest);
fprintf('The MSE for the quintic linear predictor on test data was: %.2f\n', mseQuinTest);


%% 2. kNN Regression

% Create a list of K values

% Plot training data

% Create and learn a kNN regression predictor from the data Xtr, ytr for each K.


%% 3. Hold-out and Cross-validation

% Create a copy of the data with only the first 20 points

% Pre-allocate a matrix for all of the plotted values

% Plot training data

%% 4. Nearest Neighbor Classifiers

% (A) - Plot dataset by feature values.

% (B) Learn and plot 1-nearest-neighbour predictor

% (C) Repeat for several values of k

% (D) Split data into training (80%) and valuation (20%) data. Train and 
%     validate model for multiple values of k and calculate its
%     performance.

%% 5. Perceptrons and Logistic Regression

% (A) Show the two classes in a scatter plot and verify that one is 
% linearly separable while the other is not.

%% (B) Write the function @logisticClassify2/plot2DLinear.m such that it 
%      can Plot the two classes of data in different colors, along with the 
%      decision boundary (a line). To demo your function plot the decision 
%       boundary corresponding to the classifier sign( .5 + 1x1 ? .25x2 )
%       along with the A data, and again with the B data.


%% (C) Complete the predict.m function to make predictions for your linear 
%      classifier.  Verify that your function works by computing & 
%      reporting the error rate of the classifier in the previous
%      part on both data sets A and B. (The error rate on data set A should
%      be ? 0.0505.)


%% (D) Refer to report.


%% (E) Implemented train.m


%% (F) 

