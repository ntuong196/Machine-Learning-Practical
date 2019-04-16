%% CAB420 Assignment 1
%
% Authors: 
% - Minh Nhat Tuong - Nguyen, n9776001
% - Huy - Nguyen, n9999999
% 
%
%% 1. Features, Classes, and Linear Regression
% Clean up
clc
clear
close all
%%
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
%%
% (b) Create a linear regression learner using the above functions. 
% Plot it on the same plot as the training data.

linXtr = polyx(Xtr, 1);
learner_linear = linearReg(linXtr, Ytr);
xline = [0:.01:1]'; % Transpose
yline = predict(learner_linear, polyx(xline, 1));

figure('name', 'Linear regression predictor');
plot(xline, yline);
hold on % Plot training data and label figure.
plot (Xtr, Ytr, 'bo');
legend('Linear Predictor','Training data');
xlabel('x');
ylabel('y');
title('Linear regression predictor');

%% 
% Alternative quadratic predictor.
quadXtr = polyx(Xtr, 2);
learner_quadratic = linearReg(quadXtr, Ytr); % Create and learn a regression predictor from the data Xtr, ytr.
xline = [0:.01:1]' ; % Transpose
yline = predict(learner_quadratic, polyx(xline, 2)); % Assuming quadratic features
figure('name', 'Quadratic linear predictor');
plot(xline, yline);
hold on % Plot training data and label figure.
plot (Xtr, Ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quadratic linear predictor');

%%
% (c) Create plots with the data and a higher-order polynomial (3, 5, 7, 9, 11, 13).

quinXtr = polyx(Xtr, 5);
learner_quintic = linearReg (quinXtr , Ytr);
yline = predict (learner_quintic , polyx(xline ,5)); % assuming quintic features
figure('name', 'Quintic linear predictor');
plot ( xline , yline );
hold on
plot (Xtr, Ytr, 'bo');
legend('Linear Predictor', 'Training Data');
title('Quintic linear predictor');

%%
% (d) Calculate the mean squared error associated with each of your learned 
%     models on the training data.
% Linear
yhat = predict(learner_linear, linXtr);
mseLinTrain = immse(yhat, Ytr);
fprintf('The MSE for the linear predictor on training data was: %.4f\n', mseLinTrain);

% Quadratic
yhat = predict(learner_quadratic, quadXtr);
mseQuadTrain = immse(yhat, Ytr);
fprintf('The MSE for the quadratic linear predictor on training data was: %.4f\n', mseQuadTrain);

% Quintic
yhat = predict(learner_quintic, quinXtr);
mseQinTrain = immse(yhat, Ytr);
fprintf('The MSE for the quintic linear predictor on training data was: %.4f\n', mseQinTrain);

%%
% (e,f,g) Calculate the MSE for each model on the test data (in mTestData.txt).
% Compare the obtained MAE values with the MSE values obtained above.

mTest = load('data/mTestData.txt');
xtest = mTest(: ,1); ytest = mTest(: ,2);
% Linear
Xtest = polyx(xtest, 1);
yhat = predict(learner_linear, Xtest);
mseLinTest = immse(yhat, ytest);
fprintf('The MSE for the linear predictor on test data was: %.4f\n', mseLinTest);
% Quadratic
Xtest = polyx(xtest, 2);
yhat = predict(learner_quadratic, Xtest);
mseQuadTest = immse(yhat, ytest);
fprintf('The MSE for the quadratic linear predictor on test data was: %.4f\n', mseQuadTest);
% Quintic
Xtest = polyx(xtest, 5);
yhat = predict(learner_quintic, Xtest);
mseQuinTest = immse(yhat, ytest);
fprintf('The MSE for the quintic linear predictor on test data was: %.4f\n', mseQuinTest);



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

