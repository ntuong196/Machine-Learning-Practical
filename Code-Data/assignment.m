%% CAB420 Assignment 1
%
% Authors: 
% - Minh Nhat Tuong - Nguyen, n9776001
% - Quang Huy Tran, n10069275
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

%%
% (a) Using the knnRegress class, implement (add code to) the predict
% function to make it functional.

%%
% (b) Using the same technique as in Problem 1a, plot the predicted
% function for several values of $k: 1, 2, 3, 5, 10, 50$. How does the
% choice of $k$ relate to the “complexity” of the regression function?

kV = [1,2,3,5,10,50];  % K values of Problem 2 
xline = [0:.01:1]'; % transpose 
figure('Name','Plot the KNN Regression Prediction for different polynomials');
for i = 1:length(kV)
    learner = knnRegress(kV(i),Xtr,Ytr);
    yline = predict(learner,xline);
    subplot(3,2,i); 
    plot(xline,yline,'b');
    title(['k = ',num2str(kV(i))]);
    xlabel('Features');
    ylabel('Labels');
end 

%%
% (c) What kind of functions can be output by a nearest neighbor regression
% function? Briefly justify your conclusion.

% Based on the plot of 2b, it can be observed that the output of a nearest 
% neighbor regression function is a linear function 

%% 3. Hold-out and Cross-validation

%%
% *(a) Similarly to Problem 1 and 2, compute the MSE of the test data on a
% model trained on only the first 20 training data examples for
% $k = 1, 2, 3, . . . , 140$. Plot the MSE versus $k$ on a log-log scale
% (see help loglog).*

%Clean up
clc
clear
close all

%Import data
mTrain = load('data/mTrainData.txt');
mTest = load('data/mTestData.txt');
xtest = mTest(: ,1); % Features
ytest = mTest(: ,2); % Values

% first 20 training data examples
xtr = mTrain(1:20,1);
ytr = mTrain(1:20,2); 
 
% initalise a vector for MSE of each k value
MSE1 = zeros(140, 1); 

for k=1:140 % iterate on each k value
    learner = knnRegress(k, xtr, ytr); % create the learner using k
    yhat = predict(learner, xtest); % predict and find MSE relative to test data
    MSE1(k) = mean((yhat - ytest).^2);
end 
figure;
loglog(1:140, MSE1, 'r'); % plot on a loglog graph
xlabel('K value');
ylabel('MSE');
title('MSE on different K values');
legend('Only using first 20 training data examples');

%%
% *(b) Repeat, but use all the training data. What happened? Contrast with
% your results from problem 1 (hint: which direction is complexity in this picture?).*
%

% all training data examples
xtr = mTrain(:,1);
ytr = mTrain(:,2); 

% initalise a vector for MSE of each k value
MSE2 = zeros(140, 1); 

for k=1:140 % iterate on each k value
    learner = knnRegress(k, xtr, ytr); % create the learner using k
    yhat = predict(learner, xtest); % predict and find MSE relative to test data
    MSE2(k) = mean((yhat - ytest).^2);
end 

figure;
loglog(1:140, MSE1, 'r'); % plot on a loglog graph
xlabel('K value');
ylabel('MSE');
title('MSE on different K values');
hold on;
loglog(1:140, MSE2, 'b'); % plot on the loglog graph
legend('Only using first 20 training data examples', ...
    'All training data examples');
%
% _When using all training data, the MSE was able to be reduced much
% further.Contrast to problem 1, here we are increasing the complexity  of
% the sample (or training data) to reduce cost. Whereas in problem one we
% increased the model complexity by raising to 5th degree polynomial._
%

%%
% *(c) Using only the training data, estimate the curve using 4-fold
% cross-validation. Split the training data into two parts, indices 1:20
% and 21:140; use the larger of the two as training data and the smaller as
% testing data, then repeat three more times with different sets of 20 and
% average the MSE. Add this curve to your plot. Why might we need to use
% this technique?*
%
hold on 
MSE3 = zeros(140, 1); % initalise a vector for MSE of each k value
for k=1:140 % test for 100 values of k
    MSEtemp = zeros(4, 1); % temp mse array for each i (averaged for each k)
    for i=1:4
        m = i*20; n = m-19; % local bounds
        iTest = mTrain(n:m,:); % 20 indicies for testing
        iTrain = setdiff(mTrain, iTest, 'rows'); % rest for training
        learner = knnRegress(k, iTrain(:,2), iTrain(:,1)); % train the learner
        yhat = predict(learner, iTest(:,2)); % predict at testing x values
        MSEtemp(i) = mean((yhat - iTest(:,1)).^2); 
    end
    MSE3(k) = mean(MSEtemp); % average the MSE 
end
loglog(1:140, MSE3, 'm'); % plot on the loglog graph
legend('Only using first 20 training data examples', ...
    'All training data examples', ...
    'All training data examples with 4-fold Cross-validation');
%
% _Using this technique may produce more accurate error values. This is
% because the effective 'testing data' changes multiple times per
% iteration, making it a better representation of the model. Also, in this
% case more data samples are used for testing because mTest only has 60
% samples relative to the 120 in mTrain.
%

%% 4. Nearest Neighbor Classifiers
% Load Iris Data
iris = load('data/iris.txt');
pi = randperm( size (iris ,1));
Y_iris = iris(pi ,5); X_iris = iris(pi ,1:2);

%%
% *(a) Plot the data by their feature values, using the class value to
% select the color.*
classes_iris = unique(Y_iris); % The number of iris classes 
colorString = ['b','g','r'];
figure('Name','Plot the data by the feature values');
for i = 1:length(classes_iris)
    index_feature = find(Y_iris==classes_iris(i));
    scatter(X_iris(index_feature,1),X_iris(index_feature,2),'filled',colorString(i));

    hold on 
end
legend('Class = 0','Class = 1','Class = 2'); 


%%
% *(b) Use the provided knnClassify class to learn a 1-nearest-neighbor
% predictor.*
knn1_classifier = knnClassify(1,X_iris,Y_iris);
class2DPlot(knn1_classifier,X_iris,Y_iris);

%%
% *(c) Do the same thing for several values of k (say, [1, 3, 10, 30]) and
% comment on their appearance.*
kValues = [1,3,10,30]; % K values of Problem 4

for i = 1:length(kValues)
    learner = knnClassify(kValues(i),X_iris,Y_iris);
    class2DPlot(learner,X_iris,Y_iris);
    title(strcat('k = ', num2str(kValues(i)))); 
end


%%
% *(d) Now split the data into an 80/20 training/validation split. For
% $k = [1, 2, 5, 10, 50, 100, 200]$, learn a model on the 80% and calculate
% its performance (# of data classified incorrectly) on the validation
% data. What value of k appears to generalize best given your training
% data? Comment on the performance at the two endpoints, in terms of over-
% or under-fitting.*
%


%% 5. Perceptron and Logistic Regression

% Clean up*
clear;
clc;
close all;
% Load the Iris dataset 
iris = load('data/iris.txt'); 
X = iris(:,1:2); Y = iris(:, end); 
[X Y] = shuffleData(X,Y); 
X = rescale(X); 
XA = X(Y < 2, :); YA = Y(Y < 2); % get class 0 and 1
XB = X(Y > 0, :); YB = Y(Y > 0); 

%%
% *(a) Show the two classes in a scatter plot and verify that one is
% linearly separable while the other is not
figure(); 

%Plot the dataset A having class 0 and 1 
subplot(1,2,1); 
posA = find(YA == 1); negA = find(YA == 0); 

plot(XA(posA,1), XA(posA,2), 'k+','LineWidth', 2,'MarkerSize',7); 
hold on 
plot(XA(negA,1), XA(negA,2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize',7); 
hold off 
legend('Class = 1', 'Class = 0'); 
title('Class 0 vs Class 1 (Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');

%Plot the dataset B having class 1 and 2 
subplot(1,2,2); 
posB = find(YB == 2); negB = find(YB == 1); 
xlabel('Sepal Length');
ylabel('Sepal Width');
plot(XB(posB,1), XB(posB,2), 'k+','LineWidth', 2,'MarkerSize',7); 
hold on 
plot(XB(negB,1), XB(negB,2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize',7); 
hold off 
legend('Class = 2', 'Class = 1'); 
title('Class 1 vs Class 2 (Non-Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');

%%
% *(b) Write (fill in) the function @logisticClassify2/plot2DLinear.m so that
% it plots the two classes of data in dierent colors, along with the
% decision boundary (a line). Include the listing of your code in your
% report. To demo your function plot the decision boundary corresponding
% to the classifier $$ sign(.5 + 1x_1 - .25x_2) $$*
wts = [0.5 1 -0.25]; % Set weights 
 
%Logistic Classification of Dataset A 
learner_XA = logisticClassify2(); 
learner_XA = setClasses(learner_XA, unique(YA)); 
learner_XA = setWeights(learner_XA, wts);
plot2DLinear(learner_XA,XA,YA);     %Plot the linear decision boundary line of dataset A 
title('Linear decision boundary line of dataset A'); 
legend('Class = 0', 'Class = 1', 'Decision Boundary Line');

%Logistic Classification of Dataset B 
learner_XB = logisticClassify2(); 
learner_XB = setClasses(learner_XB, unique(YB)); 
learner_XB = setWeights(learner_XB, wts);
plot2DLinear(learner_XB,XB,YB);      %Plot the linear decision boundary line of dataset B
title('Linear decision boundary line of dataset B'); 
legend('Class = 1', 'Class = 2', 'Decision Boundary Line');


%%
% *(c) Complete the predict.m function to make predictions for your linear classifier.*
% Predict the labels of dataset A and computing error 
YA_pred = predict(learner_XA,XA); 
learnerA_error = 0; 
for i = 1:length(YA_pred)
    if( YA_pred(i) ~= YA(i))
        learnerA_error  =  learnerA_error + 1; 
    end 
end 
error_rate_A = learnerA_error/length(YA_pred); 
fprintf("\nThe error rate of dataset A is %.4f\n",error_rate_A);

% Predict the labels of dataset B and computing error 
YB_pred = predict(learner_XB,XB); 
learnerB_error = 0; 
for i = 1:length(YB_pred)
    if( YB_pred(i) ~= YB(i))
        learnerB_error  =  learnerB_error + 1; 
    end 
end 
a = err(learner_XB,XB,YB); 
error_rate_B = learnerB_error/length(YB_pred); 
fprintf("\nThe error rate of dataset B is %.4f\n",error_rate_B);


%%
% *(d)*


%%
% *(e) Complete your train.m function to perform stochastic gradient descent
% on the logistic loss function.* 


%%
% *(f) Run your logistic regression classifier on both data sets (A and B);
% for this problem, use no regularization $(\alpha = 0)$. Describe your parameter
% choices (stepsize, etc.) and show a plot of both the convergence of the
% surrogate loss and error rate, and a plot of the final converged
% classifier with the data (using e.g. plotClassify2D). In your report,
% please also include the functions that you wrote (at minimum, train.m,
% but possibly a few small helper functions as well)*

%Training the dataset A 
train(learner_XA,XA,YA);
%Training the dataset B
train(learner_XA,XB,YB);

%% 
%Implement the mini batch gradient descent on the logistic function complete in train_in_batches.m 

train_in_batches(learner_XA,XA,YA,11);
%%
close all
