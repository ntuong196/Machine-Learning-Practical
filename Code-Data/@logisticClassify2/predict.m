function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

num_samples = size(Xte,1); 
Xte = [ones(num_samples,1) Xte]; 
Yte = sign(Xte * obj.wts'); 
Yte = obj.classes(ceil((Yte+3)/2));
