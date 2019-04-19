function mini_batches = create_mini_batches(obj, X,y, batch_size )

%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

data_values = [X,y];
[Nx,Nd] = size(data_values);

pi = randperm(Nx);

data_values = data_values(pi,:);%TODO  shuffle your data
n_mini_batches = round(Nx/batch_size);%TODO  based on your data and the batch size compute the number of batches
mini_batches = zeros(batch_size,3,n_mini_batches);

for i = 1:n_mini_batches
   %TODO extract the minibatch values
   mini_batches(:,:,i) = data_values(((i-1)*batch_size+1):i*batch_size,:); 
end

end