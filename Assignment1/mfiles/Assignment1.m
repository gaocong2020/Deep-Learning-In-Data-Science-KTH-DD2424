% Load data
% [training_X1, training_Y1, training_y1] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_1.mat');
% [training_X2, training_Y2, training_y2] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_2.mat');
% [training_X3, training_Y3, training_y3] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_3.mat');
% [training_X4, training_Y4, training_y4] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_4.mat');
% [training_X5, training_Y5, training_y5] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_5.mat');
% training_X = [training_X1; training_X2; training_X3; training_X4; training_X5];
% training_Y = [training_Y1 training_Y2 training_Y3 training_Y4 training_Y5];
% training_y = [training_y1; training_y2; training_y3; training_y4; training_y5];
% validation_X = training_X(1:1000, :);
% validation_Y = training_Y(:,1:1000);
% validation_y = training_y(1:1000);
% training_X = training_X(1001: size(training_X, 1),:);
% training_Y = training_Y(:,1001: size(training_Y, 2));
% training_y = training_y(1001: size(training_y));

[training_X, training_Y, training_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_1.mat');
[validation_X, validation_Y, validation_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_2.mat');
[test_X, test_Y, test_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\test_batch.mat');

% Pre-process data
training_X = double(training_X');
validation_X = double(validation_X');
test_X = double(test_X');
training_X = Normalize(training_X);
validation_X = Normalize(validation_X);
test_X = Normalize(test_X);

%Parameters
% var = 1/sqrt(3072);
var = 0.01;
eta = 0.001;
lambda = 0.01;
n_batch = 500;
n_epochs = 500;
h = 1e-6;

% Initialization
W = var*randn(10, size(training_X, 1));
b = var*randn(10, 1);

% Evaluate the network function
P = EvaluateClassifier(training_X(1:10,:), W(:,1:10), b);

% Check analytic gradient computations
[grad_W_ana, grad_b_ana] = ComputeGradients(training_X(1:10,:), training_Y(:,:), P, W(:,1:10), lambda);
[grad_b_num, grad_W_num] = ComputeGradsNumSlow(training_X(1:10,:), training_Y(:,:), W(:,1:10), b, lambda, h);
diff_b = abs(grad_b_ana - grad_b_num)
diff_W = abs(grad_W_ana - grad_W_num)

% Mini-batch gradient descent algorithm
J_train = zeros(1, n_epochs);
J_validation = zeros(1, n_epochs);
for i=1:n_epochs
%     [training_X, training_Y, training_y] = shuffle(training_X, training_Y, training_y);
    for j=1:size(training_X,2)/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = training_X(:, j_start:j_end);
        Ybatch = training_Y(:, j_start:j_end);
        [W, b] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, eta);
    end
    J_train(i) = ComputeCost(training_X, training_Y, W, b, lambda);
    J_validation(i) = ComputeCost(validation_X, validation_Y, W, b, lambda); 
%     if (mod(i, 10) == 0)
%        eta = eta * 0.9;
%     end
end

acc_training = ComputeAccuracy(training_X, training_y, W, b)
acc_validation = ComputeAccuracy(validation_X, validation_y, W, b)
acc_test = ComputeAccuracy(test_X, test_y, W, b)
plot(1:n_epochs, J_train);
hold on
plot(1:n_epochs, J_validation);
xlabel('Epoch');
ylabel('Loss');
legend('Training loss', 'Validation loss');
% visualize(W);

% Load batch data
function [X, Y, y] = LoadBatch(filename)

A = load(filename);
X = A.data;
y = A.labels;
Y = zeros(10, size(X,1));
for i = 1:size(X,1)
    Y(y(i)+1, i) = 1;
end
end

% Normalize the raw data
function X = Normalize(raw_data)
mean_X = mean(raw_data, 2);
std_X = std(raw_data, 0, 2);
X = raw_data - repmat(mean_X, [1, size(raw_data, 2)]);
X = X ./ repmat(std_X, [1, size(raw_data, 2)]);
end

% Evaluate the network function
function P = EvaluateClassifier(X, W, b)
s = W*X + b*ones(1, size(X,2));
% Softmax
s = exp(s);
basis = ones(1, 10)*s;
P = zeros(size(s));
for i =1:size(X, 2)
    P(:,i) = s(:,i)/basis(i);
end
end

% Compute cost
function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
reg_term = lambda*sum(sum(W.*W));
error_term = sum(-log(sum(Y.*P, 1)))/size(Y, 2);
J = reg_term + error_term;
end

% Compute the accuracy
function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~, index] = max(P);
sum = 0;
for i = 1: size(X,2)
    if index(i) == y(i)+1
        sum = sum + 1;
    end
end
acc = sum/size(X,2);
end

%Compute gradients
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
grad_W = (P - Y)*X'/size(X,2) + 2*lambda*W;
grad_b = (P - Y)*ones(size(X,2),1)/size(X,2);
end

% Mini-batch gradient descent algorithm
function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, eta)
P = EvaluateClassifier(X, W, b);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);
Wstar = W-eta*grad_W;
bstar = b-eta*grad_b;
end

% Visualization
function visualize(W)
mt = [];
for i=1:10
  im = reshape(W(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
  mt = [mt s_im{i}];
  montage(mt);
end
end

function [X_shuffle, Y_shuffle, y_shuffle] = shuffle(X, Y, y)
seq = randperm(size(X, 2));
X_shuffle = X;
Y_shuffle = Y;
y_shuffle = y;
for i = 1: size(X,2)
    X_shuffle(:,i) = X(:,seq(i));
    Y_shuffle(:,i) = Y(:,seq(i));
    y_shuffle(i) = y(seq(i));
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end
