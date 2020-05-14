% Load data
[training_X1, training_Y1, training_y1] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_1.mat');
[training_X2, training_Y2, training_y2] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_2.mat');
[training_X3, training_Y3, training_y3] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_3.mat');
[training_X4, training_Y4, training_y4] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_4.mat');
[training_X5, training_Y5, training_y5] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_5.mat');
training_X = [training_X1; training_X2; training_X3; training_X4; training_X5];
training_Y = [training_Y1 training_Y2 training_Y3 training_Y4 training_Y5];
training_y = [training_y1; training_y2; training_y3; training_y4; training_y5];
validation_X = training_X(1:5000, :);
validation_Y = training_Y(:,1:5000);
validation_y = training_y(1:5000);
training_X = training_X(5001: size(training_X, 1),:);
training_Y = training_Y(:,5001: size(training_Y, 2));
training_y = training_y(5001: size(training_y));

% [training_X, training_Y, training_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_1.mat');
% [validation_X, validation_Y, validation_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\data_batch_2.mat');
[test_X, test_Y, test_y] = LoadBatch('D:\学习资料\KTH\Deep learning in data science\Assignment1\mfiles\Datasets\test_batch.mat');

% Pre-process data
training_X = double(training_X');
validation_X = double(validation_X');
test_X = double(test_X');
training_X = Normalize(training_X);
validation_X = Normalize(validation_X);
test_X = Normalize(test_X);

lambda = 6.3e-3;
%     le = le_min + (le_max - le_min)*rand(1, 1)
%     lambda(k) = 10^le(k);
%Parameters
% eta = 0.002;
eta_min = 1e-5;
eta_max = 0.15;
%     lambda = 0.01;
n_batch = 100;
n_epochs = 8;
delta = 1e-6;
d = size(training_X, 1);
n_training = size(training_X, 2);
n_hidden = 200;

% Initialization
% W = var*randn(10, size(training_X, 1));
% b = var*randn(10, 1);
a = initialization(1/sqrt(d), 1/sqrt(n_hidden), d, n_hidden);

% Check analytic gradient computations
% grad_a = ComputeGradients(training_X(1:10,:), training_Y(:,:), a, lambda);
% [grad_b_num, grad_W_num] = ComputeGradsNumSlow(training_X(1:10,:), training_Y(:,:), a, lambda, delta);
% diff_b2 = abs(grad_a{4} - grad_b_num{2})
% diff_W2 = abs(grad_a{3} - grad_W_num{2})
% diff_b1 = abs(grad_a{2} - grad_b_num{1})
% diff_W1 = abs(grad_a{1} - grad_W_num{1})

% Mini-batch gradient descent algorithm
J_train = zeros(1, n_epochs);
J_validation = zeros(1, n_epochs);
% J_train = zeros(1, n_epochs*size(training_X,2)/n_batch);
% J_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Loss_train = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Loss_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Acc_training = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Acc_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
acc_test = zeros(1, 20);
cost = zeros(1, 20);
eta1 = zeros(1, 20);
t = 1;
l = 0;
ns = 1800;
for i=1:n_epochs
%     [training_X, training_Y, training_y] = shuffle(training_X, training_Y, training_y);
    for j=1:size(training_X,2)/n_batch
        % Cyclic learning rate
        if t>=2*l*ns && t <= (2*l+1)*ns
            eta = eta_min + (t - 2*l*ns)/ns*(eta_max - eta_min)
        end
        if t>=(2*l+1)*ns && t <= (2*l+2)*ns
            eta = eta_max - (t - (2*l+1)*ns)/ns*(eta_max - eta_min);
        end
        if t == (2*l + 2)*ns
            l = l + 1;
        end
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = training_X(:, j_start:j_end);
        Ybatch = training_Y(:, j_start:j_end);
        a = MiniBatchGD(Xbatch, Ybatch, a, lambda, eta);
        if mod(t, 90) == 0
            acc_test(t/90) = ComputeAccuracy(training_X, training_y, a);
            cost(t/90) = ComputeCost(training_X, training_Y, a, lambda);
            eta1(t/90) = eta;
        end
        t = t + 1;
    end
    J_train(i) = ComputeCost(training_X, training_Y, a, lambda);
    J_validation(i) = ComputeCost(validation_X, validation_Y, a, lambda);
%         Loss_train(t) = ComputeLoss(training_X, training_Y, a);
%         Loss_validation(t) = ComputeLoss(validation_X, validation_Y, a);
%         Acc_training(t) = ComputeAccuracy(training_X, training_y, a);
%         Acc_validation(t) = ComputeAccuracy(validation_X, validation_y, a);
%     J_train(i) = ComputeCost(training_X, training_Y, a, lambda);
%     J_validation(i) = ComputeCost(validation_X, validation_Y, a, lambda); 
%     if (mod(i, 10) == 0)
%        eta = eta * 0.9;
%     end
end

% acc_training = ComputeAccuracy(training_X, training_y, a)
% acc_validation = ComputeAccuracy(validation_X, validation_y, a)
% acc_test = ComputeAccuracy(test_X, test_y, a)
plot([0, eta1], [0, acc_test]);
% plot(1:n_epochs, J_validation);
xlabel('Leraning rate');
ylabel('Accuracy');
% legend('Training cost');
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
function [P, h1] = EvaluateClassifier(X, a)
W1 = a{1};
b1 = a{2};
W2 = a{3};
b2 = a{4};
s1 = W1*X + b1*ones(1, size(X,2));
h1 = s1;
h1(find(s1<0)) = 0;
s2 = W2*h1 + b2*ones(1, size(X,2));
% Softmax
s2 = exp(s2);
basis = ones(1, 10)*s2;
P = zeros(size(s2));
for i =1:size(X, 2)
    P(:,i) = s2(:,i)/basis(i);
end
end

% Compute cost
function J = ComputeCost(X, Y, a, lambda)
W1 = a{1};
W2 = a{3};
[P, ~] = EvaluateClassifier(X, a);
reg_term = lambda*(sum(sum(W1.*W1))+sum(sum(W2.*W2)));
error_term = sum(-log(sum(Y.*P, 1)))/size(Y, 2);
J = reg_term + error_term;
end

% Compute loss
function Loss = ComputeLoss(X, Y, a)
[P, ~] = EvaluateClassifier(X, a);
Loss = sum(-log(sum(Y.*P, 1)))/size(Y, 2);
end

% Compute the accuracy
function acc = ComputeAccuracy(X, y, a)
[P, ~] = EvaluateClassifier(X, a);
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
function grad_a = ComputeGradients(X, Y, a, lambda)
[P, h] = EvaluateClassifier(X, a);
W1 = a{1};
W2 = a{3};
G = P - Y;
grad_W2 = G*h'/size(X, 2) + 2*lambda*W2;
grad_b2 = G*ones(size(X, 2), 1)/size(X, 2);
G = W2'*G;
h2 = h;
h2(find(h>0)) = 1;
G = G.*h2;
grad_W1 = G*X'/size(X,2) + 2*lambda*W1;
grad_b1 = G*ones(size(X,2),1)/size(X,2);
grad_a = {grad_W1, grad_b1, grad_W2, grad_b2};
end

% Mini-batch gradient descent algorithm
function update_a = MiniBatchGD(X, Y, a, lambda, eta)
grad_a = ComputeGradients(X, Y, a, lambda);
update_a = a;
for i = 1:length(a)
    update_a{i} = a{i} - eta*grad_a{i};
end
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

%Initialization
function a = initialization(var1, var2, d, m)
a = cell(4, 1);
W1 = var1*randn(m, d);
b1 = var1*randn(m, 1);
W2 = var2*randn(10, m);
b2 = var2*randn(10, 1);
a = {W1, b1, W2, b2};
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, a, lambda, h)

W = {a{1}, a{3}};
b = {a{2}, a{4}};
grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        a1 = a;
        a1{2*j} = b_try{j};
        c1 = ComputeCost(X, Y, a1, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        a1{2*j} = b_try{j};
        c2 = ComputeCost(X, Y, a1, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        a1 = a;
        a1{2*j-1} = W_try{j};
        c1 = ComputeCost(X, Y, a1, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        a1{2*j-1} = W_try{j};
        c2 = ComputeCost(X, Y, a1, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
