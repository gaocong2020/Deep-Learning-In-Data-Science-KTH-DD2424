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

%Parameters 
lambda = 4e-3;
eta_min = 1e-5;
eta_max = 0.1;
n_batch = 100;
n_epochs = 20;
delta = 1e-6; 
d = size(training_X, 1);
n_training = size(training_X, 2);
k = 3;
n_hiddens = [50, 50];

% Initialization
[W, b] = initialization(d, n_hiddens, k);
% Check analytic gradient computations
% NetParams = SetParams(W, b, 1, gamma, beta);
% [grad_W, grad_b, grad_gamma, grad_beta] = ComputeGradients(training_X(1:10,1:100), training_Y(:,1:100), W, b, gamma, beta, k, lambda);
% Grads = ComputeGradsNumSlow(training_X(1:10,1:100), training_Y(:,1:100), NetParams, lambda, delta);
% diff_W = cell(k, 1);
% diff_b = cell(k, 1);
% for i = 1:k
%     diff_W{i} = Grads.W{i} - grad_W{i};
%     diff_W{i}
%     diff_b{i} = Grads.b{i} - grad_b{i};
%     diff_b{i}
% end
% for i = 1:k-1
%     diff_gamma{i} = Grads.gammas{i} - grad_gamma{i};
%     diff_gamma{i}
%     diff_beta{i} = Grads.betas{i} - grad_beta{i};
%     diff_beta{i}
% end

% Mini-batch gradient descent algorithm
J_train = zeros(1, n_epochs);
J_validation = zeros(1, n_epochs);
% J_train = zeros(1, n_epochs*size(training_X,2)/n_batch);
% J_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Loss_train = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Loss_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Acc_training = zeros(1, n_epochs*size(training_X,2)/n_batch);
% Acc_validation = zeros(1, n_epochs*size(training_X,2)/n_batch);
% acc_test = zeros(1, 20);
t = 1;
l = 0;
ns = 2250;
alpha = 0.9;
[P, x, mu_av, var_av, s, s_BN] = EvaluateClassifier(training_X(:, 1:n_batch), W, b, k);
for i=1:n_epochs
    [training_X, training_Y, training_y] = shuffle(training_X, training_Y, training_y);
    for j=1:size(training_X,2)/n_batch
        % Cyclic learning rate
        if t>=2*l*ns && t <= (2*l+1)*ns
            eta = eta_min + (t - 2*l*ns)/ns*(eta_max - eta_min);
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
        [W, b, mu, variance] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, eta, k);
        for p = 1:k-1
            mu_av{p} = alpha*mu_av{p} + (1 - alpha)*mu{p};
            var_av{p} = alpha*var_av{p} + (1 - alpha)*variance{p};
        end
%         if mod(t, 225) == 0
%             acc_test(t/225) = ComputeAccuracy(test_X, test_y, a);
%         end
        t = t + 1;
    end
    J_train(i) = ComputeCost(training_X, training_Y, W, b, mu_av, var_av, lambda, k);
    J_validation(i) = ComputeCost(validation_X, validation_Y, W, b, mu_av, var_av, lambda, k);
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

acc_training = ComputeAccuracy(training_X, training_y, W, b, mu_av, var_av, k)
acc_validation = ComputeAccuracy(validation_X, validation_y, W, b, mu_av, var_av, k)
acc_test = ComputeAccuracy(test_X, test_y, W, b, mu_av, var_av, k)
plot(1:n_epochs, J_train);
hold on
plot(1:n_epochs, J_validation);
xlabel('Epoch');
ylabel('Cost');
legend('Training cost', 'Validation cost');
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
function [P, x, mu, variance, s, s_BN] = EvaluateClassifier(X, W, b, k)
n = size(X, 2);
s = cell(k, 1);
x = cell(k-1, 1);
mu = cell(k-1, 1);
variance = cell(k-1, 1);
s_BN = cell(k-1, 1);
s{1} = W{1}*X + b{1}*ones(1, n);
mu{1} = mean(s{1}, 2);
variance{1} = var(s{1}, 0, 2)*(n - 1)/n;
s_BN{1} = diag((variance{1} + eps).^(-0.5))*(s{1} - mu{1}*ones(1, n));
x{1} = s_BN{1};
x{1}(find(s_BN{1}<0)) = 0;
for i = 2:k-1
    s{i} = W{i}*x{i-1} + b{i}*ones(1, n);
    mu{i} = mean(s{i}, 2);
    variance{i} = var(s{i}, 0, 2)*(n - 1)/n;
    s_BN{i} = diag((variance{i} + eps).^(-0.5))*(s{i} - mu{i}*ones(1, n));
    x{i} = s_BN{i};
    x{i}(find(s_BN{i}<0)) = 0;
end
s{k} = W{k}*x{k-1} + b{k}*ones(1, n);
% Softmax
s{k} = exp(s{k});
basis = ones(1, 10)*s{k};
P = zeros(size(s{k}));
for i =1:n
    P(:,i) = s{k}(:,i)/basis(i);
end
end

function [P, x, mu, variance, s, s_BN] = EvaluateClassifier_GD(X, W, b, mu_av, var_av, k)
n = size(X, 2);
s = cell(k, 1);
x = cell(k-1, 1);
mu = cell(k-1, 1);
variance = cell(k-1, 1);
s_BN = cell(k-1, 1);
s{1} = W{1}*X + b{1}*ones(1, n);
s_BN{1} = diag((var_av{1} + eps).^(-0.5))*(s{1} - mu_av{1}*ones(1, n));
x{1} = s_BN{1};
x{1}(find(s_BN{1}<0)) = 0;
for i = 2:k-1
    s{i} = W{i}*x{i-1} + b{i}*ones(1, n);
    s_BN{i} = diag((var_av{i} + eps).^(-0.5))*(s{i} - mu_av{i}*ones(1, n));
    x{i} = s_BN{i};
    x{i}(find(s_BN{i}<0)) = 0;
end
s{k} = W{k}*x{k-1} + b{k}*ones(1, n);
% Softmax
s{k} = exp(s{k});
basis = ones(1, 10)*s{k};
P = zeros(size(s{k}));
for i =1:n
    P(:,i) = s{k}(:,i)/basis(i);
end
end

% Compute cost
function J = ComputeCost(X, Y, W, b, mu_av, var_av, lambda, k)
reg_term = 0;
for i = 1:k
    reg_term = reg_term + sum(sum(W{i}.*W{i}));
end
[P, ~, ~, ~, ~, ~] = EvaluateClassifier_GD(X, W, b, mu_av, var_av, k);
reg_term = lambda*reg_term;
error_term = sum(-log(sum(Y.*P, 1)))/size(Y, 2);
J = reg_term + error_term;
end

% Compute the accuracy
function acc = ComputeAccuracy(X, y, W, b, mu_av, var_av, k)
[P, ~, ~, ~, ~, ~] = EvaluateClassifier_GD(X, W, b, mu_av, var_av, k);
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
function [grad_W, grad_b, mu, variance] = ComputeGradients(X, Y, W, b, k, lambda)
n = size(X, 2);
grad_W = cell(k, 1);
grad_b = cell(k, 1);
[P, x, mu, variance, s, s_BN] = EvaluateClassifier(X, W, b, k);
G = P - Y;
grad_W{k} = G*x{k-1}'/size(X, 2) + 2*lambda*W{k};
grad_b{k} = G*ones(n, 1)/n;
G = W{k}'*G;
h = x{k-1};
h(find(x{k-1}>0)) = 1;
G = G.*h;
for i = k-1:-1:1
    % BatchNormBackPass
    sigma1 = (variance{i} + eps).^(-0.5);
    sigma2 = (variance{i} + eps).^(-1.5);
    G1 = G.*(sigma1*ones(1, n));
    G2 = G.*(sigma2*ones(1, n));
    D = s{i} - mu{i}*ones(1, n);
    c = (G2.*D)*ones(n, 1);
    G = G1 - (G1*ones(n, 1))*ones(1, n)/n - D.*(c*ones(1, n))/n;
    if i == 1
        grad_W{1} = G*X'/n + 2*lambda*W{1};
        grad_b{1} = G*ones(n,1)/n;
        break
    end
    grad_W{i} = G*x{i-1}'/n + 2*lambda*W{i};
    grad_b{i} = G*ones(n, 1)/n;
    G = W{i}'*G;
    h = x{i-1};
    h(find(x{i-1}>0)) = 1;
    G = G.*h;
end
end

% Mini-batch gradient descent algorithm
function [update_W, update_b, mu, variance] = MiniBatchGD(X, Y, W, b, lambda, eta, k)
[grad_W, grad_b, mu, variance] = ComputeGradients(X, Y, W, b, k, lambda);
update_W = W;
update_b = b;
for i = 1:k
    update_W{i} = W{i} - eta*grad_W{i};
    update_b{i} = b{i} - eta*grad_b{i};
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
function [W, b] = initialization(d, n_hiddens, k)
W = cell(k, 1);
b = cell(k, 1);
n_hiddens  = [d, n_hiddens, 10];
for i = 1:k
    W{i} = 0.0001*randn(n_hiddens(i+1), n_hiddens(i));
%     W{i} = (1/sqrt(n_hiddens(i)))*randn(n_hiddens(i+1), n_hiddens(i));
    b{i} = randn(n_hiddens(i+1), 1);
end
end

function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)
k = numel(NetParams.W);
Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y,NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y,NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry.W, NetTry.b, NetTry.gammas, NetTry.betas, lambda, k);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end

function NetParams = SetParams(W, b, use_bn,gammas, betas)
    NetParams.W = W;
    NetParams.b = b;
    NetParams.use_bn = use_bn;
    NetParams.betas = betas;
    NetParams.gammas = gammas;
end