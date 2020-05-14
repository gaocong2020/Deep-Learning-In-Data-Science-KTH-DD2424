% Read in the training data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

% Get a vector containing the unique characters in book_data
book_chars = unique(book_data);

% Initialize map containers
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

RNN.K = length(book_chars);
for i=1:RNN.K
  char_to_ind(book_chars(i)) = i;
  ind_to_char(int32(i)) = book_chars(i);
end

% Set hyper-parameters and initialize the RNN's parameters
RNN.m = 100;
eta = 0.1;
RNN.seq_length = 25;
sig = 0.01;
RNN.U = randn(RNN.m, RNN.K)*sig;
RNN.W = randn(RNN.m, RNN.m)*sig;
RNN.V = randn(RNN.K, RNN.m)*sig;
RNN.b = zeros(RNN.m, 1);
RNN.c = zeros(RNN.K, 1);
RNN.g = [4, 5, 6, 7, 8];
RNN.epsilon = 1e-8;
N = size(book_data, 2);
h0 = zeros(RNN.m, 1);
h = 1e-4;
epoch = 7 ;

% syn_ind = synthesize(h0, RNN, 100);
% for i=1:100
%   syn_chars(i) = ind_to_char(find(syn_ind(:, i) == 1));
% end

X = zeros(RNN.K, N);
for i=1:N
    X(char_to_ind(book_data(i)), i) = 1;
end

smooth_loss = ComputeLoss(X(:, 1:RNN.seq_length), X(:, 2:RNN.seq_length+1), RNN, h0)
f = fieldnames(RNN)';
for i=RNN.g
    m.(f{i}) = zeros(size(RNN.(f{i})));
end
loss_table = zeros(1, N*epoch);
count = 0;
for i = 1: epoch
    e = 1;
    hprev = h0;
    while RNN.seq_length+e<N
        X_batch = X(:, e:RNN.seq_length+e-1);
        Y_batch = X(:, e+1:RNN.seq_length+e);
        [RNN, H, m] = MiniBatch(X_batch, Y_batch, RNN, hprev, eta, m);
        loss = ComputeLoss(X_batch, Y_batch, RNN, hprev);
        smooth_loss = 0.999* smooth_loss + 0.001 * loss;
        if mod(count, 10000) == 0 && count <= 100000
            count
            smooth_loss
            syn_ind = synthesize(hprev, X_batch(:, 1), RNN, 200);
            for o=1:200
                syn_chars(o) = ind_to_char(find(syn_ind(:, o) == 1));
            end
            syn_chars
        end
        if mod(count, 100) == 0
            loss_table(count/100 + 1) = smooth_loss;
        end
        hprev = H(:, end);
        e = e + RNN.seq_length;
        count = count + 1;
    end

end

plot(1: count/100 + 1, loss_table(1: count/100 + 1));
xlabel('100 iterations');
ylabel('Loss');
% X_ind = X(:, 1:RNN.seq_length);
% Y_ind = X(:, 2:RNN.seq_length+1);

% ana_grad = ComputeGradients(X_ind, Y_ind, RNN, h0);
% num_grad = ComputeGradNum(X_ind, Y_ind, RNN, h);
% max(max(ana_grad.W - num_grad.W))
% max(max(ana_grad.V - num_grad.V))
% max(max(ana_grad.U - num_grad.U))
% max(max(ana_grad.b - num_grad.b))
% max(max(ana_grad.c - num_grad.c))

function Y_predict = synthesize(h0, x0, RNN, n)
h = h0;
x = x0;
Y_predict = zeros(RNN.K, n);
for i = 1:n
    a = RNN.W*h + RNN.U*x + RNN.b;
    h = tanh(a);
    o = RNN.V*h + RNN.c;
    p = exp(o)/sum(exp(o));
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    Y_predict(ii, i) = 1;
    x = Y_predict(:, i);
end
end

function cost = ComputeLoss(X, Y, RNN, hprev)
[P, ~, ~] = ForwardPass(X, RNN, hprev);
cost = -sum(log(sum(Y.*P, 1)));
end

function [P, H, A] = ForwardPass(X, RNN, hprev)
h = hprev;
P = zeros(RNN.K, RNN.seq_length);
H = zeros(RNN.m, RNN.seq_length);
A = zeros(RNN.m, RNN.seq_length);
for i = 1:RNN.seq_length
    x = X(:, i);
    a = RNN.W*h + RNN.U*x + RNN.b;
    h = tanh(a);
    o = RNN.V*h + RNN.c;
    p = exp(o)/sum(exp(o));
    P(:, i) = p;
    H(:, i) = h;
    A(:, i) = a;
end
end

function [grads, H] = ComputeGradients(X, Y, RNN, hprev)
[P, H, A] = ForwardPass(X, RNN, hprev);
f = fieldnames(RNN)';
dH = zeros(RNN.seq_length, RNN.m);
dA = zeros(RNN.seq_length, RNN.m);
for i=RNN.g
    grads.(f{i}) = zeros(size(RNN.(f{i})));
end
G = P - Y;
for i = 1: RNN.seq_length
    grads.V = grads.V + G(:, i)*H(:,i)';
    grads.c = grads.c + G(:, i);
end
dH(RNN.seq_length, :) = G(:, RNN.seq_length)'*RNN.V;
dA(RNN.seq_length, :) = dH(RNN.seq_length, :)*diag(1-tanh(A(:, RNN.seq_length)).^2);
for i = RNN.seq_length-1:-1:1
    dH(i, :) = G(:, i)'*RNN.V + dA(i+1, :)*RNN.W;
    dA(i, :) = dH(i, :)*diag(1-tanh(A(:, i)).^2);
end
for i = 1: RNN.seq_length
    if i == 1
        grads.W = grads.W + dA(i, :)'*hprev';
    else
        grads.W = grads.W + dA(i, :)'*H(:, i-1)';
    end
    grads.U = grads.U + dA(i, :)'*X(:, i)'; 
    grads.b = grads.b + dA(i, :)';
end
end

function [RNN, H, m] = MiniBatch(X, Y, RNN, hprev, eta, m)
[grads, H] = ComputeGradients(X, Y, RNN, hprev);
f = fieldnames(RNN)';
for i=RNN.g
    m.(f{i}) = m.(f{i})+grads.(f{i}).^2;
    RNN.(f{i}) = RNN.(f{i}) - eta * grads.(f{i})./sqrt(m.(f{i})+RNN.epsilon);
end
end

function num_grads = ComputeGradNum(X, Y, RNN, h)
f = fieldnames(RNN)';
for i = RNN.g
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{i} ]);
    num_grads.(f{i}) = ComputeGradNumSlow(X, Y, f{i}, RNN, h);
end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end
