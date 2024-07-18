function [ L,E,rank] = dictionary_learning( XX, opts)
[L,E,rank] = trpca_tnn(XX,opts.lambda,opts);
end

