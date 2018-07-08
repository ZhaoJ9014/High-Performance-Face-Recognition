% Train Joint Bayes script
clc
clear all;

addpath(genpath('/Users/zhaojian/Desktop/MSLowShotC2/Evaluation/UTIL/mypca'));
addpath(genpath('/Users/zhaojian/Desktop/MSLowShotC2/Evaluation/UTIL/joint bayesian/'));

% Load training data
disp('Loading ...');
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/S'
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/eigvector'
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/mean_fea'

Fea_train_jb = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_train_jb.mat');
train_feat = Fea_train_jb.Fea_train_jb;

Label_train_jb = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Label_train_jb.mat');
train_label = Label_train_jb.Label_train_jb;

% Perform PCA
% train_feat= svm_scale(Fea_train_jb);
% mean_fea = mean(train_feat, 1);
% train_feat = train_feat - repmat(mean_fea, size(train_feat,1), 1);
% disp('start pca');
% % % % --------------------whitening-------------------- 
% epsilon = 0.1;
% options.ReducedDim = 512;
% [eigvector, S, eigvalue] = PCA(train_feat, options); 
% train_feat = train_feat * eigvector* diag(1./sqrt(diag(S) + epsilon));
% 
% save('/Users/zhaojian/Desktop/MSLowShotC2/jb_models/eigvector','eigvector');
% save('/Users/zhaojian/Desktop/MSLowShotC2/jb_models/S','S');
% save('/Users/zhaojian/Desktop/MSLowShotC2/jb_models/mean_fea','mean_fea');

train_feat = train_feat';

disp('training plda model...');
plda_model = trainplda(train_feat, train_label);
save('/Users/zhaojian/Desktop/MSLowShotC2/jb_models/jb_model.mat','plda_model');