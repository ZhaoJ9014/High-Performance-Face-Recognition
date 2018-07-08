clc
clear all;

load '/home/zhaojian/Documents/Projects/FACE/IJBA_split1/Evaluation/verification/train_featureProcess/features_negative.mat'
load '/home/zhaojian/Documents/Projects/FACE/IJBA_split1/Evaluation/identification/probe_featureProcess/features_probe.mat'
load '/home/zhaojian/Documents/Projects/FACE/IJBA_split1/Evaluation/identification/gallery_featureProcess/features_gallery.mat'

features_negative=svm_scale(features_negative);
%reduce dimentation using PCA
mean_fea = mean(features_negative,1);
features_negative = features_negative - repmat(mean_fea, size(features_negative,1), 1);
disp('start pca');
tic
options.ReducedDim = 2048;
[eigvector, S, eigvalue] = PCA(features_negative, options);       
epsilon = 0.1; 
% % % % --------------------whitening-------------------- 
features_negative = features_negative * eigvector* diag(1./sqrt(diag(S) + epsilon));
save('features_negative', 'features_negative');

features_probe=svm_scale(features_probe);
%reduce dimentation using PCA
features_probe = features_probe - repmat(mean_fea, size(features_probe,1), 1);
disp('start pca');      
% % % % --------------------whitening-------------------- 
features_probe = features_probe * eigvector* diag(1./sqrt(diag(S) + epsilon));
save('features_probe', 'features_probe');

features_gallery=svm_scale(features_gallery);
%reduce dimentation using PCA
features_gallery = features_gallery - repmat(mean_fea, size(features_gallery,1), 1);
disp('start pca');      
% % % % --------------------whitening-------------------- 
features_gallery = features_gallery * eigvector* diag(1./sqrt(diag(S) + epsilon));
save('features_gallery', 'features_gallery');