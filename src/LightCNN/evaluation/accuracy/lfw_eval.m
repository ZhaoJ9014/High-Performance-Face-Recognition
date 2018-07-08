clc;
clear all;
close all;

% load feature you extracted
% load('../CNNCaffeExtraction/lfwfeatures.mat');

% or the features extracted by myself 
% model downloaded from "https://github.com/ydwen/caffe-face"
% load('features/lfwfeatures.mat');

% or some other features extracted by myself
% 99.53% performance on LFW after doing a PCA on LFW 13233 faces
load('LFW_feas.mat');

%feature = cell2mat(lfwfeatures);
feature = data';
imglist = label;

% load pair list and label
% generated from original lfw view 2 file. 
fea_list = 'pair.label';
[label img1 img2]= textread(fea_list,'%d %s %s');

% PCA
%[eigvec, ~, ~, sampleMean] = PCA(feature');
%feature = ( bsxfun(@minus, feature', sampleMean)* eigvec )';

% generate scores
for i = 1:size(label,1)
    % find feature 1
    index1 = find(strcmp(imglist, img1{i}) == 1);
    fea1 = feature(:,index1);
    % find feature 2
    index2 = find(strcmp(imglist, img2{i}) == 1);
    fea2 = feature(:,index2);
      
    % cosine distance
    cos(i) = (fea1' * fea2)/(norm(fea1) * norm(fea2));
end

% ROC and accuracy
[fpr, tpr, auc, eer, acc] = ROCcurve(cos, label);
tmp=sprintf('ACC: %f \nEER: %f \nAUC: %f',acc,eer,auc);
disp(tmp);

plot(fpr, tpr);
axis([0,0.05,0.95,1]);
legend('cos');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
hold on;

