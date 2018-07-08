% Test jb script
clc
clear all;

addpath(genpath('/Users/zhaojian/Desktop/MSLowShotC2/Evaluation/UTIL/joint bayesian'));

% Load data and jb model
disp('Loading ...');
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/jb_model';
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/S'
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/eigvector'
load '/Users/zhaojian/Desktop/MSLowShotC2/jb_models/mean_fea'

Dev_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/DevFea.mat');
Dev_Fea = Dev_Fea.DevFea;

% Perform PCA
Dev_Fea = svm_scale(Dev_Fea);
Dev_Fea = Dev_Fea - repmat(mean_fea, size(Dev_Fea, 1), 1);
disp('start pca');
epsilon = 0.1;
% % --------------------whitening-------------------- 
Dev_Fea = Dev_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

Dev_Fea = Dev_Fea(20001:end, :);

BaseSet_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_Sub_BaseSet.mat');
BaseSet_Fea = BaseSet_Fea.Fea_Sub;
NovelSet_1_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_Sub_NovelSet_1.mat');
NovelSet_1_Fea = NovelSet_1_Fea.Fea_Sub;
Ori_Fea = [BaseSet_Fea; NovelSet_1_Fea];

% Perform PCA
Ori_Fea = svm_scale(Ori_Fea);
Ori_Fea = Ori_Fea - repmat(mean_fea, size(Ori_Fea, 1), 1);
disp('start pca');
% % --------------------whitening-------------------- 
Ori_Fea = Ori_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

% Load labels
MID_DEV = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_DEV.mat');
MID_BASE = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_BASE.mat');
MID_NOVEL = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_NOVEL.mat');
MID_DEV = MID_DEV.MID_DEV;
MID_ORI = [MID_BASE.MID_BASE; MID_NOVEL.MID_NOVEL];

MID_DEV = MID_DEV(20001:end, :);

% Compute Cosine similarities
disp('Compute Cosine similarities ...');
score = plda_eval(plda_model, Dev_Fea', Ori_Fea');

% Predict score and MID
disp('Prediction ...');
Predict_Score = zeros(size(score, 1), 1);
Predict_MID = cell(size(score, 1), 1);
for i=1:size(score, 1)
    
    [tmp, ind] = max(score(i,:));
    Predict_Score(i, 1) = tmp;
    Predict_MID(i, 1) = MID_ORI(ind, 1);
 
end
% save('/Users/zhaojian/Desktop/MSLowShotC2/Results/Predict_Score_jb', 'Predict_Score');
% save('/Users/zhaojian/Desktop/MSLowShotC2/Results/Predict_MID_jb', 'Predict_MID');
%% ==========================================
% Evaluate Top-1 acc
disp('Evaluation ...');
C = 0;
N = size(Predict_Score, 1);
for i=1:size(Predict_Score, 1)

    if(strcmp(MID_DEV{i ,1}, Predict_MID{i, 1}(1:end-4)))
        
        C = C + 1;
        
    end

end

Top1 = C / N;

disp('Top-1 acc:');
disp(Top1);