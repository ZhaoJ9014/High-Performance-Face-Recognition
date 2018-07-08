%% Test & evaluation script
clc
clear all;

disp('Loading ...');
% Load pre-processed features
Dev_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/DevFea.mat');
Dev_Fea = Dev_Fea.DevFea;

Dev_Fea = Dev_Fea(20001:end, :);

% Perform PCA
Dev_Fea = svm_scale(Dev_Fea);
mean_fea = mean(Dev_Fea, 1);
Dev_Fea = Dev_Fea - repmat(mean_fea, size(Dev_Fea, 1), 1);
disp('start pca');
options.ReducedDim = 200;
[eigvector, S, eigvalue] = PCA(Dev_Fea, options);       
epsilon = 0.1;
% % --------------------whitening-------------------- 
Dev_Fea = Dev_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

BaseSet_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_Sub_BaseSet.mat');
BaseSet_Fea = BaseSet_Fea.Fea_Sub;
NovelSet_1_Fea = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_Sub_NovelSet_1.mat');
NovelSet_1_Fea = NovelSet_1_Fea.Fea_Sub;
Ori_Fea = [BaseSet_Fea; NovelSet_1_Fea];

% Perform PCA
Ori_Fea = svm_scale(Ori_Fea);
Ori_Fea = Ori_Fea - repmat(mean_fea, size(Ori_Fea,1), 1);
disp('start pca');
% % --------------------whitening-------------------- 
Ori_Fea = Ori_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

% Load labels
MID_DEV = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_DEV.mat');
MID_BASE = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_BASE.mat');
MID_NOVEL = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_NOVEL.mat');
MID_DEV = MID_DEV.MID_DEV;
MID_ORI = [MID_BASE.MID_BASE; MID_NOVEL.MID_NOVEL];

MID_DEV = MID_DEV(20001:end, 1);

% Compute Cosine similarities
disp('Compute Cosine similarities ...');
score = Dev_Fea * Ori_Fea';

% Predict score and MID
disp('Prediction ...');
Predict_Score = zeros(size(score, 1), 1);
Predict_MID = cell(size(score, 1), 1);
for i=1:size(score, 1)
    
    [tmp, ind] = max(score(i,:));
    Predict_Score(i, 1) = tmp;
    Predict_MID(i, 1) = MID_ORI(ind, 1);
 
end
save('/Users/zhaojian/Desktop/MSLowShotC2/Results/Predict_Score', 'Predict_Score');
save('/Users/zhaojian/Desktop/MSLowShotC2/Results/Predict_MID', 'Predict_MID');
%% ==========================================
% Evaluate Coverage (M/N) @ Precision (C/M) = 0.99 and 0.95
disp('Evaluation ...');
C = 0;
N = size(Predict_Score, 1);
for i=1:size(Predict_Score, 1)

    if(strcmp(MID_DEV{i ,1}, Predict_MID{i, 1}(1:end-4)))
        
        C = C + 1;
        
    end

end

M_099 = C / 0.99;
M_095 = C / 0.95;

Coverage_099 = M_099 / N;
Coverage_095 = M_095 / N;

disp('Coverage @ Precision = 0.99:');
disp(Coverage_099);
disp('Coverage @ Precision = 0.95:');
disp(Coverage_095);