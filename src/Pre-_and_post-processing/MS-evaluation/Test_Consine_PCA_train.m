%% Test & evaluation script
clc
clear all;

disp('Loading ...');
% Load pre-processed features
BaseSet_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_Fea_Sub_BaseSet.mat');
BaseSet_Fea = BaseSet_Fea.En_Fea_Sub;
NovelSet_1_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_Fea_Sub_NovelSet_1.mat');
NovelSet_1_Fea = NovelSet_1_Fea.En_Fea_Sub_NovelSet_1;
Ori_Fea = [BaseSet_Fea; NovelSet_1_Fea];

% Perform PCA
Ori_Fea = svm_scale(Ori_Fea);
mean_fea = mean(Ori_Fea, 1);
Ori_Fea = Ori_Fea - repmat(mean_fea, size(Ori_Fea,1), 1);
disp('start pca');
% % --------------------whitening-------------------- 
options.ReducedDim = 512;
[eigvector, S, eigvalue] = PCA(Ori_Fea, options);       
epsilon = 0.1;
Ori_Fea = Ori_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

save('/media/zhaojian/6TB/MSLowShotC2/jb_models/eigvector','eigvector');
save('/media/zhaojian/6TB/MSLowShotC2/jb_models/S','S');
save('/media/zhaojian/6TB/MSLowShotC2/jb_models/mean_fea','mean_fea');

Test_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_TestFea.mat');
Test_Fea = Test_Fea.En_TestFea;

% Perform PCA
Test_Fea = svm_scale(Test_Fea);
Test_Fea = Test_Fea - repmat(mean_fea, size(Test_Fea, 1), 1);
disp('start pca');
% % --------------------whitening-------------------- 
Test_Fea = Test_Fea * eigvector * diag(1./sqrt(diag(S) + epsilon));

% Test_Fea = Test_Fea(20001:end, :);

% Load labels
% MID_DEV = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_MID_DEV.mat');
MID_BASE = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_MID_BASE.mat');
MID_NOVEL = load('/media/zhaojian/6TB/MSLowShotC2/Features/En_MID_NOVEL.mat');
% MID_DEV = MID_DEV.En_MID_DEV;
MID_ORI = [MID_BASE.En_MID_BASE; MID_NOVEL.En_MID_NOVEL];

% MID_DEV = MID_DEV(20001:end, :);

% Compute Cosine similarities
disp('Compute Cosine similarities ...');
score = Test_Fea * Ori_Fea';

% Predict score and MID
disp('Prediction ...');
Predict_Score = zeros(size(score, 1), 1);
Predict_MID = cell(size(score, 1), 1);
for i=1:size(score, 1)
    
    [tmp, ind] = max(score(i,:));
    Predict_Score(i, 1) = tmp;
    Predict_MID(i, 1) = MID_ORI(ind, 1);
 
end
save('/media/zhaojian/6TB/MSLowShotC2/Results/Test_DenseNet_GoogleNet_BN_Predict_Score', 'Predict_Score');
save('/media/zhaojian/6TB/MSLowShotC2/Results/Test_DenseNet_GoogleNet_BN_Predict_MID', 'Predict_MID');
%% ==========================================
% Evaluate Top-1 acc
% disp('Evaluation ...');
% C = 0;
% N = size(Predict_Score, 1);
% for i=1:size(Predict_Score, 1)
% 
%     if(strcmp(MID_DEV{i ,1}, Predict_MID{i, 1}(1:end-4)))
%         
%         C = C + 1;
%         
%     end
% 
% end
% 
% Top1 = C / N;
% 
% disp('Top1 acc:');
% disp(Top1);