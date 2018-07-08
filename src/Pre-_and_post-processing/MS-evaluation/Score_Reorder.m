DenseNet_GoogleNet_Dev_Novel1_Score = load('/media/zhaojian/6TB/MSLowShotC2/C2_DenseNet_GoogleNet_BN_Results/Dev_DenseNet_GoogleNet_BN_Predict_Score.mat');
DenseNet_GoogleNet_Dev_Novel1_Score = DenseNet_GoogleNet_Dev_Novel1_Score.Predict_Score;
DenseNet_GoogleNet_Dev_Novel1_MID = load('/media/zhaojian/6TB/MSLowShotC2/C2_DenseNet_GoogleNet_BN_Results/Dev_DenseNet_GoogleNet_BN_Predict_MID.mat');
DenseNet_GoogleNet_Dev_Novel1_MID = DenseNet_GoogleNet_Dev_Novel1_MID.Predict_MID;

Reordered_DenseNet_GoogleNet_Dev_Novel1_Score = zeros(5000, 1);
Reordered_DenseNet_GoogleNet_Dev_Novel1_MID = cell(5000, 1);

for i=1:size(DenseNet_GoogleNet_Dev_Novel1_Score, 1)
    
    disp(i);
    tmp_1 = cy_list(i, :);
    
    for j=1:size(DenseNet_GoogleNet_Dev_Novel1_Score, 1)
        
        tmp_2 = Google_list(j, :);
        
        if strcmp(tmp_1, tmp_2)
        
            Reordered_DenseNet_GoogleNet_Dev_Novel1_Score(i, :) = DenseNet_GoogleNet_Dev_Novel1_Score(j, :);
            tmp = DenseNet_GoogleNet_Dev_Novel1_MID(j, :);
            Reordered_DenseNet_GoogleNet_Dev_Novel1_MID(i, :) = cellstr(tmp{1}(1:end-4));
        
        end
        
    end
    
end

save('/media/zhaojian/6TB/MSLowShotC2/C2_DenseNet_GoogleNet_BN_Results/Reordered_DenseNet_GoogleNet_Dev_Novel1_Score.mat', 'Reordered_DenseNet_GoogleNet_Dev_Novel1_Score');
save('/media/zhaojian/6TB/MSLowShotC2/C2_DenseNet_GoogleNet_BN_Results/Reordered_DenseNet_GoogleNet_Dev_Novel1_MID.mat', 'Reordered_DenseNet_GoogleNet_Dev_Novel1_MID');