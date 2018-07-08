% Script for model merge
% clc
% clear all;

Dense_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_TestFea.mat');
Dense_Fea = Dense_Fea.TestFea;
JK_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_TestFea.mat');
JK_Fea = JK_Fea.TestFea;

En_TestFea = zeros(size(JK_Fea, 1), size(JK_Fea, 2) + size(Dense_Fea, 2));

for i=1:size(Dense_Fea, 1)
    
    disp(i);
    tmp_1 = Dense_list(i, :);
    
    for j=1:size(JK_Fea, 1)
        
        tmp_2 = JK_list(j, :);
        
        if strcmp(tmp_1, tmp_2)
        
            En_TestFea(i, :) = [Dense_Fea(i, :), JK_Fea(j, :)];
        
        end
        
    end
    
end

save('/media/zhaojian/6TB/MSLowShotC2/Features/En_TestFea', 'En_TestFea', '-v7.3');