% Script for model merge
clc
clear all;

Dense_MID = load('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_MID_DEV.mat');
Dense_MID = Dense_MID.MID_DEV;
JK_MID = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_MID_DEV.mat');
JK_MID = JK_MID.MID_DEV;

Dense_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_DevFea.mat');
Dense_Fea = Dense_Fea.DevFea;
JK_Fea = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_DevFea.mat');
JK_Fea = JK_Fea.DevFea;

En_DevFea = zeros(size(JK_Fea, 1), size(JK_Fea, 2) + size(Dense_Fea, 2));

for i=1:size(Dense_Fea, 1)
    
    disp(i);
    mid_tmp_1 = Dense_MID(i, :);
    
    for j=1:size(JK_Fea, 1)
        
        mid_tmp_2 = JK_MID(j, :);
        
        if strcmp(mid_tmp_1, mid_tmp_2)
        
            En_DevFea(i, :) = [Dense_Fea(i, :), JK_Fea(j, :)];
        
        end
        
    end
    
end

En_MID_DEV = Dense_MID;

save('/media/zhaojian/6TB/MSLowShotC2/Features/En_DevFea', 'En_DevFea', '-v7.3');
save('/media/zhaojian/6TB/MSLowShotC2/Features/En_MID_DEV', 'En_MID_DEV', '-v7.3');