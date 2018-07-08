% BaseSet Feature process script
clc
clear all;

features = zeros(20000, 2048);
folder=fullfile('/media/zhaojian/6TB/MSLowShotC2/Features/BaseSetIdentityFeature/');
folder_dirOutput=dir(fullfile(folder));
fileNames={folder_dirOutput.name}';

for i=3:length(fileNames)
    
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/media/zhaojian/6TB/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    
    if(size(tmp.identityFeature, 1) > 1)
        
        features(i-2,:) = sum(tmp.identityFeature) / size(tmp.identityFeature, 1); 
    
    else
        
        features(i-2,:) = tmp.identityFeature;
    
    end
    
end

% Generate feature via mean encoding for each subject
Fea_Sub = zeros(20000, 2048);
MID_BASE = cell(20000, 1);

for i=1:20000
    
    Fea_Sub(i, :) = features(i, :);
    MID_BASE(i, 1) = fileNames(i+2);
    
end

save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_Fea_Sub_BaseSet', 'Fea_Sub', '-v7.3');
save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_MID_BASE', 'MID_BASE', '-v7.3');