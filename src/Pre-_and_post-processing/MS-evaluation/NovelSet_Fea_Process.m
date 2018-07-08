% NovelSet Feature process script
clc
clear all;

features = zeros(1000, 2048);
folder=fullfile('/media/zhaojian/6TB/MSLowShotC2/Features/NovelSet_1_randomCropIdentityFeatureMean/');
folder_dirOutput=dir(fullfile(folder));
fileNames={folder_dirOutput.name}';

for i=3:length(fileNames)
    
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/media/zhaojian/6TB/MSLowShotC2/Features/NovelSet_1_randomCropIdentityFeatureMean/', file));
        
    features(i-2,:) = tmp.identityFeature; 
    
end

% Generate feature via mean encoding for each subject
Fea_Sub = zeros(1000, 2048);
MID_NOVEL = cell(1000, 1);

for i=1:1000
    
    Fea_Sub(i, :) = features(i, :);
    MID_NOVEL(i, 1) = fileNames(i+2);
    
end

save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_Fea_Sub_NovelSet_1', 'Fea_Sub', '-v7.3');
save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_MID_NOVEL', 'MID_NOVEL', '-v7.3');