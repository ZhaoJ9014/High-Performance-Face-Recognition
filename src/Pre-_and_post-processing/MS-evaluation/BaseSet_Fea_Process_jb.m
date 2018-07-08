% BaseSet Feature process for jb script 
clc
clear all;

folder=fullfile('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/');
folder_dirOutput=dir(fullfile(folder));
fileNames={folder_dirOutput.name}';

features_b1 = [];
labels_b1 = [];
for i=3:502
    
    disp('Batch 1 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b1 = [features_b1; tmp.identityFeature];
    labels_b1 = [labels_b1; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b2 = [];
labels_b2 = [];
for i=503:1002
    
    disp('Batch 2 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b2 = [features_b2; tmp.identityFeature];
    labels_b2 = [labels_b2; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b3 = [];
labels_b3 = [];
for i=1003:1502
    
    disp('Batch 3 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b3 = [features_b3; tmp.identityFeature];
    labels_b3 = [labels_b3; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b4 = [];
labels_b4 = [];
for i=1503:2002
    
    disp('Batch 4 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b4 = [features_b4; tmp.identityFeature];
    labels_b4 = [labels_b4; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b5 = [];
labels_b5 = [];
for i=2003:2502
    
    disp('Batch 5 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b5 = [features_b5; tmp.identityFeature];
    labels_b5 = [labels_b5; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b6 = [];
labels_b6 = [];
for i=2503:3002
    
    disp('Batch 6 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b6 = [features_b6; tmp.identityFeature];
    labels_b6 = [labels_b6; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b7 = [];
labels_b7 = [];
for i=3003:3502
    
    disp('Batch 7 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b7 = [features_b7; tmp.identityFeature];
    labels_b7 = [labels_b7; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b8 = [];
labels_b8 = [];
for i=3503:4002
    
    disp('Batch 8 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b8 = [features_b8; tmp.identityFeature];
    labels_b8 = [labels_b8; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b9 = [];
labels_b9 = [];
for i=4003:4502
    
    disp('Batch 9 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b9 = [features_b9; tmp.identityFeature];
    labels_b9 = [labels_b9; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b10 = [];
labels_b10 = [];
for i=4503:5002
    
    disp('Batch 10 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b10 = [features_b10; tmp.identityFeature];
    labels_b10 = [labels_b10; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b11 = [];
labels_b11 = [];
for i=5003:5502
    
    disp('Batch 11 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b11 = [features_b11; tmp.identityFeature];
    labels_b11 = [labels_b11; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b12 = [];
labels_b12 = [];
for i=5503:6002
    
    disp('Batch 12 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b12 = [features_b12; tmp.identityFeature];
    labels_b12 = [labels_b12; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b13 = [];
labels_b13 = [];
for i=6003:6502
    
    disp('Batch 13 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b13 = [features_b13; tmp.identityFeature];
    labels_b13 = [labels_b13; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b14 = [];
labels_b14 = [];
for i=6503:7002
    
    disp('Batch 14 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b14 = [features_b14; tmp.identityFeature];
    labels_b14 = [labels_b14; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b15 = [];
labels_b15 = [];
for i=7003:7502
    
    disp('Batch 15...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b15 = [features_b15; tmp.identityFeature];
    labels_b15 = [labels_b15; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b16 = [];
labels_b16 = [];
for i=7503:8002
    
    disp('Batch 16 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b16 = [features_b16; tmp.identityFeature];
    labels_b16 = [labels_b16; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b17 = [];
labels_b17 = [];
for i=8003:8502
    
    disp('Batch 17 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b17 = [features_b17; tmp.identityFeature];
    labels_b17 = [labels_b17; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b18 = [];
labels_b18 = [];
for i=8503:9002
    
    disp('Batch 18 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b18 = [features_b18; tmp.identityFeature];
    labels_b18 = [labels_b18; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b19 = [];
labels_b19 = [];
for i=9003:9502
    
    disp('Batch 19 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b19 = [features_b19; tmp.identityFeature];
    labels_b19 = [labels_b19; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b20 = [];
labels_b20 = [];
for i=9503:10002
    
    disp('Batch 20 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b20 = [features_b20; tmp.identityFeature];
    labels_b20 = [labels_b20; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b21 = [];
labels_b21 = [];
for i=10003:10502
    
    disp('Batch 21 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b21 = [features_b21; tmp.identityFeature];
    labels_b21 = [labels_b21; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b22 = [];
labels_b22 = [];
for i=10503:11002
    
    disp('Batch 22 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b22 = [features_b22; tmp.identityFeature];
    labels_b22 = [labels_b22; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b23 = [];
labels_b23 = [];
for i=11003:11502
    
    disp('Batch 23 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b23 = [features_b23; tmp.identityFeature];
    labels_b23 = [labels_b23; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b24 = [];
labels_b24 = [];
for i=11503:12002
    
    disp('Batch 24 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b24 = [features_b24; tmp.identityFeature];
    labels_b24 = [labels_b24; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b25 = [];
labels_b25 = [];
for i=12003:12502
    
    disp('Batch 25 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b25 = [features_b25; tmp.identityFeature];
    labels_b25 = [labels_b25; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b26 = [];
labels_b26 = [];
for i=12503:13002
    
    disp('Batch 26 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b26 = [features_b26; tmp.identityFeature];
    labels_b26 = [labels_b26; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b27 = [];
labels_b27 = [];
for i=13003:13502
    
    disp('Batch 27 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b27 = [features_b27; tmp.identityFeature];
    labels_b27 = [labels_b27; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b28 = [];
labels_b28 = [];
for i=13503:14002
    
    disp('Batch 28 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b28 = [features_b28; tmp.identityFeature];
    labels_b28 = [labels_b28; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b29 = [];
labels_b29 = [];
for i=14003:14502
    
    disp('Batch 29 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b29 = [features_b29; tmp.identityFeature];
    labels_b29 = [labels_b29; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b30 = [];
labels_b30 = [];
for i=14503:15002
    
    disp('Batch 30 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b30 = [features_b30; tmp.identityFeature];
    labels_b30 = [labels_b30; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b31 = [];
labels_b31 = [];
for i=15003:15502
    
    disp('Batch 31 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b31 = [features_b31; tmp.identityFeature];
    labels_b31 = [labels_b31; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b32 = [];
labels_b32 = [];
for i=15503:16002
    
    disp('Batch 32 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b32 = [features_b32; tmp.identityFeature];
    labels_b32 = [labels_b32; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b33 = [];
labels_b33 = [];
for i=16003:16502
    
    disp('Batch 33 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b33 = [features_b33; tmp.identityFeature];
    labels_b33 = [labels_b33; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b34 = [];
labels_b34 = [];
for i=16503:17002
    
    disp('Batch 34 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b34 = [features_b34; tmp.identityFeature];
    labels_b34 = [labels_b34; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b35 = [];
labels_b35 = [];
for i=17003:17502
    
    disp('Batch 35...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b35 = [features_b35; tmp.identityFeature];
    labels_b35 = [labels_b35; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b36 = [];
labels_b36 = [];
for i=17503:18002
    
    disp('Batch 36 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b36 = [features_b36; tmp.identityFeature];
    labels_b36 = [labels_b36; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b37 = [];
labels_b37 = [];
for i=18003:18502
    
    disp('Batch 37 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b37 = [features_b37; tmp.identityFeature];
    labels_b37 = [labels_b37; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b38 = [];
labels_b38 = [];
for i=18503:19002
    
    disp('Batch 38 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b38 = [features_b38; tmp.identityFeature];
    labels_b38 = [labels_b38; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b39 = [];
labels_b39 = [];
for i=19003:19502
    
    disp('Batch 39 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b39 = [features_b39; tmp.identityFeature];
    labels_b39 = [labels_b39; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end
features_b40 = [];
labels_b40 = [];
for i=19503:20002
    
    disp('Batch 40 ...');
    disp(i-3);

    file = fileNames{i};
    tmp = load(strcat('/Users/zhaojian/Desktop/MSLowShotC2/Features/BaseSetIdentityFeature/', file));
    features_b40 = [features_b40; tmp.identityFeature];
    labels_b40 = [labels_b40; repmat(i-3, size(tmp.identityFeature, 1), 1)];
    
end

disp('Saving ...');
features = [features_b1; features_b2; features_b3; features_b4; features_b5; features_b6; features_b7; features_b8; features_b9; features_b10; features_b11; features_b12; features_b13; features_b14; features_b15; features_b16; features_b17; features_b18; features_b19; features_b20; features_b21; features_b22; features_b23; features_b24; features_b25; features_b26; features_b27; features_b28; features_b29; features_b30; features_b31; features_b32; features_b33; features_b34; features_b35; features_b36; features_b37; features_b38; features_b39; features_b40];
labels = [labels_b1; labels_b2; labels_b3; labels_b4; labels_b5; labels_b6; labels_b7; labels_b8; labels_b9; labels_b10; labels_b11; labels_b12; labels_b13; labels_b14; labels_b15; labels_b16; labels_b17; labels_b18; labels_b19; labels_b20; labels_b21; labels_b22; labels_b23; labels_b24; labels_b25; labels_b26; labels_b27; labels_b28; labels_b29; labels_b30; labels_b31; labels_b32; labels_b33; labels_b34; labels_b35; labels_b36; labels_b37; labels_b38; labels_b39; labels_b40];

save('/Users/zhaojian/Desktop/MSLowShotC2/Features/Fea_train_jb', 'features', '-v7.3');
save('/Users/zhaojian/Desktop/MSLowShotC2/Features/Label_train_jb', 'labels', '-v7.3');