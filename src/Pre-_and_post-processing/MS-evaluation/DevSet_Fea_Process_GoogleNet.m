% Dev Feature process script
% DevBaseFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_GoogleNet_BN_Dev/JK_GoogleNet_BN_Dev_BaseSet_pool5.txt');
% DevNovelFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_GoogleNet_BN_Dev/JK_GoogleNet_BN_Dev_NovelSet_pool5.txt');
% DevNovelFea = DevNovelFea(1:5000, :);
% 
% DevFea = [DevBaseFea; DevNovelFea];
TestFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/JK_GoogleNet_BN_challenge2/JK_GoogleNet_BN_challenge2_pool5.txt');

% MID_DEV = [DevBaseSetLabelList; DevNovelSetLabelList];

save('/media/zhaojian/6TB/MSLowShotC2/Features/JK_TestFea', 'TestFea');
% save('/media/zhaojian/6TB/MSLowShotC2/Features/JK_MID_DEV', 'MID_DEV');