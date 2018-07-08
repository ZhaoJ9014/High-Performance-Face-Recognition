% Dev Feature process script
% DevBaseFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/DevBaseSet_feature_batch/DevBaseSet_Feature.txt');
% DevNovelFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/DevNovelSet_feature_batch/DevNovelSet_Feature.txt');
% DevFea = [DevBaseFea; DevNovelFea];
TestFea = load('/media/zhaojian/6TB/MSLowShotC2/Features/challenge2_Feature.txt');

% MID_DEV = [DevBaseSetLabelList; DevNovelSetLabelList];

save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_TestFea', 'TestFea');
% save('/media/zhaojian/6TB/MSLowShotC2/Features/Dense_MID_DEV', 'MID_DEV');