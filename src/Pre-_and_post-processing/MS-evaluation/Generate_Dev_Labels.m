% Script for generating Dev Set labels for jb
MID_DEV = load('/Users/zhaojian/Desktop/MSLowShotC2/Features/MID_DEV.mat');
MID_DEV = MID_DEV.MID_DEV;
unique_MID_DEV = unique(MID_DEV);
labels = zeros(length(MID_DEV), 1);

for i =1:length(MID_DEV)
    
    disp(i);
    id = MID_DEV{i, 1};
    
    for j=1:length(unique_MID_DEV)
    
        if strcmp(id, unique_MID_DEV{j, 1})
            
            labels(i, 1) = j - 1;
        
        end
    
    end

end

save('/Users/zhaojian/Desktop/MSLowShotC2/Features/Label_DEV_jb', 'labels');