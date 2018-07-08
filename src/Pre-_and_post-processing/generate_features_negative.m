
clc
clear all;

%load 'IJBA_VGG_AUG_split2_train.txt';
%load 'orig_model60000_fc7_CleanData_IJBA_split2_train.txt'
load 'model80000_CleanData_CENTERLOSS_IJBA_split1_train.txt'
load 'IJBA_GoogleNet_loss1fc_split1_train.txt'
load 'IJBA_GoogleNet_loss2fc_split1_train.txt'
load 'IJBA_GoogleNet_pool5_split1_train.txt'

load 'train_TEMPLATE_ID.txt';
load 'train_MEDIA_ID.txt';
load 'train_SUBJECT_ID.txt';

%train_features = model80000_CleanData_CENTERLOSS_IJBA_split1_train(1:16889,:);
train_features = [model80000_CleanData_CENTERLOSS_IJBA_split1_train(1:16889, :), IJBA_GoogleNet_loss1fc_split1_train(1:16889, :), IJBA_GoogleNet_loss2fc_split1_train(1:16889, :), IJBA_GoogleNet_pool5_split1_train(1:16889, :)];
% find unique template id and media id
unique_template_id = unique(train_TEMPLATE_ID);
unique_media_id = unique(train_MEDIA_ID);

%generate features based on media
features_negative = zeros(size(unique_media_id, 1), 10240);
subject_negative = zeros(size(unique_media_id, 1), 1);



for i = 1:length(unique_media_id)
    index_media = find(train_MEDIA_ID == unique_media_id(i));
    if (size(index_media, 1) == 1) % img
        features_negative(i, :) = train_features(index_media, :);
    else % video
        features_negative(i, :) = sum(train_features(index_media, :)) / size(index_media, 1);
    end
    subject_negative(i, 1) = train_SUBJECT_ID(index_media(1), 1);
end

Template_Media = cell(size(unique_template_id, 1),1);
for i = 1:size(unique_template_id, 1)
    index_template = find(train_TEMPLATE_ID == unique_template_id(i));
    Template_Media{i} = unique(train_MEDIA_ID(index_template,:));
end

save('features_negative', 'features_negative');
save('Template_Media', 'Template_Media');
save('unique_media_id', 'unique_media_id');
save('unique_template_id', 'unique_template_id');
save('subject_negative', 'subject_negative');