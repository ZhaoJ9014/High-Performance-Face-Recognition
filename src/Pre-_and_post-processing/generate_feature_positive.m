
clc
clear all;

%load 'IJBA_VGG_AUG_split2_test.mat';
%load 'orig_model60000_fc7_CleanData_IJBA_split2_test.txt'
load 'model80000_CleanData_CENTERLOSS_IJBA_split1_test.txt'
load 'IJBA_GoogleNet_loss1fc_split1_test.txt'
load 'IJBA_GoogleNet_loss2fc_split1_test.txt'
load 'IJBA_GoogleNet_pool5_split1_test.txt'


load 'test_TEMPLATE_ID.txt';
load 'test_MEDIA_ID.txt';
load 'test_SUBJECT_ID.txt';

%test_features = model80000_CleanData_CENTERLOSS_IJBA_split1_test(1:17985, :);
test_features=[model80000_CleanData_CENTERLOSS_IJBA_split1_test(1:17985, :), IJBA_GoogleNet_loss1fc_split1_test(1:17985, :), IJBA_GoogleNet_loss2fc_split1_test(1:17985, :), IJBA_GoogleNet_pool5_split1_test(1:17985, :)];
%model60000_fc7_feas_IJBA_split2_TEST = model60000_fc7_feas_IJBA_split2_TEST(1:18715,:);
%test_features = [model60000_fc7_feas_IJBA_split2_TEST, IJBA_VGG_AUG_split2_test];
% find unique template id and media id
unique_template_id = unique(test_TEMPLATE_ID);
unique_media_id = unique(test_MEDIA_ID);

% generate features based on media
features_positive = zeros(size(unique_media_id, 1), 10240);
subject_positive = zeros(size(unique_media_id, 1), 1);
features_test = zeros(size(unique_template_id, 1), 10240);

%for i = 1:size(unique_template_id, 1)
%    index_template = find(TEMPLATE_ID == unique_template_id(i)); % Find media index
%    if (size(index_template, 1) == 1)
%        features_test(i, :) = test_features(index_template, :);
%    else
%        features_test(i, :) = sum(test_features(index_template, :)) / size(index_template, 1);
%    end
%end

for i = 1:length(unique_media_id)
    index_media = find(test_MEDIA_ID == unique_media_id(i));
    if (size(index_media, 1) == 1) % img
        features_positive(i, :) = test_features(index_media, :);
    else % video
        features_positive(i, :) = sum(test_features(index_media, :)) / size(index_media, 1);
    end
    subject_positive(i, 1) = test_SUBJECT_ID(index_media(1), 1);
end

Template_Media = cell(size(unique_template_id, 1),1);
for i = 1:size(unique_template_id, 1)
    index_template = find(test_TEMPLATE_ID == unique_template_id(i));
    Template_Media{i} = unique(test_MEDIA_ID(index_template,:));
end

save('features_positive', 'features_positive');
save('Template_Media', 'Template_Media');
save('unique_media_id', 'unique_media_id');
save('unique_template_id', 'unique_template_id');
%save('features_test', 'features_test');
save('subject_positive','subject_positive');