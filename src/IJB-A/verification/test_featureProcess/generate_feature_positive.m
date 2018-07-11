
clc
clear all;

load 'IJBA_VGG_AUG_split2_test.mat';

load 'TEMPLATE_ID.mat';
load 'MEDIA_ID.mat';
load 'SUBJECT_ID.mat';

% test_features = model60000_fc7_feas_IJBA_split2_TEST(1:18715,:);
% find unique template id and media id
unique_template_id = unique(TEMPLATE_ID);
unique_media_id = unique(MEDIA_ID);

% generate features based on media
features_positive = zeros(size(unique_media_id, 1), 4096);
subject_positive = zeros(size(unique_media_id, 1), 1);
features_test = zeros(size(unique_template_id, 1), 4096);

%for i = 1:size(unique_template_id, 1)
%    index_template = find(TEMPLATE_ID == unique_template_id(i)); % Find media index
%    if (size(index_template, 1) == 1)
%        features_test(i, :) = test_features(index_template, :);
%    else
%        features_test(i, :) = sum(test_features(index_template, :)) / size(index_template, 1);
%    end
%end

for i = 1:length(unique_media_id)
    index_media = find(MEDIA_ID == unique_media_id(i));
    if (size(index_media, 1) == 1) % img
        features_positive(i, :) = IJBA_VGG_AUG_split2_test(index_media, :);
    else % video
        features_positive(i, :) = sum(IJBA_VGG_AUG_split2_test(index_media, :)) / size(index_media, 1);
    end
    subject_positive(i, 1) = SUBJECT_ID(index_media(1), 1);
end

Template_Media = cell(size(unique_template_id, 1),1);
for i = 1:size(unique_template_id, 1)
    index_template = find(TEMPLATE_ID == unique_template_id(i));
    Template_Media{i} = unique(MEDIA_ID(index_template,:));
end

save('features_positive', 'features_positive');
save('Template_Media', 'Template_Media');
save('unique_media_id', 'unique_media_id');
save('unique_template_id', 'unique_template_id');
%save('features_test', 'features_test');
save('subject_positive','subject_positive');