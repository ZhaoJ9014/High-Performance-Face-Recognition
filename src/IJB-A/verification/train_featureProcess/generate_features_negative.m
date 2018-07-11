
clc
clear all;

load 'IJBA_VGG_AUG_split2_train.txt';

load 'TEMPLATE_ID';
load 'MEDIA_ID';
load 'SUBJECT_ID';

train_features = IJBA_VGG_AUG_split2_train(1:16354, :);
% find unique template id and media id
unique_template_id = unique(TEMPLATE_ID);
unique_media_id = unique(MEDIA_ID);

%generate features based on media
features_negative = zeros(size(unique_media_id, 1), 4096);
subject_negative = zeros(size(unique_media_id, 1), 1);

for i = 1:length(unique_media_id)
    index_media = find(MEDIA_ID == unique_media_id(i));
    if (size(index_media, 1) == 1) % img
        features_negative(i, :) = train_features(index_media, :);
    else % video
        features_negative(i, :) = sum(train_features(index_media, :)) / size(index_media, 1);
    end
    subject_negative(i, 1) = SUBJECT_ID(index_media(1), 1);
end

Template_Media = cell(size(unique_template_id, 1),1);
for i = 1:size(unique_template_id, 1)
    index_template = find(TEMPLATE_ID == unique_template_id(i));
    Template_Media{i} = unique(MEDIA_ID(index_template,:));
end

save('features_negative', 'features_negative');
save('Template_Media', 'Template_Media');
save('unique_media_id', 'unique_media_id');
save('unique_template_id', 'unique_template_id');
save('subject_negative', 'subject_negative');