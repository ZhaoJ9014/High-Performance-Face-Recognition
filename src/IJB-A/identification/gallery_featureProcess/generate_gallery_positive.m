
clc
clear all

load 'IJBA_VGG_AUG_split2_gallery.txt';
load 'TEMPLATE_ID_GALLERY';
load 'MEDIA_ID_GALLERY';
load 'SUBJECT_ID';

test_features = IJBA_VGG_AUG_split2_gallery(1:3261,:);
% find unique template id and media id
unique_template_id_gallery = unique(TEMPLATE_ID);
unique_media_id_gallery = unique(MEDIA_ID);

%generate features based on media
features_gallery = zeros(size(unique_media_id_gallery, 1), 4096);

for i = 1:length(unique_media_id_gallery)
    index_media = find(MEDIA_ID == unique_media_id_gallery(i));
    if (size(index_media, 1) == 1) % img
        features_gallery(i, :) = test_features(index_media, :);
    else % video
        features_gallery(i, :) = sum(test_features(index_media, :)) / size(index_media, 1);
    end 
end

Template_Media_GALLERY = cell(size(unique_template_id_gallery, 1),1);
for i = 1:size(unique_template_id_gallery, 1)
    index_template = find(TEMPLATE_ID == unique_template_id_gallery(i));
    Template_Media_GALLERY{i} = unique(MEDIA_ID(index_template,:));
    template_subject_gallery(i) = SUBJECT_ID(index_template(1)); 
end

save('features_gallery', 'features_gallery');
save('template_subject_gallery', 'template_subject_gallery');
save('Template_Media_GALLERY', 'Template_Media_GALLERY');
save('unique_media_id_gallery', 'unique_media_id_gallery');
save('unique_template_id_gallery', 'unique_template_id_gallery');