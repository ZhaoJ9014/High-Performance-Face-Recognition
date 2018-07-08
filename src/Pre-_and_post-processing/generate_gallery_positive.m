
clc
clear all

load 'model80000_CleanData_CENTERLOSS_IJBA_split1_gallery.txt'
%load 'orig_model60000_fc7_CleanData_IJBA_split2_gallery.txt';
load 'IJBA_GoogleNet_loss1fc_split1_gallery.txt'
load 'IJBA_GoogleNet_loss2fc_split1_gallery.txt'
load 'IJBA_GoogleNet_pool5_split1_gallery.txt'

load 'gallery_TEMPLATE_ID.txt';
load 'gallery_MEDIA_ID.txt';
load 'gallery_SUBJECT_ID.txt';

test_features = [model80000_CleanData_CENTERLOSS_IJBA_split1_gallery(1:3000, :), IJBA_GoogleNet_loss1fc_split1_gallery(1:3000,:), IJBA_GoogleNet_loss2fc_split1_gallery(1:3000,:), IJBA_GoogleNet_pool5_split1_gallery(1:3000, :)];
%test_features = model80000_CleanData_CENTERLOSS_IJBA_split1_gallery(1:3000,:);
% find unique template id and media id
unique_template_id_gallery = unique(gallery_TEMPLATE_ID);
unique_media_id_gallery = unique(gallery_MEDIA_ID);

%generate features based on media
features_gallery = zeros(size(unique_media_id_gallery, 1), 10240);

for i = 1:length(unique_media_id_gallery)
    index_media = find(gallery_MEDIA_ID == unique_media_id_gallery(i));
    if (size(index_media, 1) == 1) % img
        features_gallery(i, :) = test_features(index_media, :);
    else % video
        features_gallery(i, :) = sum(test_features(index_media, :)) / size(index_media, 1);
    end 
end

Template_Media_GALLERY = cell(size(unique_template_id_gallery, 1),1);
for i = 1:size(unique_template_id_gallery, 1)
    index_template = find(gallery_TEMPLATE_ID == unique_template_id_gallery(i));
    Template_Media_GALLERY{i} = unique(gallery_MEDIA_ID(index_template,:));
    template_subject_gallery(i) = gallery_SUBJECT_ID(index_template(1)); 
end

save('features_gallery', 'features_gallery');
save('template_subject_gallery', 'template_subject_gallery');
save('Template_Media_GALLERY', 'Template_Media_GALLERY');
save('unique_media_id_gallery', 'unique_media_id_gallery');
save('unique_template_id_gallery', 'unique_template_id_gallery');