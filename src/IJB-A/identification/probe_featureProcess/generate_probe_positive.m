
clc
clear all;

load 'IJBA_VGG_AUG_split2_probe.txt';
load 'TEMPLATE_ID_PROBE';
load 'MEDIA_ID_PROBE';
load 'SUBJECT_ID';

test_features = IJBA_VGG_AUG_split2_probe(1:13983,:);

% find unique template id and media id
unique_template_id_probe = unique(TEMPLATE_ID);
unique_media_id_probe = unique(MEDIA_ID);

%generate features based on media
features_probe = zeros(size(unique_media_id_probe, 1), 4096);

for i = 1:length(unique_media_id_probe)
    index_media = find(MEDIA_ID == unique_media_id_probe(i));
    if (size(index_media, 1) == 1) % img
        features_probe(i, :) = test_features(index_media, :);
    else % video
        features_probe(i, :) = sum(test_features(index_media, :)) / size(index_media, 1);
    end
end

Template_Media_PROBE = cell(size(unique_template_id_probe, 1),1);
for i = 1:size(unique_template_id_probe, 1)
    index_template = find(TEMPLATE_ID == unique_template_id_probe(i));
    Template_Media_PROBE{i} = unique(MEDIA_ID(index_template,:));
    template_subject_probe(i) = SUBJECT_ID(index_template(1)); 
end

save('features_probe', 'features_probe');
save('template_subject_probe', 'template_subject_probe');
save('Template_Media_PROBE', 'Template_Media_PROBE');
save('unique_media_id_probe', 'unique_media_id_probe');
save('unique_template_id_probe', 'unique_template_id_probe');