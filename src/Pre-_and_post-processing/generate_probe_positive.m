
clc
clear all;

load 'model80000_CleanData_CENTERLOSS_IJBA_split1_probe.txt';
%load 'orig_model60000_fc7_CleanData_IJBA_split2_probe.txt';
load 'IJBA_GoogleNet_loss1fc_split1_probe.txt'
load 'IJBA_GoogleNet_loss2fc_split1_probe.txt'
load 'IJBA_GoogleNet_pool5_split1_probe.txt'

load 'probe_TEMPLATE_ID.txt';
load 'probe_MEDIA_ID.txt';
load 'probe_SUBJECT_ID.txt';

test_features = [model80000_CleanData_CENTERLOSS_IJBA_split1_probe(1:13726, :), IJBA_GoogleNet_loss1fc_split1_probe(1:13726,:), IJBA_GoogleNet_loss2fc_split1_probe(1:13726,:), IJBA_GoogleNet_pool5_split1_probe(1:13726, :)];
%test_features = model80000_CleanData_CENTERLOSS_IJBA_split1_probe(1:13726, :);
% find unique template id and media id
unique_template_id_probe = unique(probe_TEMPLATE_ID);
unique_media_id_probe = unique(probe_MEDIA_ID);

%generate features based on media
features_probe = zeros(size(unique_media_id_probe, 1), 10240);

for i = 1:length(unique_media_id_probe)
    index_media = find(probe_MEDIA_ID == unique_media_id_probe(i));
    if (size(index_media, 1) == 1) % img
        features_probe(i, :) = test_features(index_media, :);
    else % video
        features_probe(i, :) = sum(test_features(index_media, :)) / size(index_media, 1);
    end
end

Template_Media_PROBE = cell(size(unique_template_id_probe, 1),1);
for i = 1:size(unique_template_id_probe, 1)
    index_template = find(probe_TEMPLATE_ID == unique_template_id_probe(i));
    Template_Media_PROBE{i} = unique(probe_MEDIA_ID(index_template,:));
    template_subject_probe(i) = probe_SUBJECT_ID(index_template(1)); 
end

save('features_probe', 'features_probe');
save('template_subject_probe', 'template_subject_probe');
save('Template_Media_PROBE', 'Template_Media_PROBE');
save('unique_media_id_probe', 'unique_media_id_probe');
save('unique_template_id_probe', 'unique_template_id_probe');