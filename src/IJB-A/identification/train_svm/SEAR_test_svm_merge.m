clc
clear all

load 'unique_template_id_probe'
load 'unique_template_id_gallery'
load 'unique_media_id_probe'
load 'unique_media_id_gallery'
load 'features_probe'
load 'features_gallery'
load 'template_subject_probe'
load 'template_subject_gallery'
load 'Template_Media_PROBE'
load 'Template_Media_GALLERY'

svm_scaled_features_probe=svm_scale(features_probe);
svm_scaled_features_gallery=svm_scale(features_gallery);
beta =10;

for i=1:length(unique_template_id_probe)
    pair_1_id = unique_template_id_probe(i);
    pair_1_model_name = strcat('/home/zhaojian/liblinear-2.1/matlab/probe_models/IJBA_split1_svm_model_', num2str(pair_1_id), '.mat');
    pair_1_svm = load(pair_1_model_name);   
    pair1_m = Template_Media_PROBE{i};
    for j = 1:length(unique_template_id_gallery)
        pair_2_id = unique_template_id_gallery(j);
        pair_2_model_name = strcat('/home/zhaojian/liblinear-2.1/matlab/gallery_models/IJBA_split1_svm_model_', num2str(pair_2_id), '.mat');
        pair_2_svm = load(pair_2_model_name);
        label = double(1);
        pair2_m = Template_Media_GALLERY{j};
        
        Sum1 = 0;
        Sum2 = 0;
        res_buf = 0;
        for p1 = 1:size(pair1_m)
           pair_1_i = find(unique_media_id_probe == pair1_m(p1));
	       pair_1_features = sparse(svm_scaled_features_probe(pair_1_i,:)); % find the probe and reference features
           [predict_label_1_tmp,accuracy_1_tmp,dec_values_1_tmp]=predict(label,pair_1_features,pair_2_svm.model); % predict
           for p2 = 1:size(pair2_m)
              pair_2_i = find(unique_media_id_gallery == pair2_m(p2));
	          pair_2_features = sparse(svm_scaled_features_gallery(pair_2_i,:));
	          [predict_label_2_tmp,accuracy_2_tmp,dec_values_2_tmp]=predict(label,pair_2_features,pair_1_svm.model);
              res_buf = 0.5*dec_values_1_tmp+0.5*dec_values_2_tmp; 
              Sum1 = Sum1 + res_buf * exp(res_buf*beta);
              Sum2 = Sum2 + exp(res_buf*beta);
           end
        end
        S_PQ(j,i) = Sum1/Sum2;
    end
end
save('S_PQ_SVM','S_PQ')
