%% Test IJBA split1 on verification task
clc
clear all;

load 'PROBE'
load 'REFERENCE'
load 'unique_template_id'
load 'unique_media_id'
load 'Template_Media'
load 'features_positive'
load 'pair_labels'

svm_scaled_features=svm_scale(features_positive);

 beta =10;

for i=1:size(PROBE,1)
    Sum1 = 0;
    Sum2 = 0;
	pair_1=PROBE(i); % template_id for each pair
	pair_2=REFERENCE(i);
       
	pair_1_index=find(unique_template_id==pair_1); % find the relative template_id
	pair_2_index=find(unique_template_id==pair_2);
    
    pair1_m = Template_Media{pair_1_index}; %find the features based on media for a template
    pair2_m = Template_Media{pair_2_index};
    
    pair_1_model_name = strcat('/home/zhaojian/liblinear-2.1/matlab/models/IJBA_split2_svm_model_', num2str(pair_1), '.mat'); % Load svm
    pair_2_model_name = strcat('/home/zhaojian/liblinear-2.1/matlab/models/IJBA_split2_svm_model_', num2str(pair_2), '.mat');
 	pair_1_svm = load(pair_1_model_name);
	pair_2_svm = load(pair_2_model_name);
    label=pair_labels(i,1); % find the pair label
    
    for p1 = 1:size(pair1_m)
        pair_1_i = find(unique_media_id == pair1_m(p1));
	    pair_1_features = sparse(svm_scaled_features(pair_1_i,:)); % find the probe and reference features
       [predict_label_1_tmp,accuracy_1_tmp,dec_values_1_tmp]=predict(label,pair_1_features,pair_2_svm.model); % predict
        for p2 = 1:size(pair2_m)
            pair_2_i = find(unique_media_id == pair2_m(p2));
	        pair_2_features = sparse(svm_scaled_features(pair_2_i,:));     
	        [predict_label_2_tmp,accuracy_2_tmp,dec_values_2_tmp]=predict(label,pair_2_features,pair_1_svm.model);
            S(p1,p2) = 0.5*dec_values_1_tmp+0.5*dec_values_2_tmp; 
            Sum1 = Sum1 + S(p1,p2) * exp(S(p1,p2)*beta);
            Sum2 = Sum2 + exp(S(p1,p2)*beta);
        end
    end
    S_PQ(i) = Sum1/Sum2;
end

save('S_PQ_SVM','S_PQ');

% auc = plot_roc(S_PQ,pair_labels)
