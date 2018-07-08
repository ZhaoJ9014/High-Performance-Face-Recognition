%Train SVM for IJB-A template adaptation
clc
clear all;

%%Load positive & negative features
load 'features_probe';
load 'features_negative';
load 'unique_template_id_probe';
load 'unique_media_id_probe';
load 'TEMPLATE_ID_PROBE';
load 'MEDIA_ID_PROBE';

svm_features_positive=svm_scale(features_probe);
svm_features_negative=svm_scale(features_negative);

for i=1:size(unique_template_id_probe,1)
    index_template = find(TEMPLATE_ID == unique_template_id_probe(i));
    media_all = unique(MEDIA_ID(index_template));
    index = [];
    for j = 1:length(media_all)
        dex = find(unique_media_id_probe == media_all(j));
        index = [index;dex];
    end

	positive_samples=svm_features_positive(index,:);
	negative_samples=svm_features_negative;
	features=[positive_samples;negative_samples];
        labels=[1*ones(size(positive_samples,1),1);-1*ones(size(negative_samples,1),1)]; % Update labels
        Np=size(positive_samples,1); % Number of possitives
        Nn=size(negative_samples,1); % Number of negatives
	    C=1;
        Ep=(Np+Nn)/(2.0*Np); % Coefficient of possitive constraint iterm
        En=(Np+Nn)/(2.0*Nn); % Coefficient of negative constraint iterm

	sparsed_features=sparse(features);
	model=train(labels, sparsed_features, sprintf('-s 2 -c %f -w1 %f -w-1 %f', C, Ep, En)); % Train 2-class SVM
	model_name=strcat('/home/zhaojian/liblinear-2.1/matlab/sear_models/IJBA_split2_svm_model_', num2str(unique_template_id_probe(i)));
	save(model_name,'model');/home/zhaojian/liblinear-2.1/matlab/sear_models/
end


% Load positive & negative features
load 'features_gallery';

load 'unique_template_id_gallery';
load 'unique_media_id_gallery';
load 'TEMPLATE_ID_GALLERY';
load 'MEDIA_ID_GALLERY';

svm_features=svm_scale(features_gallery);

for i=1:size(unique_template_id_gallery,1)
    index_template = find(TEMPLATE_ID == unique_template_id_gallery(i));
    media_all = unique(MEDIA_ID(index_template));
    index = [];
    for j = 1:length(media_all)
        dex = find(unique_media_id_gallery == media_all(j));
        index = [index;dex];
    end

        labels= -1*ones(size(svm_features,1),1); % Update labels
        labels(index) = 1; 
        Np=size(index,1); % Number of possitives
        Nn=size(svm_features,1)-Np; % Number of negatives
	    C=1;
        Ep=(Np+Nn)/(2.0*Np); % Coefficient of possitive constraint iterm
        En=(Np+Nn)/(2.0*Nn); % Coefficient of negative constraint iterm

	sparsed_features=sparse(svm_features);
	model=train(labels, sparsed_features, sprintf('-s 2 -c %f -w1 %f -w-1 %f', C, Ep, En)); % Train 2-class SVM
	model_name=strcat('/home/zhaojian/liblinear-2.1/matlab/sear_models/IJBA_split1_svm_model_', num2str(unique_template_id_gallery(i)));
	save(model_name,'model');
end

