%% Train SVM for IJB-A template adaptation
clc
clear all;
% Load positive & negative features

load './split1/features_negative_VGG_s';
load './split1/features_positive_VGG_s';
load './split1/unique_template_id';
load './split1/unique_media_id';
load './split1/TEMPLATE_ID';
load './split1/MEDIA_ID';

svm_features_negative=svm_scale(features_negative_VGG_s);
svm_features_positive=svm_scale(features_positive_VGG_s);

for i=1:size(unique_template_id,1)
    index_template = find(TEMPLATE_ID == unique_template_id(i));
    media_all = unique(MEDIA_ID(index_template));
    index = [];
    for j = 1:length(media_all)
        dex = find(unique_media_id == media_all(j));
        index = [index;dex];
    end

	positive_samples=svm_features_positive(index,:);
	negative_samples=svm_features_negative;
	features=[positive_samples;negative_samples];
        %labels=[1*ones(size(positive_samples,1),1);-1*ones(size(negative_samples,1),1)]; % Update labels
        labels=[1*ones(size(positive_samples,1),1);zeros(size(negative_samples,1),1)]; % Update labels

        Np=size(positive_samples,1); % Number of possitives
        Nn=size(negative_samples,1); % Number of negatives
	    C=1;
        Ep=(Np+Nn)/(2.0*Np); % Coefficient of possitive constraint iterm
        En=(Np+Nn)/(2.0*Nn); % Coefficient of negative constraint iterm

	sparsed_features=sparse(features);
    %model = svmtrain(sparsed_features,labels);
	model=train(labels, sparsed_features, sprintf('-s 2 -c %f -w1 %f -w-1 %f', C, Ep, En)); % Train 2-class SVM
	model_name=strcat('.\SVMmodels_split1_fc7_s\IJBA_split1_svm_model_', num2str(unique_template_id(i)));
	save(model_name,'model');
end
%save ('PCA_model', 'mean_fea','eigvector');