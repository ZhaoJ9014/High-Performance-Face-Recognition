clc
clear all

load 'PROBE'
load 'REFERENCE'
load 'unique_template_id'
load 'unique_media_id'
load 'Template_Media'
load 'features_positive'
load 'pair_labels'

features_positive=svm_scale(features_positive);
%reduce dimentation using PCA
mean_fea = mean(features_positive,1);
features_positive = features_positive - repmat(mean_fea, size(features_positive,1), 1);
disp('start pca');
tic
options.ReducedDim = 1024;
[eigvector, S, eigvalue] = PCA(features_positive, options);       
epsilon = 0.1; 
% % % --------------------whitening-------------------- 
features_positive = features_positive * eigvector* diag(1./sqrt(diag(S) + epsilon));

%reduce dimentation using PCA
mean_fea = mean(features_positive,1);
features_positive = features_positive - repmat(mean_fea, size(features_positive,1), 1);
disp('start pca');
tic
options.ReducedDim = 256;
[eigvector, S, eigvalue] = PCA(features_positive, options);       
epsilon = 0.1; 
% % % --------------------whitening-------------------- 
features_positive = features_positive * eigvector* diag(1./sqrt(diag(S) + epsilon));


beta =10;
S_PQ = zeros(size(PROBE,1), 1);

for i=1:size(PROBE,1)
    Sum1 = 0;
    Sum2 = 0;
	pair_1=PROBE(i); % template_id for each pair
	pair_2=REFERENCE(i);
      
	pair_1_index=find(unique_template_id==pair_1); % find the relative template_id
	pair_2_index=find(unique_template_id==pair_2);
    
    pair1_m = Template_Media{pair_1_index}; %find the features based on media for each template
    pair2_m = Template_Media{pair_2_index};
    
    for p1 = 1:size(pair1_m)
        res_buf = 0;
        pair_1_i = find(unique_media_id == pair1_m(p1));
	    pair_1_features=features_positive(pair_1_i,:); % find the probe and reference features
        for p2 = 1:size(pair2_m)
            pair_2_i = find(unique_media_id == pair2_m(p2));
	        pair_2_features=features_positive(pair_2_i,:);
            res_buf = dot(pair_1_features,pair_2_features)/(norm(pair_1_features,2)*norm(pair_2_features,2)); 
            Sum1 = Sum1 + res_buf * exp(res_buf*beta);
            Sum2 = Sum2 + exp(res_buf*beta);
        end
    end
    S_PQ(i) = Sum1/Sum2; % softmax fusion
end

save('S_PQ_cosine','S_PQ');

auc = plot_roc(S_PQ, pair_labels)