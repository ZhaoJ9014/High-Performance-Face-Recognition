% clc
% clear all;

load 'unique_template_id_probe'
load 'unique_template_id_gallery'
load 'features_gallery'
load 'features_probe'
load 'template_subject_probe'
load 'template_subject_gallery'
load 'unique_media_id_probe'
load 'unique_media_id_gallery'
load 'Template_Media_PROBE'
load 'Template_Media_GALLERY'

features_probe=svm_scale(features_probe);
features_gallery=svm_scale(features_gallery);

% %reduce dimentation using whitening PCA
mean_fea = mean(features_probe,1);
features_probe = features_probe - repmat(mean_fea, size(features_probe,1), 1);
features_gallery = features_gallery - repmat(mean_fea, size(features_gallery,1), 1);
disp('start pca');
options.ReducedDim = 1024;
[eigvector, S, eigvalue] = PCA(features_probe, options);       
epsilon = 0.1;
% % --------------------whitening-------------------- 
features_probe = features_probe * eigvector* diag(1./sqrt(diag(S) + epsilon));
features_gallery = features_gallery * eigvector* diag(1./sqrt(diag(S) + epsilon));

% %reduce dimentation using whitening PCA
mean_fea = mean(features_probe,1);
features_probe = features_probe - repmat(mean_fea, size(features_probe,1), 1);
features_gallery = features_gallery - repmat(mean_fea, size(features_gallery,1), 1);
disp('start pca');
options.ReducedDim = 256;
[eigvector, S, eigvalue] = PCA(features_probe, options);       
epsilon = 0.1;
% % --------------------whitening-------------------- 
features_probe = features_probe * eigvector* diag(1./sqrt(diag(S) + epsilon));
features_gallery = features_gallery * eigvector* diag(1./sqrt(diag(S) + epsilon));


beta =10;

for i=1:length(unique_template_id_probe)
       
    pair1_m = Template_Media_PROBE{i};
    for j = 1:length(unique_template_id_gallery)
        res_buf = 0;
        pair2_m = Template_Media_GALLERY{j};
        Sum1 = 0;
        Sum2 = 0;
        for p1 = 1:size(pair1_m)
           pair_1_i = find(unique_media_id_probe == pair1_m(p1));
	       pair_1_features=features_probe(pair_1_i,:); % find the probe and reference features
           for p2 = 1:size(pair2_m)
              pair_2_i = find(unique_media_id_gallery == pair2_m(p2));
	          pair_2_features=features_gallery(pair_2_i,:);
              res_buf = dot(pair_1_features,pair_2_features)/(norm(pair_1_features,2)*norm(pair_2_features,2)); 
              Sum1 = Sum1 + res_buf * exp(res_buf*beta);
              Sum2 = Sum2 + exp(res_buf * beta);
           end
        end
        S_PQ(j, i) = Sum1/Sum2;
    end
end

%open set
%%find the subjects unenrolled
enroll = zeros(length(template_subject_probe),1);
right_size = 0;
for u = 1:length(template_subject_probe)
    if(length(find(template_subject_gallery==template_subject_probe(u))))
        enroll(u) = 1;
        right_size = right_size+1;
    end
end
unenroll = find(enroll==0);
en = find(enroll==1);

%calculate max score for wrong pairs
score_max = zeros(length(unenroll),1); 
for i=1:length(unenroll)
    ue_index = unenroll(i);
    [dist_val dist_ind] = max(S_PQ(:,ue_index));
    score_max(i) = dist_val;
end

%obtain threshold
min_value = min(score_max);
max_value = max(score_max);
inteval = abs(max_value-min_value)/1000;
for i=1:1000
    stop(i) = min_value + inteval*i;
    x(i) = length(find(score_max>=stop(i)))/length(score_max);
    is_correct = 0;
    for e=1:length(en)
        e_index = en(e);
        [dist_val dist_ind] = max(S_PQ(:,e_index));
        if template_subject_gallery(dist_ind) == template_subject_probe(e_index) && dist_val >= stop(i)
            is_correct = is_correct + 1;
        end
    end
    y(i) = is_correct / length(en);
        
end
plot(x,y);

%%%close set
acc = [];
for r = 1:100
    correct = 0;
    for p = 1:length(unique_template_id_probe)
        [Y,I] = sort(S_PQ(:,p));
        if (find(template_subject_gallery(I(end-(r-1):end)) == template_subject_probe(p)))
            correct = correct +1;
        end  
    end
    acc(r) = correct / (length(unique_template_id_probe)-551);
end

Rank = [1:1:100];
figure,
plot(Rank, acc);