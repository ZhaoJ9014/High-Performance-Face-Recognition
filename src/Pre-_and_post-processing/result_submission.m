clc
clear all;

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

% Find probe & gallery template features
probe_feas = cell(size(unique_template_id_probe,1), 1);
gallery_feas = cell(size(unique_template_id_gallery,1), 1);

for i=1:size(unique_template_id_probe,1)
	pair_1=unique_template_id_probe(i); % template_id for each pair    
	pair_1_index=find(unique_template_id_probe==pair_1); % find the relative template_id
    pair1_m = Template_Media_PROBE{i}; %find the features based on media for each template
    pair_1_features = [];
    for p1 = 1:size(pair1_m)
        pair_1_i = find(unique_media_id_probe == pair1_m(p1));
	    pair_1_features=[pair_1_features; features_probe(pair_1_i,:)];
    end
    probe_feas{i} = pair_1_features;
end
clear i
clear pair_1

for j=1:size(unique_template_id_gallery,1)
	pair_2=unique_template_id_gallery(j); % template_id for each pair
	pair_2_index=find(unique_template_id_gallery==pair_2); % find the relative template_id
    pair2_m = Template_Media_GALLERY{j}; %find the features based on media for each template
    pair_2_features = [];
    for p2 = 1:size(pair2_m)
        pair_2_i = find(unique_media_id_gallery == pair2_m(p2));
	    pair_2_features=[pair_2_features; features_gallery(pair_2_i,:)];
    end
    gallery_feas{j} = pair_2_features;
end
clear j

% Write output files 
fid = fopen('/home/zhaojian/Documents/Projects/FACE/NIST_Submission/IJB-A_1N_output/split1/split1.candidate_lists','w');
fprintf(fid, 'SEARCH_TEMPLATE_ID CANDIDATE_RANK ENROLL_TEMPLATE_ID ENROLL_TEMPLATE_SIZE_BYTES SEARCH_TEMPLATE_SIZE_BYTES RETCODE SIMILARITY_SCORE\n');
RETCODE = 0;
load 'iden_S_PQ_split1'
% Normalize S_PQ
for i=1:length(unique_template_id_probe)  
    disp(i);
    for j = 1:length(unique_template_id_gallery)   
        S_PQ(j, i) = 1/(1+exp(-S_PQ(j, i)));
    end    
end
clear i
clear j

for i=1:length(unique_template_id_probe)   
    SEARCH_TEMPLATE_ID = unique_template_id_probe(i); % template_id for probe
    pair_1_feas = probe_feas{i}; 
    p = whos('pair_1_feas');
    SEARCH_TEMPLATE_SIZE_BYTES = p.bytes;
    [Y, I] = sort(S_PQ(:,i));
    for r = 1:20
        CANDIDATE_RANK = r - 1;
        SIMILARITY_SCORE = Y(end-(r-1));
        ENROLL_TEMPLATE_ID = unique_template_id_gallery(I(end-(r-1)));
        pair_2_feas = gallery_feas{I(end-(r-1))}; 
        g = whos('pair_2_feas');
        ENROLL_TEMPLATE_SIZE_BYTES = g.bytes;
        fprintf(fid,'%d %d %d %d %d %d %0.9f\n',SEARCH_TEMPLATE_ID, CANDIDATE_RANK, ENROLL_TEMPLATE_ID, ENROLL_TEMPLATE_SIZE_BYTES, SEARCH_TEMPLATE_SIZE_BYTES, RETCODE, SIMILARITY_SCORE);
    end    
end
fclose(fid);