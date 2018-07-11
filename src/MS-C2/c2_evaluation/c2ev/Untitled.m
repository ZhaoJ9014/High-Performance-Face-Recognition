load('base1.mat')
feature = data;
load('noveltest.mat')
feature = [feature; data];
[eigvec, ~, ~, sampleMean] = PCA(feature);
feature = ( bsxfun(@minus, feature, sampleMean)* eigvec );
a = feature(1:21000,:);
b = feature(21001:26000,:);
save rt a
save lt b