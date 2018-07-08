function [ data ] = svm_scale( data )
%SVM_SCALE Summary of this function goes here
%   Detailed explanation goes here
% each row is a sample
for i=1:size(data,1)
    %data(:,i) = (2*(data(:,i)-min(data(:,i))))/(max(data(:,i))-min(data(:,i)))-1;
    if norm(data(i,:))~=0
        data(i,:) = data(i,:)/norm(data(i,:));
    end
end

