clc
clear

load 'PROBE'
load 'REFERENCE'
load 'TEMPLATE_ID'
load 'SUBJECT_ID'

pair_labels=zeros(length(PROBE),1);
for i=1:length(PROBE)
    pair1=PROBE(i);
    pair2=REFERENCE(i);
    index_pair1=find(TEMPLATE_ID==pair1);
    index_pair2=find(TEMPLATE_ID==pair2);
    if(SUBJECT_ID(index_pair1(1))==SUBJECT_ID(index_pair2(1)))
        pair_labels(i,1)=1;
    else
        pair_labels(i,1)=0;
    end
end

save('pair_labels','pair_labels');