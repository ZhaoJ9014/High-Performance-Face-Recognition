clc
clear all;

load 'FILE.mat'
load 'SUBJECT_ID.mat'
load 'TEMPLATE_ID.mat'
load 'PAIR1.mat'
load 'PAIR2.mat'
load 'S_PQ_V_Norm_Beta_0_split1.mat'

pair_labels = zeros(length(PAIR1),1);

for i=1:length(PAIR1)
    p1 = PAIR1(i);
    ind1 = find(TEMPLATE_ID == p1);
    p2 = PAIR2(i);
    ind2 = find(TEMPLATE_ID == p2);
    if (SUBJECT_ID(ind1(1)) == SUBJECT_ID(ind2(1)))
        pair_labels(i) = 1;
    else 
        pair_labels(i) = 0;
    end
end
clear i

pos_ind = find(pair_labels == 1);
neg_ind = find(pair_labels == 0);

mated_score = S_PQ(pos_ind);
nonmated_score = S_PQ(neg_ind);

[sorted_mated_score, sorted_mated_ind] = sort(mated_score);
[sorted_nonmated_score, sorted_nonmated_ind] = sort(nonmated_score);

% Best mated
top30_mated_score = sorted_mated_score((end-29):end);
top30_mated_ind = pos_ind(sorted_mated_ind((end-29):end));
top30_mated_p1 = PAIR1(top30_mated_ind);
top30_mated_p2 = PAIR2(top30_mated_ind);

for i = 1:length(top30_mated_p1)
    tem1 = top30_mated_p1(i);
    tem2 = top30_mated_p2(i);
    score = top30_mated_score(i);
    
    tem1_ind = find(TEMPLATE_ID == tem1); 
    tem2_ind = find(TEMPLATE_ID == tem2);
    img_file_name1 = strcat('top_', num2str(i), '_best_mated_tem_', num2str(tem1), '.txt');
    img_file_name2 = strcat('top_', num2str(i), '_best_mated_tem_', num2str(tem2), '.txt');
    for j = 1:length(tem1_ind)
        tem1_file = FILE{tem1_ind(j),1};
        fid1 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_mated1/img_file/', img_file_name1), 'a+');
        fprintf(fid1,'\r\n%s', tem1_file);   
        fclose(fid1);
    end
    
    for k = 1:length(tem2_ind)
        tem2_file = FILE{tem2_ind(k),1};
        fid2 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_mated2/img_file/', img_file_name2), 'a+');
        fprintf(fid2,'\r\n%s', tem2_file);   
        fclose(fid2);
    end
    
    score_file_name = strcat('top_', num2str(i), '_best_mated', '.txt');
    fid3 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_mated1/score_file/', score_file_name), 'a+');
    fprintf(fid3,'\r\n%6.5f', score); 
    fclose(fid3);
    
    tem_file_name1 = strcat('top_', num2str(i), '_best_mated', '.txt');
    fid4 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_mated1/template_file/', tem_file_name1), 'a+');
    fprintf(fid4,'\r\n%d', tem1);
    fclose(fid4);
    tem_file_name2 = strcat('top_', num2str(i), '_best_mated', '.txt');
    fid5 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_mated2/template_file/', tem_file_name2), 'a+');
    fprintf(fid5,'\r\n%d', tem2);
    fclose(fid5);
end

% Worst mated
bottom30_mated_score = sorted_mated_score(1:30);
bottom30_mated_ind = pos_ind(sorted_mated_ind(1:30));
bottom30_mated_p1 = PAIR1(bottom30_mated_ind);
bottom30_mated_p2 = PAIR2(bottom30_mated_ind);

for i = 1:length(bottom30_mated_p1)
    tem1 = bottom30_mated_p1(i);
    tem2 = bottom30_mated_p2(i);
    score = bottom30_mated_score(i);
    
    tem1_ind = find(TEMPLATE_ID == tem1); 
    tem2_ind = find(TEMPLATE_ID == tem2);
    img_file_name1 = strcat('bottom_', num2str(i), '_worst_mated_tem_', num2str(tem1), '.txt');
    img_file_name2 = strcat('bottom_', num2str(i), '_worst_mated_tem_', num2str(tem2), '.txt');
    for j = 1:length(tem1_ind)
        tem1_file = FILE{tem1_ind(j),1};
        fid1 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_mated1/img_file/', img_file_name1), 'a+');
        fprintf(fid1,'\r\n%s', tem1_file);   
        fclose(fid1);
    end
    
    for k = 1:length(tem2_ind)
        tem2_file = FILE{tem2_ind(k),1};
        fid2 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_mated2/img_file/', img_file_name2), 'a+');
        fprintf(fid2,'\r\n%s', tem2_file);   
        fclose(fid2);
    end
    
    score_file_name = strcat('bottom_', num2str(i), '_worst_mated', '.txt');
    fid3 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_mated1/score_file/', score_file_name), 'a+');
    fprintf(fid3,'\r\n%6.5f', score); 
    fclose(fid3);
    
    tem_file_name1 = strcat('bottom_', num2str(i), '_worst_mated', '.txt');
    fid4 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_mated1/template_file/', tem_file_name1), 'a+');
    fprintf(fid4,'\r\n%d', tem1);
    fclose(fid4);
    tem_file_name2 = strcat('top_', num2str(i), '_worst_mated', '.txt');
    fid5 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_mated2/template_file/', tem_file_name2), 'a+');
    fprintf(fid5,'\r\n%d', tem2);
    fclose(fid5);
end

% Best nonmated
top30_nonmated_score = sorted_nonmated_score(1:30);
top30_nonmated_ind = neg_ind(sorted_nonmated_ind(1:30));
top30_nonmated_p1 = PAIR1(top30_nonmated_ind);
top30_nonmated_p2 = PAIR2(top30_nonmated_ind);

for i = 1:length(top30_nonmated_p1)
    tem1 = top30_nonmated_p1(i);
    tem2 = top30_nonmated_p2(i);
    score = top30_nonmated_score(i);
    
    tem1_ind = find(TEMPLATE_ID == tem1); 
    tem2_ind = find(TEMPLATE_ID == tem2);
    img_file_name1 = strcat('top_', num2str(i), '_best_nonmated_tem_', num2str(tem1), '.txt');
    img_file_name2 = strcat('top_', num2str(i), '_best_nonmated_tem_', num2str(tem2), '.txt');
    for j = 1:length(tem1_ind)
        tem1_file = FILE{tem1_ind(j),1};
        fid1 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_nonmated1/img_file/', img_file_name1), 'a+');
        fprintf(fid1,'\r\n%s', tem1_file);   
        fclose(fid1);
    end
    
    for k = 1:length(tem2_ind)
        tem2_file = FILE{tem2_ind(k),1};
        fid2 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_nonmated2/img_file/', img_file_name2), 'a+');
        fprintf(fid2,'\r\n%s', tem2_file);   
        fclose(fid2);
    end
    
    score_file_name = strcat('top_', num2str(i), '_best_nonmated', '.txt');
    fid3 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_nonmated1/score_file/', score_file_name), 'a+');
    fprintf(fid3,'\r\n%6.5f', score); 
    fclose(fid3);
    
    tem_file_name1 = strcat('top_', num2str(i), '_best_nonmated', '.txt');
    fid4 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_nonmated1/template_file/', tem_file_name1), 'a+');
    fprintf(fid4,'\r\n%d', tem1);
    fclose(fid4);
    tem_file_name2 = strcat('top_', num2str(i), '_best_nonmated', '.txt');
    fid5 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/best_nonmated2/template_file/', tem_file_name2), 'a+');
    fprintf(fid5,'\r\n%d', tem2);
    fclose(fid5);
end

% Worst nonmated
bottom30_nonmated_score = sorted_nonmated_score((end-29):end);
bottom30_nonmated_ind = neg_ind(sorted_nonmated_ind((end-29):end));
bottom30_nonmated_p1 = PAIR1(bottom30_nonmated_ind);
bottom30_nonmated_p2 = PAIR2(bottom30_nonmated_ind);

for i = 1:length(bottom30_nonmated_p1)
    tem1 = bottom30_nonmated_p1(i);
    tem2 = bottom30_nonmated_p2(i);
    score = bottom30_nonmated_score(i);
    
    tem1_ind = find(TEMPLATE_ID == tem1); 
    tem2_ind = find(TEMPLATE_ID == tem2);
    img_file_name1 = strcat('bottom_', num2str(i), '_worst_nonmated_tem_', num2str(tem1), '.txt');
    img_file_name2 = strcat('bottom_', num2str(i), '_worst_nonmated_tem_', num2str(tem2), '.txt');
    for j = 1:length(tem1_ind)
        tem1_file = FILE{tem1_ind(j),1};
        fid1 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_nonmated1/img_file/', img_file_name1), 'a+');
        fprintf(fid1,'\r\n%s', tem1_file);   
        fclose(fid1);
    end
    
    for k = 1:length(tem2_ind)
        tem2_file = FILE{tem2_ind(k),1};
        fid2 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_nonmated2/img_file/', img_file_name2), 'a+');
        fprintf(fid2,'\r\n%s', tem2_file);   
        fclose(fid2);
    end
    
    score_file_name = strcat('bottom_', num2str(i), '_worst_nonmated', '.txt');
    fid3 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_nonmated1/score_file/', score_file_name), 'a+');
    fprintf(fid3,'\r\n%6.5f', score); 
    fclose(fid3);
    
    tem_file_name1 = strcat('bottom_', num2str(i), '_worst_nonmated', '.txt');
    fid4 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_nonmated1/template_file/', tem_file_name1), 'a+');
    fprintf(fid4,'\r\n%d', tem1);
    fclose(fid4);
    tem_file_name2 = strcat('top_', num2str(i), '_worst_nonmated', '.txt');
    fid5 = fopen(strcat('/home/zhaojian/Documents/IJB-A/error_case_visu/split1/worst_nonmated2/template_file/', tem_file_name2), 'a+');
    fprintf(fid5,'\r\n%d', tem2);
    fclose(fid5);
end