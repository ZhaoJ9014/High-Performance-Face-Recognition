function auc = plot_roc(predict,groundtruth)
    pos_num = sum (groundtruth == 1);
    neg_num = sum (groundtruth == 0);
    
    m = size(groundtruth , 1);
    [pre,index] = sort(predict);
    groundtruth = groundtruth (index);
    x = zeros (m+1,1);
    y = zeros (m+1,1);
    auc = 0;
    x(1) = 1;
    y(1) = 1;
    
    for i = 2:m
        TP = sum(groundtruth(i:m)==1);
        FP = sum(groundtruth(i:m)==0);
        x(i) = FP / neg_num;
        y(i) = TP / pos_num;
        auc = auc + (y(i)+y(i-1))*(x(i-1)-x(i))/2;
    end
    
    x(m+1) = 0;
    y(m+1) = 0;
    
    I0001= find (x==0.001)
    I001 = find(x==0.01)
    I01  = find(x==0.1)
    Y0001= y(I0001)
    Y001 = y(I001)
    Y01  = y(I01)
    auc = auc + y(m) * x(m)/2;
    plot(x,y)
    hold on
    xlabel ('FAR')
    ylabel ('TAR')
end