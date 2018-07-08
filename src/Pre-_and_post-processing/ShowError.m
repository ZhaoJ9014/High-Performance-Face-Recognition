%%%%%%% error analysis

Results = pair_labels .* Similarity;

Error = find(Results<0);

Pos_Neg = find(Error<1804);
[S,IPN] = sort(Similarity(Pos_Neg));

Neg_Pos = find(Error> 1803);
[S,INP] = sort(Similarity(Neg_Pos));

[numPN, Nsize] = size(Pos_Neg);

[numNP, Nsize] = size(Neg_Pos);

PathPN = 'G:/NIST.FACE/Results/ShowError/PN/';
PathNP = 'G:/NIST.FACE/Results/ShowError/NP/';

for i = 1: numPN
    index = Pos_Neg(IPN(i));
    Ind   = Error(index);
    P1 = verifycomparisons1(Ind,1);
    P2 = verifycomparisons1(Ind,2);
    TInd1 = find (cell2mat(verifymetadata1(:,1))==P1);
    [N1,M1] = size(TInd1);
    TInd2 = find (cell2mat(verifymetadata1(:,1))==P2);
    [N2,M2] = size(TInd2);
    
    savePath1 = [PathPN num2str(i) '/Template' num2str(P1) '/' ];
    savePath2 = [PathPN num2str(i) '/Template' num2str(P2) '/' ];
     SPath11 = [savePath1 'img/'];
     mkdir(SPath11);
     SPath12 = [savePath1 'frame/'];
     mkdir(SPath12);
    for a = 1: N1
     I = imread(['H:/NIST_IJB/From Zhaojian/T2/Test/' verifymetadata1{TInd1(a),3}]);
     G = find (verifymetadata1{TInd1(a),3} == '/');
     imwrite(I, [savePath1 verifymetadata1{TInd1(a),3}]);
    end
     SPath21 = [savePath2 'img/'];
     mkdir(SPath21);
     SPath22 = [savePath2 'frame/'];
     mkdir(SPath22);
    for b = 1: N2
     J = imread(['H:/NIST_IJB/From Zhaojian/T2/Test/' verifymetadata1{TInd2(b),3}]);
     G = find (verifymetadata1{TInd2(b),3} == '/');
     imwrite(J, [savePath2 verifymetadata1{TInd2(b),3}]);       
    end
       
end  

for i = 1: numNP
    index = Neg_Pos(INP(i));
    Ind   = Error(index);
    P1 = verifycomparisons1(Ind,1);
    P2 = verifycomparisons1(Ind,2);
    TInd1 = find (cell2mat(verifymetadata1(:,1))==P1);
    [N1,M1] = size(TInd1);
    TInd2 = find (cell2mat(verifymetadata1(:,1))==P2);
    [N2,M2] = size(TInd2);
    
    savePath1 = [PathNP num2str(i) '/Template' num2str(P1) '/' ];
    savePath2 = [PathNP num2str(i) '/Template' num2str(P2) '/' ];
     SPath11 = [savePath1 'img/'];
     mkdir(SPath11);
     SPath12 = [savePath1 'frame/'];
     mkdir(SPath12);
    for a = 1: N1
     I = imread(['H:/NIST_IJB/From Zhaojian/T2/Test/' verifymetadata1{TInd1(a),3}]);
     G = find (verifymetadata1{TInd1(a),3} == '/');
     imwrite(I, [savePath1 verifymetadata1{TInd1(a),3}]);
    end
     SPath21 = [savePath2 'img/'];
     mkdir(SPath21);
     SPath22 = [savePath2 'frame/'];
     mkdir(SPath22);
    for b = 1: N2
     J = imread(['H:/NIST_IJB/From Zhaojian/T2/Test/' verifymetadata1{TInd2(b),3}]);
     G = find (verifymetadata1{TInd2(b),3} == '/');
     imwrite(J, [savePath2 verifymetadata1{TInd2(b),3}]);       
    end
       
end  