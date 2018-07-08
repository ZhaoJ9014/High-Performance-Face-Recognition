function [faRate, hitRate, AUC, EER, Acc] = ROCcurve(score, trueLabel, doPlot)
%% Plot a receiver operating curve
% function [faRate, hitRate, AUC, EER] = plotROCcurve(score, trueLabel, doPlot)
%
% score(i) = confidence in i'th detection (bigger means more confident)
% trueLabel(i) = 0 if background or 1 if target
% doPlot - optional (default 0)
%
% faRate(t) = false alarm rate at t'th threshold
% hitRate(t) = detection rate at t'th threshold 
% AUC = area under curve
% EER = equal error rate
% Acc = Accuracy

if nargin < 3, doPlot = 0; end

class1 = find(trueLabel==1);
class0 = find(trueLabel==0);

thresh = sort(score);
Nthresh = length(thresh);
hitRate = zeros(1, Nthresh);
faRate = zeros(1, Nthresh);
rec = zeros(1, Nthresh);
for thi=1:length(thresh)
    th = thresh(thi);
    % hit rate = TP/P
    hitRate(thi) = sum(score(class1) >= th) / length(class1);
    % fa rate = FP/N
    faRate(thi) = sum(score(class0) >= th) / length(class0);
    rec(thi) =  (sum(score(class1) >= th) + sum(score(class0) < th)) / Nthresh;
end

% area under curve
AUC = sum(abs(faRate(2:end) - faRate(1:end-1)) .* hitRate(2:end));

% equal error rate
i1 = find(hitRate >= (1-faRate), 1, 'last' ) ;
i2 = find((1-faRate) >= hitRate, 1, 'last' ) ;
EER = 1 - max(1-faRate(i1), hitRate(i2)) ;

% Accuracy
Acc = max(rec);

if ~doPlot, return; end 

plot(faRate, hitRate, '-');
      

xlabel('False Positiucve Rate')
ylabel('True Positive Rate')
grid on
title(sprintf('AUC = %5.4f, EER = %5.4f', AUC, EER))

end