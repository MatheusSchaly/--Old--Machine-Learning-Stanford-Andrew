function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    binary_pred = pval < epsilon; # convert the probability(?) (pval) to 1 when pval < epsilon and 0 otherwise
    tp = sum((binary_pred == 1) & (yval == 1)); # predicted 1 when ground truth was 1 (true positive)
    fp = sum((binary_pred == 1) & (yval == 0)); # predicted 1 when ground truth was 0 (false positive)
    fn = sum((binary_pred == 0) & (yval == 1)); # predicted 0 when ground truth was 1 (false negative)
    
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    F1 = (2*precision*recall)/(precision+recall);


    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
