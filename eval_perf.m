% performance evaluation: recall, balanced accuracy and f-measure

function [recall,bAcc,F1]=eval_perf(y_test,y_pred,posclass)
confMat = confusionmat(y_test, y_pred);
    TP = confMat(2, 2); 
    FP = confMat(1, 2); 
    FN = confMat(2, 1); 
    TN = confMat(1, 1); 
    if posclass==0
        TP = confMat(1, 1); 
        FP = confMat(2, 1); 
        FN = confMat(1, 2); 
        TN = confMat(2, 2); 
    end
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    bAcc = (sensitivity + specificity) / 2; %balanced accuracy
    F1 = 2 * (precision * recall) / (precision + recall);
    if isnan(bAcc), bAcc = 0; end
    if isnan(recall), recall = 0; end
    if isnan(F1), F1 = 0; end
end
