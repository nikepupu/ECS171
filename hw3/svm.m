function svm(datay, num_category,selectedData, index, type)

fprintf('training for %s start\n', type)
testx = zeros(size(selectedData,1),19);
testy = zeros(19,1);
x = zeros(size(selectedData,1),175);
y = zeros(175,1);

for category = 1 : num_category % one vs all
  figure
  for j = 1 : 10; % 10 fold cross validation
    count1 = 1;
    count2 = 1;
    
  for k = 1 : 194
    if  k >= ((j-1)*19  + 1) && k <= j * 19
      testx(:,count1) = selectedData(:,index(k)); 
       if datay(index(k)) == category
        testy(count1) = 1;
      else
        testy(count1) = -1;
      end
      
      count1 = count1 + 1;
    else
      x(:,count2) =selectedData(:,index(k));
      if datay(index(k)) == category
        y(count2) = 1;
      else
        y(count2) = -1;
      end
      count2 = count2 + 1;
    end % if
  end % for k
  
  %% SMO
 [a, b] = trainingsvm(x,y);
 optw = x * (a .* y);
   
 %%
  
    predict = optw' * testx + b;
    prediction = zeros(19,1);
    count = 1;
      l = length((min(predict)-1):0.01:(max(predict)+1));
      TP = zeros(l,1); 
      FP = zeros(l,1);
      FN = zeros(l,1);
      TN = zeros(l,1);
  
     for t = (min(predict)-1):0.01:(max(predict)+1)
      
      for i = 1 : 19
        if predict(i) > t
          prediction(i) = 1;
        else
          prediction(i) = -1;
        end
    
        if prediction(i) == 1 && testy(i) == 1
          TP(count) = TP(count) + 1;
        
        end
    
        if prediction(i) == 1 && testy(i) == -1
          FP(count) = FP(count) + 1;
        end
    
        if prediction(i) == -1 && testy(i) == -1
          TN(count) = TN(count) + 1;
        end
    
        if prediction(i) == -1 && testy(i) == 1
          FN(count) = FN(count) + 1;
        end
          
      end % for i = 1 :19
      
      count = count + 1;
      
    end  % for t = min
    correct = max(TP + TN);
    fprintf('correct: %d out of 19\n', correct)
    recall = (TP + 1) ./ (TP + FN + 1);
    specificity = (TN + 1) ./(TN + FP + 1);
    FPR = 1 - specificity;
    precision = (TP + 1) ./(TP + FP + 1);
      
  
    
    subplot(1,2,1)
    hold on
    if j == 1
      name = sprintf('ROC-graph category category%d %s',category);
      title(name)
      ylabel('reacll')
      xlabel('false positive rate')
    end
    AUC = area(FPR, recall);
    name = sprintf('The %dth classifier AUC = %f', j, AUC);
    plot(FPR, recall, 'DisplayName', name);
    legend('-DynamicLegend');
    
   
    subplot(1,2,2)
    hold on
    if j == 1
      name = sprintf('Precision-Recall graph category%d', category);
      title(name)
      ylabel('precision')
      xlabel('recall')
    end
    AUCPR = area(recall, precision);
    name = sprintf('The %dth classifier AUCPR = %f', j, AUCPR);
    plot(recall, precision, 'DisplayName', name);
    legend('-DynamicLegend');
   fprintf('%dth fold complete\n', j)
   
  end % for j 10-cross
  fprintf('category %d complete\n', category)
end % for category
  fprintf('training for %s complete\n', type)
end