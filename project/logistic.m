function [optw, datax] = logistic(datay, datax, num_x, num_category)
  
datax = [ones(1531, 1) datax];
optw = zeros(num_x + 1, num_category);

for category = 1: num_category
  y = zeros(size(datay,1),1);  
  for i = 1 : num_x
    
   if datay(i) == category
     y(i) = 1;
   end
   
  end

  iter_num = 1000;
  a = 0.00001;
  w = rand(num_x+1,1);
  MSE = zeros(iter_num,1);
  for i = 1 : iter_num
    w = w + a * datax' * (y - 1./(1 + exp(-datax * w)));
    MSE(i) = 1/1531 * sum(y - 1./(1 + exp(-datax * w))).^2;
  end
  optw(:,category) = w;
  figure
  t = 1 : iter_num;
  plot(t, MSE);
  
  
end
