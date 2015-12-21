%% read data

data = xlsread('data2.xls');
data = data(:,4:55 );
x = data(:,1:46) ;
y = data(:,47:52);

rng(100)
index = randsample(1531,1531);


for i = 1: 46
  if max(x(:,i)) - min(x(:,i)) ~= 0
    x(:,i) = (x(:,i) - min(x(:,i)))/(max(x(:,i)) - min(x(:,i)));
  end
end

%% generate label
boundary = zeros(6,4);
for i = 1 : 6
  boundary(i,:) = quantile(y(:,i), [ 0.2 0.4 0.6 0.8]);
end


  a = y(:,2);
  low = (a < boundary(i,1));
  med1 = 2* (a >= boundary(i,1) & a < boundary(i,2));
  med2 = 3 * (a >= boundary(i, 2) & a < boundary(i,3));
  med3 = 4 * (a >= boundary(i, 3) & a < boundary(i,4));
  high = 5 *(a >= boundary(i,4) );
  datay = low + med1 + med2 + med3 + high;



%{

  [totalcorrect, TP, w,bb] = svm(y(:,2)', 3, x', index);
  predict = zeros(1531);
  totalcorrectRate = totalcorrect/(30* 10)
  
 result = x * w;
 predicty = zeros(1531,1);
 correct  = 0;
 b = bb(:,1);
 for i = 1 : 1531
   result = x(i,:) * w + b';
   [~, k] = max(result);
   predicty(i) = k;
   if predicty(i) == y(i,2)
     correct = correct + 1;
   end
 end
 correct
%}
 
 %% logistic

   %{
 [w, datax]  = logistic(datay, x, 46, 5);
 
 predictiony = 1./(1 + exp(- datax * w));
 predictiony = predictiony';
 [~, I] = max(predictiony);
  finalPrediction = I';
  
  correct = 0;
  for i = 1 : 1531
    if datay(i) == finalPrediction(i)
      correct = correct  + 1;
    end
  end
 
 
 %}
 
 
