%% read data

data = xlsread('ecs171.dataset.xlsx');
load 'GP.txt'
load 'strain.txt'
load 'stress.txt'
load 'medium.txt';


close all
yy = data(:,1);
xx = data(:,2:4496)' ;
q3 = zeros(1, 4495);
%% data regularization
for i = 1 : 4495
  if max(xx(i,:)) ~= min (xx(i,:))
    xx(i,:) = (xx(i,:) - min(xx(i,:)))/(max(xx(i,:)) - min(xx(i,:)));
    q3(i) = mean(xx(i,:));
  end
end


%% 10-fold randomlization

rng(100)
index = randsample(194,194);


%% initialization 



generalizationError = [];
prediction = [];
a = 0.1; % learning Rate
l = 10; % lambda; constrain parameter
num_iter = 500;

MSE = zeros(1, num_iter);
MSE2 = zeros(1, num_iter);
x = zeros(4495,175);
y = zeros(175,1);
testx = zeros(4495, 19);
testy = zeros(19,1);

%% update
% I use the update rule: 
% w = w(|w|/w - a * l) - a *(h(x) - y)* x

close all

for j = 1 : 10
  bias = ones(1,1);
  w = ones(4495, 1);
  count1 = 1;
  count2 = 1;
  
  for k = 1 : 194
    if  k >= ((j-1)*19  + 1) && k <= j * 19
      testx(:,count1) = xx(:, index(k)); 
      testy(count1) = yy(index(k));
      count1 = count1 + 1;
    else
      x(:,count2) =xx(:,index(k));
      y(count2) = yy(index(k));
      count2 = count2 + 1;
    end
  end
  
  for m = 1 : num_iter
   
    for i = 1 : 175
      temp = bias;
      bias = bias - a * (w' * x(:,i) + bias - y(i))/175;
      w = w .* (abs(w)./w - a * l/175) - a * (w' * x(:,i) + temp - y(i)) .* x(:,i)/175;
    end
   
    MSE(m) = sum((w' * x + bias)' - y)^2/175;
    MSE2(m) = sum((w' * testx + bias)' - testy)^2/19; 
  end

  generalizationError = [generalizationError MSE(m)-MSE2(m)];
  
  if j == 1
    minIndex = j;
    smallGE = abs(MSE(m) - MSE2(m));
    wmin = w;
    biasmin = bias;
  end
  
  if abs(MSE(m) - MSE2(m)) < smallGE
      minIndex = j;
      wmin = w;
      biasmin = bias;
      smallGE = abs(MSE(m) - MSE2(m));
  end
    
  % plot
  figure 
  t = 1 : num_iter;
  plot(t(400:num_iter), MSE(400:num_iter));
  
  prediction = [prediction (w' * xx + bias)'];
end

GE = sum(generalizationError)/10

  
non_zero = 4495 - count
%% Q2
fprintf('Q2:\n')
interval = [];
interval = [interval mean(prediction,2) - 1.96*std(prediction,0,2)/sqrt(194)];
interval = [interval mean(prediction,2) + 1.96*std(prediction,0,2)/sqrt(194)];

%% Q3
fprintf('Q3:\n')
predictionQ3 = q3 * wmin + biasmin

  count = 0;
  for i = 1 : 4495
    if wmin(i) <= 0.0008 && wmin(i) >= -0.0008
      count = count + 1;
      wmin(i) = 0;
    end   
  end


 %% Q4 data selection
fprintf('Q4:\n')
% data selection
selectedData = zeros(non_zero,194);
count1 = 1;
for i = 1 : 4495
  if wmin(i) ~= 0
    selectedData(count1, :) = xx(i+1,:);
    count1 = count1 + 1;
  end
end
%}

%% Q4 medium
 %correct category * fold

svm(medium, 18,selectedData, index, 'individual medium');


%% Q4 strain type

svm(strain, 10, selectedData, index, 'individual strain');
%% Q4 GP
svm(GP, 12, selectedData, index, 'individual GP');

%% Q4 stress
svm(stress, 8, selectedData, index ,'individual stress');

%% Q5 composite medium + stress

newy = zeros(194,1);
for i = 1 : 194
    newy(i) = 18 * stress(i) + medium(i);  
end
 
svm(newy, 144, selectedData, index, 'composite');   

%% Q6


A = xx' * xx;
[V, D] = eig(A);

D = sum(D);
[~, I] = max(D);
vector1 = V(:,I);

D(I) = 0;
[~, I] = max(D);
vector2 = V(:,I);

D(I) = 0;
[~, I] = max(D);
vector3 = V(:,I);


data1 =((xx * vector1) / norm(vector1))';
data2 =((xx * vector2) / norm(vector2))';
data3 =((xx * vector3) / norm(vector3))';

dataAll = [data1; data2; data3];

svm(medium, 18,dataAll, index, 'PCA medium');
svm(strain, 10, dataAll, index, 'PCA strain');
svm(GP, 12, dataAll, index, 'PCA GP');
svm(stress, 8, dataAll, index, 'PCA stress');



  
  




