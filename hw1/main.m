close all
load 'auto-mpg.data.txt'
data = auto_mpg_data;
a = data(:,1);
 % problem #1

%boundary = quantile(a, [0.33 0.66]);
a = sort(a);

boundary = [a(130) a(260)];
fprintf('Problem #1\nthe threshold is %f and %f\n', boundary(1), boundary(2) );

%problem #2
a = data(:,1);
low = (a < boundary(1));
med = 2* (a >= boundary(1) & a <= boundary(2));
high = 3 *(a > boundary(2) );

label = low + med + high;

gplotmatrix(data, [], label);

%problem #3 see linearSolver.m
%problem #4
s = 100;
rng(s)
trainingIndex = randsample(392, 280);
trainingSet = [];
testingSet = [];

for i = 1 : 392
  if ismember(i, trainingIndex)
      trainingSet = [trainingSet; data(i, :)];
  else
      testingSet = [testingSet; data(i, :)];
  end
end


sprintf('#4 \n')
for i = 2 : 8
linearSolver( trainingSet, testingSet, i);
% w is the result for each feature
end


 % problem #5
 sprintf('#5 \n')
for i = 0:2
  theta = solver2(trainingSet, testingSet,  i);
end

% problem #6

 sprintf('#6 \n')
 [w1, w2, w3, range, mi] = logistic(trainingSet, testingSet, boundary); 


% problem #7
model = [6 300 170 3600 9 80 1];
model =  [1 model model.^2];
mpgPrediction = model * theta

model2 = [1 6 300 170 3600 9 80 1];

model2 = (model2 - mi) ./ range;

p1 =  1./(1 +exp(-model2 * w1))
p2 =  1./(1 +exp(-model2 * w2))
p3 =  1./(1 +exp(-model2 * w3))
   
    

   
