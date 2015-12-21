  %CYT (cytosolic or cytoskeletal)                    1
  %NUC (nuclear)                                      2
  %MIT (mitochondrial)                                3
  %ME3 (membrane protein, no N-terminal signal)       4
  %ME2 (membrane protein, uncleaved signal)           5
  %ME1 (membrane protein, cleaved signal)             6
  %EXC (extracellular)                                7
  %VAC (vacuolar)                                     8
  %POX (peroxisomal)                                  9
  %ERL (endoplasmic reticulum lumen)                  10
  
  
nodes = 12; % number of node per hidden layer;
load data.txt;
rng(200)
trainingIndex = randsample(1484, 965);

trainingSet = zeros(9, 965);
testingSet = zeros(9, 519);

count1 = 1;
count2 = 1;
for k = 1 : 1484
  if ismember(k, trainingIndex)
    trainingSet(:,count1 ) =  data(k, :)';
    count1 = count1 + 1;
  else
    testingSet(:, count2) =  data(k, :)';
    count2 = count2 + 1;
  end
end


trainingX = trainingSet(1:8, :); %8 * 965
trainingY = trainingSet(9, :);
testingX = testingSet(1:8,:);
testingY = testingSet(9, :);

% convert y to vectors
trainingYvec = zeros(10, 965);
testingYvec = zeros(10, 519);
for i = 1 : 965
k = zeros(10, 1);
k(trainingY(i)) = 1;
trainingYvec(:, i) = k;
end

for i = 1 : 519
k = zeros(10, 1);
k(testingY(i)) = 1;
testingYvec(:, i) = k;
end


rng(200)
w1 = rand(nodes,8);
w2 = rand(nodes,nodes);
w3 = rand(10, nodes);
a = 0.1; % training rate

iter_num = 1000;
bias = rand(1,1);
bias2 = rand(1,1);
bias3 = rand(1,1);

trainingError = zeros(iter_num, 1);
testingError = zeros(iter_num, 1);
prediction = zeros(10, 965);
predictionTest = zeros(10, 519);


for m = 1 : iter_num

  for k = 1 : 965 % sample
      % a2 is a 3 * 1484 matrix
      x = trainingX(:,k);
      z1 = w1 * x  + bias;
      a2 = sigmoid(z1);
        
      % a3 is a 10 * 965 matrix
      z2 = w2 * a2 +  bias2;
      a3 = sigmoid(z2);
      
      z3 = w3 * a3 + bias3;
      a4 = sigmoid(z3);
      prediction(:,k) = a4;
  
     
      % bp 
      delta4 = a4 - trainingYvec(:, k); % 10 * 1
      delta3 = w3' * delta4 .* ( a3 .* (1 - a3));
      delta2 = w2' * delta3 .* ( a2 .* (1 - a2));
      w3 = w3 - a * delta4 * a3';
      w2 = w2 - a * delta3 * a2' ;
      w1 = w1 - a * delta2 * x' ;
  end


      % a2 is a 3 * 965 matrix
      xx = testingX;
      z1 = w1 * xx + bias;
      a2 = sigmoid(z1);

      % a3 is a 10 * 965 matrix
      z2 = w2 * a2 + bias2;
      a3 = sigmoid(z2);

      z3 = w3 * a3 + bias3;
      a4 = sigmoid(z3);
      predictionTest = a4;

      testingError(m) = sum(1/519 * sum( (predictionTest - testingYvec).^2 ));
      trainingError(m) = sum(1/1484 * sum( (prediction - trainingYvec).^2 ));

end    

for t = 1 : 965
  [M, indice] = max(prediction(:,t));
  prediction(:,t) = zeros(10, 1);
  prediction(indice,t) = 1;
end

error = prediction - trainingYvec;
count = 0;
for t = 1 : 965
  if isequal(error(:,t),zeros(10,1))
    count = count + 1;
  end
end

for t = 1 : 519
  [M, indice] = max(predictionTest(:,t));
  predictionTest(:,t) = zeros(10, 1);
  predictionTest(indice, t) = 1;
end

count2 = 0;
error = predictionTest - testingYvec;
for t = 1 : 519
  if isequal(error(:,t),zeros(10,1))
    count2 = count2 + 1;
  end
end



close all
% plot error chnage
t = 1:iter_num;
plot(t,trainingError)
hold
plot(t, testingError)
xlabel('number of iterations')
ylabel('error')
legend('training error', 'testingError')
title('Error--iterations(0 - 3.16)')


fprintf('The training correct percentage is: %f\n', count/965)
fprintf('The testing correct percentage is: %f\n', count2/519)
fprintf('The testing error is: %f\n', testingError(iter_num))

 