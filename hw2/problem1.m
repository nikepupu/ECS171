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
  
load data.txt;
rng(200)
nodes = 3; % number of node per hidden layer;

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
w2 = rand(10,nodes);
w1
w2
a = 0.1; % training rate
iter_num = 1000;
bias = rand(1,1) ;
bias2 = rand(1,1)  ;
trainingError = zeros(iter_num, 1);
testingError = zeros(iter_num, 1);
prediction = zeros(10, 965);
predictionTest = zeros(10, 519);

changew1 = zeros(8, iter_num);
changew2 = zeros(3, iter_num);
changeOutput = zeros(1, iter_num);
for m = 1 : iter_num
  changew1(:, m) = w1(1,:);
  changew2(:,m)  = w2(1,:);
  for k = 1 : 965 % sample 
        if k== 2 && m == 1
          fprintf('after the first iteration, w1 and w2 becomes: ')
          w1
          w2
          p1 = w1;
          p2 = w2;
        end
        
        % a2 is a 3 * 965 matrix
        x = trainingX(:,k);
        z1 = w1 * x  + bias;
        a2 = sigmoid(z1);
        
        % a3 is a 10 * 965 matrix
        z2 = w2 * a2 +  bias2;
        a3 = sigmoid(z2);
        prediction(:,k) = a3;
        
        if k == 1
          changeOutput(m) = a3(1);
        end
        % bp 
      delta3 = a3 - trainingYvec(:, k); % 10 * 1
      delta2 = w2' * delta3 .* ( a2 .* (1 - a2));% 3 * 1
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
        predictionTest = a3; 
  
  testingError(m) = sum(1/519 * sum( (predictionTest - testingYvec).^2 )); 
  trainingError(m) = sum(1/965 * sum( (prediction - trainingYvec).^2 ));
 
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
 legend('training error', 'testing error')
 title('Error--iterations(0 - 3.16)')
 hold off
 
 % plot change of the weights in the 1st node in the hidden layer
 figure
 title('change of weights coming to the first node in the hidden layer')
 hold
 for i = 1:8
   plot(t, changew1(i, :) );
 end
 legend('1st feature', '2nd feature', '3rd feature', '4th feature','5th feature', ... 
 '6th feature', '7th feature', '8th feature')
xlabel('number of iterations')
ylabel('weights')
hold off

% plot change of the weights in the 1st node in the output layer
figure
title('change of weights coming to the first node in the output layer')
hold
for i = 1:3
  plot(t, changew2(i, :) );
end
legend('1st feature', '2nd feature', '3rd feature')
xlabel('number of iterations')
ylabel('weights')
hold off

% plot change of the output for the 1st node of the 1st sample
figure
hold
title('change of output of the first node in the output layer for the 1st training sample')
xlabel('number of iterations')
ylabel('output(0 to 1)')
plot(t, changeOutput)
hold off        
fprintf('The training correct percentage is: %f\n', count/965)
fprintf('The testing correct percentage is: %f\n', count2/519)

 