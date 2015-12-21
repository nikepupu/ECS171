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

trainingX = data(:, 1:8)'; %10 * 1484
trainingY = data(:, 9)';


% convert y to vectors
trainingYvec = zeros(10, 1484);

for i = 1 : 1484
  k = zeros(10, 1);
  k(trainingY(i)) = 1;
  trainingYvec(:, i) = k;
end

rng(200)
w1 = rand(3,8);
w2 = rand(10,3);

a = 0.1; % training rate
iter_num = 1000;
bias = rand(1,1) ;
bias2 = rand(1,1)  ;
trainingError = zeros(iter_num, 1);
prediction = zeros(10, 1484);

for m = 1 : iter_num

  for k = 1 : 1484 % sample 
      % a2 is a 3 * 965 matrix
      x = trainingX(:,k);
      z1 = w1 * x  + bias;
      a2 = sigmoid(z1);
        
      % a3 is a 10 * 965 matrix
      z2 = w2 * a2 +  bias2;
      a3 = sigmoid(z2);
      prediction(:,k) = a3;
  
      % bp 
      delta3 = a3 - trainingYvec(:, k); % 10 * 1
      delta2 = w2' * delta3 .* ( a2 .* (1 - a2));% 3 * 1
      w2 = w2 - a * delta3 * a2' ;
      w1 = w1 - a * delta2 * x' ;
  end
  
  
  trainingError(m) = sum(sum(trainingYvec .* log(prediction) + (1 - trainingYvec) .* log(1 - prediction)));
  %sum(1/1484 * sum( (prediction - trainingYvec).^2 ));

end    

  for t = 1 : 1484
      [M, indice] = max(prediction(:,t));
      prediction(:,t) = zeros(10, 1);
      prediction(indice,t) = 1; 
  end

  error = prediction - trainingYvec;
  count = 0;
  for t = 1 : 1484
    if isequal(error(:,t),zeros(10,1))
      count = count + 1;
    end  
  end


  
 close all 
 % plot error chnage
 t = 1:iter_num;
 plot(t,trainingError)
 xlabel('number of iterations')
 ylabel('error')
 legend('training error')
 title('Error--iterations(0 - 3.16)')
        
fprintf('The training correct percentage is: %f\n', count/1484)


 