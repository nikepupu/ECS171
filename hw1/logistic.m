function [theta1, theta2, theta3, range, mi] = logistic(trainingSet, testingSet, boundary)

  theta1 = ones(8, 1);
  theta2 = ones(8, 1);
  theta3 = ones(8, 1);
  range = max(trainingSet) - min(trainingSet);
  mi = min(trainingSet);
  trainingSet =bsxfun(@rdivide, bsxfun(@minus, trainingSet, mi), range);
  testingSet = bsxfun(@rdivide, bsxfun(@minus, testingSet, mi), range) ;
  
  xTest = [ones(112, 1), testingSet(:, 2:8)];
  alpha =  1; % learning rate
  num_iters = 10000;
  
 % training for the 1 st category
  x = [ones(280,1) trainingSet(:,2:8) ];
  y = trainingSet(:, 1);
  for i = 1 : 280
    if y(i) < boundary(1)
      y(i) = 1;
    else
      y(i) = 0;
    end
  end
  
  for i = 1 : num_iters
    delta =  x' * (y - 1./(1 + exp(- x * theta1) ) );
    theta1 = theta1 + alpha * delta;
  end
  
   MSE1 = 1/280 * sum( (y - 1./(1 +exp(-x * theta1)) ).^2 )
   %testing part1
   
    yTest = testingSet(:, 1);
    for i = 1 : 112
    if yTest(i) < boundary(1)
      yTest(i) = 1;
    else
      yTest(i) = 0;
    end
    end
  
    MSE1test = 1/112 * sum( (yTest - 1./(1 +exp(-xTest * theta1)) ).^2 ) 
   % training for the 2nd category
   
    y = trainingSet(:, 1);
    for i = 1 : 280
      if (y(i) >= boundary(1) && y(i) <= boundary(2))
        y(i) = 1;
      else
        y(i) = 0;
      end
    end


  for i = 1 : num_iters
    delta =  x' * (y - 1./(1 + exp(-x * theta2)));
    theta2 = theta2 + alpha * delta;
  end
  
   MSE2 = 1/280 * sum( (y - 1./(1 +exp(-x * theta2)) ).^2 )
   
   %testing for part2
   yTest = testingSet(:, 1);
   for i = 1 : 112
    if (yTest(i) >= boundary(1) & yTest(i) <= boundary(2))
      yTest(i) = 1;
    else
      yTest(i) = 0;
    end
     end
  
     MSE2test = 1/112 * sum( (yTest - 1./(1 +exp(-xTest * theta2)) ).^2 )
   % training for the 3rd category
   y = trainingSet(:, 1);
   for i = 1 : 280
     if y(i) > boundary(2)
       y(i) = 1;
     else
       y(i) = 0;
     end
   end

   for i = 1 : num_iters
     delta =  x' * (y - 1./(1 + exp(-x * theta3)) );
     theta3 = theta3 + alpha * delta;
   end
  
   MSE3 = 1/280 * sum( (y - 1./(1 +exp(-x * theta3)) ).^2 )
   
   %testing for part3
   yTest = testingSet(:, 1);
   for i = 1 : 112
     if (yTest(i) > boundary(2))
       yTest(i) = 1;
     else
       yTest(i) = 0;
     end
   end
   
   MSE3test = 1/112 * sum( (yTest - 1./(1 +exp(-xTest * theta3)) ).^2 )
   
end