function w = linearSolver(trainingSet, testingSet, feature )
  figure
  scatter(testingSet( :, feature), testingSet(:, 1))
  ylabel('mpg')
  str = sprintf('the %dth feature', feature );
  xlabel(str)
  hold
  
  y = trainingSet( :, 1);  
  x = [];
  % i controls the order
  for i = 0 : 4 
    x = [x trainingSet( :, feature).^i];
  
    w = pinv(x' * x) * x' * y;
    
    MSE1 = 1/280 * sum( (trainingSet( :, 1) - x * w ).^2 );
    fprintf('Training mean square error for %dth order %dth feature is: %f\n',i, feature, MSE1)
  
    xTest = [];
    for k = 0 : i 
      xTest = [xTest testingSet( :, feature).^k];
    end

    MSE2 = 1/112 * sum( (testingSet( : , 1) - xTest * w ).^2 ) ;
    fprintf('Testing mean square error for %dth order %dth feature is: %f\n', i, feature, MSE2)

    l = (min(testingSet( : , feature)):0.1:max(testingSet( : , feature)))';
    p = [];

    for k = 0 : i
      p = [p  l.^k ];
    end

    q = p * w;

    plot(l, q)

  end % for i
    legend('data points', '0 the order', '1st order', '2nd order', '3rd order', '4th order')
    hold off
    
end % end of function