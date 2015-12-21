function theta = solver2(trainingSet, testingSet, order)

    data1 = ones(280,1);
    
    for i = 1 : order
        k = trainingSet(:, 2:8) .^i;
        data1 = [data1 k];
    end
    
    result = trainingSet( : , 1); 
    theta = pinv(data1' * data1) * data1' * result;
    
    MSE1 = 1/280 * sum( (result - data1 * theta ).^2 ) ;
    disp(sprintf('training error is %f', MSE1))
    xTest = ones(112, 1);
    
    for i = 1 : order
      xTest = [xTest testingSet( : , 2:8).^i];
    end

    MSE2 = 1/112 * sum( (testingSet( : , 1) - xTest * theta ).^2 ) ;
    disp(sprintf('testing error is %f', MSE2))
    
    
end