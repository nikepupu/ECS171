function result = area(x, y)

result = 0;


for i = 2 : length(x)
  if x(i) ~= x(i - 1)
    result = result + (y(i) + y(i-1))* abs((x(i) - x(i-1)))/2;
  end
end

if max(x) < 1  
    [~, I] = max(x);
    result = result + (1 - max(x)) * y(I);
end

if min(x) > 0
  
  [~, I] = min(x);
  result = result + min(x) * y(I);
  
end

if min(x) == max(x) 
  result = 0;
end
end

