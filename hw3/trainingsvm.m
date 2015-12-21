function [a, b] = trainingsvm(x, y)
 
 C = 1;
 a = zeros(175,1);
 b = 0;
 pass = 0;
 maxPass = 100;
 tol = 0.01;
 count = 0;
 disp('training...')
 while pass < maxPass && count < 5000
   num_change_alpha = 0;
   for i = 1 : 175
     E1 = sum(a .* y .* (x' *  x(:,i) )) - y(i);
     if (y(i) * E1 < - tol && a(i) < C )|| (y(i) * E1 > tol && a(i) > 0)
       j = randi([1 175], 1,1);
       E2 = sum(a .* y .* (x' *  x(:,j) )) - y(j);
       a1old = a(i);
       a2old = a(j);
       if y(i) == y(j)
         L = max(0, a(i)+a(j)-C);
         H = min(C, a(i) + a(j));
       else
         L = max(a(j)-a(i), 0);
         H = min(C, C+a(j)-a(i));
       end
      
       if L == H  % test for L and H
         continue
       end
       n = 2* kernal(x(:,i), x(:,j)) - kernal(x(:,i), x(:,i)) - kernal(x(:,j), x(:,j));
       if n >= 0 % test for second derivative
         continue
       end
       
       a(j) = a(j) - y(j)*(E1 - E2)/n;
       if a(j) > H
         a(j) = H;
       elseif a(j) < L
           a(j) = L;
       end
         
       if abs(a(j) - a2old) < 10^(-5)
         continue
       end
       
       a(i) = a(i) + y(i)*y(j)*(a2old - a(j));
       b1 = b - E1 -y(i)*(a(i) - a1old)* kernal(x(:,i), x(:,i)) - y(j)*(a(j) - a2old)*kernal(x(:,i), x(:,j));
       b2 = b - E2 -y(i)*(a(i) - a1old)* kernal(x(:,i), x(:,j)) - y(j)*(a(j) - a2old)*kernal(x(:,j), x(:,j));
       
       if a(i) > 0 && a(i) < C
         b = b1;
       elseif a(j) > 0 && a(j) < C
         b = b2;
       else
         b = 0.5 * (b1 + b2);
       end
       num_change_alpha = num_change_alpha + 1;
     end % end if
    
   end % end for
   if num_change_alpha == 0
     pass = pass + 1;
   else
     pass = 0;
   end
   count = count + 1;
 end % end while
disp('training complete')
end