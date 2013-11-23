function [x] = scale(x, min_val, max_val)
  x = x/max(range(x)); %max(x);
  x = abs(min(min(x))) + x; 
  x = min_val + x*(max_val-min_val);
  x(x>max_val) = max_val;
  x(x<min_val) = min_val;
  
  
  % scale(0:10, 1,5)' 
  % scale(0:10, 1,5)' 