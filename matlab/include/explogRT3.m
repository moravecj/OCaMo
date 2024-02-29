function T = explogRT3(logT)
 %
 % EXPLOGRT3 The exponential of the logarithm of a 3D rigid motion matrix
 %
 % T = explogRT3(logT)
 %
 % Assumes logT as a 4x4 matrix as from logRT3(). Useful for motion
 % interpolation.
 %
 % See also: logRT3, ou2M, mcpoints
 
 % Reference:
 %
 % Allen-Blanchette, C. -- Leonardos, S. -- Gallier, J. Motion Interpolation in
 % SIM(3). Dept of Computer and Information Science, University of Pennsylvania,
 % December 15, 2014.

 % (c) 2017 Radim Sara (sara@cmp.felk.cvut.cz) FEE CTU Prague
 %
 % $Id$
 
 if any(size(logT) ~= [4,4])
  error([mfilename, ':WrongSize'], ...
   '%s requires a 4x4 matrix', mfilename)
 end
 
 B = logT(1:3,1:3);
 w = logT(1:3,4);
 
 theta = sqrt(sum(B(:).^2)/2);
 
 if theta > eps
  
  R = eye(3) + (sin(theta)*eye(3) + (1-cos(theta))/theta*B)*B/theta;
  b = B*w/theta^2;
  t = w + (1-cos(theta))*b + (theta-sin(theta))/theta*B*b;
  
  T = [R,t;0,0,0,1];
 else
  T = eye(4,4);
  T(1:3,4) = w;
 end
end
