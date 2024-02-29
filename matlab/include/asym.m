function A = asym(a)
 %
 % ASYM Skew-symmetric 3x3 matrix from 3D vector
 %
 % A = asym(a)
 %
 % Constructs 3x3 skew-symmetric (antisymmetric) matrix A=[a]_x from vector a.
 % It holds that cross(a, b) = asym(A)*b and C*b = cross(asym2v(C), b).
 %
 % See also: asym2v
 
 % (c) 2000-2016 Radim Sara (sara@cmp.felk.cvut.cz) FEE CTU Prague

 % $Id$
 
 if (numel(a) ~= 3)
  error([mfilename, ':WrongDimension'], 'Dimension of the input vector must be 3')
 end
 
 A = [0, -a(3), a(2);  a(3), 0, -a(1);  -a(2), a(1), 0]; 
end
