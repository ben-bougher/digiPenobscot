load tempIG.mat

C = opCurvelet(size(I,1),size(I,2));

cI = C*I(:);

cG = C*G(:);

save tempCurve.mat
