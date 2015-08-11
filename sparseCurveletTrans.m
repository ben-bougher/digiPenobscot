load tempIG.mat

M = coSparseCurvelet(I,G);

cI = M(:,1);
cG = M(:,2);

save tempCurve.mat, cI, cG

