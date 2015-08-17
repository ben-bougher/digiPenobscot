function [M,R,F,INFO] = coSparseCurvelet(I, G)


    C = opCurvelet(size(I,1), size(I,2));
    
    C = C'
    D = [I(:) G(:)];
    
    M0 = zeros(size(C,1), 2);
    
    size(D)
    options = {};
    options.iterations=200;
    [M, R, F, INFO] = spg_mmv(C,D,0,options);
    

end


function D = matrixCurve(M, mode, d1,d2)


    C = opCurvelet(d1,d2);
    
    if mode==1
        D = C'*M;
    end
    
    if mode ==2
        D = C*M;
    end
    
end