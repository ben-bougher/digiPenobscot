function crossplot()



parfor inline=1001:1600
    
    intData = [];
    gradData = [];

    
    n_blocks=575;

l1_old=0;
for i = 1:n_blocks
    
    grad_file =['output/digi_results/digi_gradient_w_range_2_1_7/digi_gradient_w_range_2_1_7_block_',num2str(i),'.segy']; 
    int_file =['output/digi_results/digi_intercept_w_range_2_1_7/digi_intercept_w_range_2_1_7_block_',num2str(i),'.segy']; 

    [grad_data, gradHeaders, ~] = ReadSegy(grad_file);
    [int_data, intHeaders, ~] = ReadSegy(int_file);

    l1 = size(gradData,2);
    
    gradData = [gradData, filterInline(grad_data, gradHeaders, inline)];
    intData = [intData, filterInline(int_data, intHeaders, inline)]; 
    
    l2 = size(gradData,2);
%     
%     if l1 == l2 & l1>l1_old
%         
%         figure
%         scatter(intData(:), gradData(:));
%         xlim([-2000,2000]);ylim([-2000,2000]);title('Model Domain');
%         xlabel('intercept');ylabel('gradient');   
%     
%         C = opCurvelet(size(gradData,1), size(gradData, 2));
% 
%     
%         Ci = C*intData(:);    
%         Cg = C*gradData(:);
% 
%         figure;
%         scatter(Ci(:), Cg(:));ylim([-2000,2000]);xlim([-2000,2000]);
%         title('Curvelet Domain');xlabel('intercept');ylabel('gradient');
%         
%         figure;
%         imagesc(gradData);
%         figure;
%         imagesc(intData(10:end,:));
%         
%     end

    l1_old = l1;
end


    parsave(inline, gradData, intData);
    
end
end


function parsave(inline, gradData, intData)
save(['/data/slim/bbougher/', 'inline_', num2str(inline), '.mat'], 'gradData', 'intData');
end

function output = filterInline(data, headers, inline)

    indices = find([headers.Inline3D] == inline);
    
    output = data(2:end, indices);
    

    
    
end