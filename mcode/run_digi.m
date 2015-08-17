addpath(genpath('~/src/APS_Release'));
addpath('~/src/APS_Release/extra');

% 
% filepath = {'data/'}
% filename_string = 'gather'
% il_byte = '189'
% xl_byte = '193'
% offset_byte = '37'
% parallel = '0'
% output_dir = '/data/Global/segy/test_digi_datset/Penobscot/Misc/output/'
% 
% segy_make_job({'data/'},'angle_stack','189','193','37','0','0','output/')
% 

load 'output/job_meta/job_meta_14Jul2015.mat'
% for i=433:n_blocks
%     
%     blockstr = num2str(i);
%     wavelet_estimation('output/job_meta/job_meta_14Jul2015.mat',blockstr,'1','1')
% end    
% wavelet_avg('output/job_meta/job_meta_14Jul2015.mat')
% meta_data_2d_smo_xy('output/job_meta/job_meta_14Jul2015.mat')
parfor i=1:n_blocks
block = num2str(i);
int_grad_inv_proj('output/job_meta/job_meta_14Jul2015.mat',block,'2','1','7','0','500', '0')
end
