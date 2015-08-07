function H = make_plots(infile, start_ind)

load(infile)

intData = intData(start_ind:end,:);
gradData = gradData(start_ind:end,:);


% figure
% scatter(intData(:), gradData(:));
% xlim([-2000,2000]);ylim([-2000,2000]);title('Model Domain');
% xlabel('intercept');ylabel('gradient');

nscales = ceil(log2(min(size(gradData))) - 3);
C = opCurvelet(size(gradData,1), size(gradData, 2));


% find the sparsest curvelet representation
m0 = randn(size(C, 1),1)*.0001;
tau = 0;
sigma = 0;
options = {};
options.iterations=500;

[mI, r, g, info] = spgl1(C', intData(:), tau, sigma, m0, options);

[mG, r, g, info] = spgl1(C', gradData(:), tau, sigma, m0, options);

Ci = C*intData(:);
Cg = C*gradData(:);

%mI = Ci;
%mG = Cg;
%load testing.mat

%H = digi(intData, gradData, Ci, Cg, mI, mG, C);


size_in = size(intData);

filt_opt = struct();
filt_opt.min_margin = [0,0];

% shannon filter bank as the perfect frequency tiling property
% i.e. it has a constant 1 littlewood paley sum
filt_opt.filter_type = 'shannon';

scat_opt = struct();
scat_opt.oversampling = 0;
[Wop, filters] = wavelet_factory_2d(size_in, filt_opt, scat_opt);

[Si, Ui] = scat(intData, Wop);

[Sg, Ug] = scat(gradData, Wop);


% grab the scattering coefficients from the last layer
% Iscat = [Si{3}.signal{:}];
% Gscat = [Sg{3}.signal{:}];
% 
% figure
% scatter(Iscat(:), Gscat(:));title('Scattering xPlot');
% xlabel('Intercept'); ylabel('Gradient');
% 
% figure
% scatter(real(Ci(:)), real(Cg(:)));title('Real Curvelet xPlot');
% xlabel('Intercept'); ylabel('Gradient');
% 
% 
% figure
% scatter(abs(Ci(:)), abs(Cg(:)));title('Magnitude Curvelet xPlot');
% xlabel('Intercept'); ylabel('Gradient');
% 
% 
% figure
% scatter(intData(:), gradData(:));title('IG xPlot');
% xlabel('Intercept'); ylabel('Gradient');




end