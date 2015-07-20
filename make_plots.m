function make_plots(infile, start_ind)

load(infile)

intData = intData(start_ind:end,:);
gradData = gradData(start_ind:end,:);


figure
scatter(intData(:), gradData(:));
xlim([-2000,2000]);ylim([-2000,2000]);title('Model Domain');
xlabel('intercept');ylabel('gradient');

C = opCurvelet(size(gradData,1), size(gradData, 2));


% find the sparsest curvelet representation
m0 = randn(size(C, 1),1)*.0001;
tau = 0;
sigma = 0;
options = {};
options.iterations=500;

[mI, r, g, info] = spgl1(C', intData(:), tau, sigma, m0, options);

[mG, r, g, info] = spgl1(C', gradData(:), tau, sigma, m0, options);


thresh_low = [mI < -500 & mG < -500];
thresh_high = [mI > 500 & mG > 500];

threshI = zeros(size(mI));
threshG = zeros(size(mG));

threshI(thresh_low) = mI(thresh_low);
threshI(thresh_high) = mI(thresh_high);

threshG(thresh_low) = mG(thresh_low);
threshG(thresh_high) = mG(thresh_high);


newI = C'*threshI;
newG = C'*threshG;

figure;

I_image = gray2ind(intData, 64);
I_image = ind2rgb(I_image, colormap());
I_image = rgb2hsv(I_image);
I_image(:,:,3) = new_i / max(newI);

imshow(I_image);





Ci = C*intData(:);
Cg = C*gradData(:);

figure;
scatter(Ci(:), Cg(:));ylim([-2000,2000]);xlim([-2000,2000]);
title('Curvelet Domain');xlabel('intercept');ylabel('gradient');

figure;
imagesc(gradData);
figure;
imagesc(intData);

end