function [score] = MADDLS(ori, dist)

a = 1/16;
sr = 4; 
b = 2;

addpath('matconvnet-1.0-beta25/matlab')
addpath('matconvnet-1.0-beta25/gbvs')

% Setup MatConvNet.
run matconvnet-1.0-beta25/matlab/vl_setupnn;

% Load a model and upgrade it to MatConvNet current version.
% net = load('imagenet-vgg-f.mat');
net = load('matconvnet-1.0-beta25/imagenet-vgg-f.mat');

net = vl_simplenn_tidy(net);

ori_ = single(ori); % note: 255 range
resized_ave_image = imresize(net.meta.normalization.averageImage, size(ori(:, :, 1)));
ori_ = ori_ - resized_ave_image;

dist_ = single(dist); % note: 255 range
dist_ = dist_ - resized_ave_image;

 %% GBVS
gbvs_map = gbvs(ori);
saliency_map = gbvs_map.master_map_resized;

% Run the CNN.
res_ori = vl_simplenn(net, ori_);
res_dist = vl_simplenn(net, dist_);


feat_map1_ori = rgb2gray(ori);
feat_map1_dist = rgb2gray(dist);

feat_map4_ori = double(res_ori(4).x(:, :, 1:sr:end));
feat_map4_dist = double(res_dist(4).x(:, :, 1:sr:end));

feat_map8_ori = double(res_ori(8).x(:, :, 1:sr:end));
feat_map8_dist = double(res_dist(8).x(:, :, 1:sr:end));

feat_map11_ori = double(res_ori(11).x(:, :, 1:sr:end));
feat_map11_dist = double(res_dist(11).x(:, :, 1:sr:end));

feat_map13_ori = double(res_ori(13).x(:, :, 1:sr:end));
feat_map13_dist = double(res_dist(13).x(:, :, 1:sr:end));

feat_map15_ori = double(res_ori(15).x(:, :, 1:sr:end));
feat_map15_dist = double(res_dist(15).x(:, :, 1:sr:end));


 %% structure quality score
%% 1
saliency_map1 = imresize(saliency_map, [ceil(size(feat_map1_ori, 1)/2), ceil(size(feat_map1_ori, 2)/2)]);
[score1, ssim_map1] = DLS(feat_map1_ori, feat_map1_dist);

wscore1 = sum(sum(saliency_map1/sum(sum(saliency_map1)).*ssim_map1))^((a^score1)^b)*(a^score1)^(1-(a^score1)^b);

%% 4        
saliency_map4 = imresize(saliency_map1, [ceil(size(feat_map4_ori, 1)/2), ceil(size(feat_map4_ori, 2)/2)]);
for j = 1:size(feat_map4_ori, 3)
    [score4, ssim_map4_temp] = DLS(feat_map4_ori(:, :, j), feat_map4_dist(:, :, j));

    wscore4(j) = sum(sum(saliency_map4/sum(sum(saliency_map4)).*ssim_map4_temp))^((a^score4)^b)*(a^score4)^(1-(a^score4)^b);
end

%% 8     
saliency_map8 = imresize(saliency_map4, [ceil(size(feat_map8_ori, 1)/2), ceil(size(feat_map8_ori, 2)/2)]);
for j = 1:size(feat_map8_ori, 3)
    [score8, ssim_map8_temp] = DLS(feat_map8_ori(:, :, j), feat_map8_dist(:, :, j));

    wscore8(j) = sum(sum(saliency_map8/sum(sum(saliency_map8)).*ssim_map8_temp))^((a^score8)^b)*(a^score8)^(1-(a^score8)^b);
end

%% 11   
saliency_map11 = imresize(saliency_map8, [ceil(size(feat_map11_ori, 1)/2), ceil(size(feat_map11_ori, 2)/2)]);
for j = 1:size(feat_map11_ori, 3)
    [score11, ssim_map11_temp] = DLS(feat_map11_ori(:, :, j), feat_map11_dist(:, :, j));

    wscore11(j) = sum(sum(saliency_map11/sum(sum(saliency_map11)).*ssim_map11_temp))^((a^score11)^b)*(a^score11)^(1-(a^score11)^b);
end

%% 13
saliency_map13 = saliency_map11; 
for j = 1:size(feat_map13_ori, 3)
    [score13, ssim_map13_temp] = DLS(feat_map13_ori(:, :, j), feat_map13_dist(:, :, j));

    wscore13(j) = sum(sum(saliency_map13/sum(sum(saliency_map13)).*ssim_map13_temp))^((a^score13)^b)*(a^score13)^(1-(a^score13)^b);
end


%% 15
saliency_map15 = saliency_map13;   
for j = 1:size(feat_map15_ori, 3)
    [score15, ssim_map15_temp] = DLS(feat_map15_ori(:, :, j), feat_map15_dist(:, :, j));

    wscore15(j) = sum(sum(saliency_map15/sum(sum(saliency_map15)).*ssim_map15_temp))^((a^score15)^b)*(a^score15)^(1-(a^score15)^b);
end

score = mean([mean(wscore1), mean(wscore4), mean(wscore8), mean(wscore11), mean(wscore13), mean(wscore15)]);

end

