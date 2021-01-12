clc, clear, close all

I1 = imread('car.jpg');
I2 = imread('bike.jpg');
I3 = imread('kh.jpg');
I4 = imread('ali.jpg');

% out1 = gbvs(I1);
% out2 = gbvs(I2);
% out3 = gbvs(I3);
% out4 = gbvs(I4);

out1 = ittikochmap(I1);
out2 = ittikochmap(I2);
out3 = ittikochmap(I3);
out4 = ittikochmap(I4);

figure,
subplot(2,4,1)
imshow(I1)
subplot(2,4,2)
imshow(out1.master_map_resized)
subplot(2,4,3)
imshow(I2)
subplot(2,4,4)
imshow(out2.master_map_resized)
subplot(2,4,5)
imshow(I3)
subplot(2,4,6)
imshow(out3.master_map_resized)
subplot(2,4,7)
imshow(I4)
subplot(2,4,8)
imshow(out4.master_map_resized)