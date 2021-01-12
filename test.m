clc, clear, close all

ori_img = imread('ori_img.bmp');
dist_img1 = imread('dist_img1.bmp');
dist_img2 = imread('dist_img2.bmp');
dist_img3 = imread('dist_img3.bmp');

score1 = MADDLS(ori_img, dist_img1) % 0.9872
score2 = MADDLS(ori_img, dist_img2) % 0.9532
score3 = MADDLS(ori_img, dist_img3) % 0.8954
