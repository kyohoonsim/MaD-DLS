function [score, quality_map] = DLS(Y1, Y2)

T = 170; 
Down_step = 2;
dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';

aveKernel = fspecial('average',2);
aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:Down_step:end,1:Down_step:end);
Y2 = aveY2(1:Down_step:end,1:Down_step:end);

IxY1 = conv2(Y1, dx, 'same');     
IyY1 = conv2(Y1, dy, 'same');    
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');     
IyY2 = conv2(Y2, dy, 'same');

gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);
quality_map = (2*gradientMap1.*gradientMap2 + T) ./(gradientMap1.^2+gradientMap2.^2 + T);
score = std2(quality_map);
