%EMG(0,'stadium.bmp',4);
%EMG(0,'stadium.bmp',8);
%EMG(0,'stadium.bmp',12);
%EMG(0,'goldy.bmp',7);

%{
[img cmap] = imread('goldy.bmp');
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);
[n m] = size(img);
strip = reshape(img_double,[n*m 3]);

k = 7;
idx = kmeans(strip,k);

for i = 1 : k
    strip(idx==i,:) = repmat(mean(strip(idx==i,:)),length(strip(idx==i,:)),1);
end

img_double_new = reshape(strip,n,m,3);
image(img_double_new);
title('build-in-k-means');
%}

EMG(1,'goldy.bmp',7);
%EMG(0,'stadium.bmp',4);