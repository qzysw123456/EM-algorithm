
close all;

[img cmap] = imread('goldy.bmp');
%[img cmap] = imread('stadium.bmp');
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