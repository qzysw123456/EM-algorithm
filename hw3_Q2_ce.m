%Hi Ta, would you mind mannually choose to execute some part of the code.
%I hope my explanation is clear. Thank you very much!

%problem a,b
%[h m Q] = EMG(0,'stadium.bmp',4);
%[h m Q] = EMG(0,'stadium.bmp',8);
%[h m Q] = EMG(0,'stadium.bmp',12);


%problem c
%[h m Q] = EMG(0,'goldy.bmp',7);

%below part invoke the build in kmeans for question c

%%%%start build in k means

%%{
[img cmap] = imread('goldy.bmp');
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);
[n m] = size(img);
strip = reshape(img_double,[n*m 3]);

k = 7;
idx = kmeans(strip,k);

for i = 1 : k
    strip(idx==i,:) = repmat(mean(strip(idx==i,:)),length(strip(idx==i,:)),1);
    length(strip(idx==i,:))
end

img_double_new = reshape(strip,n,m,3);
image(img_double_new);
title('build-in-k-means');
%}

%%%%%% end for build in k means

%problem e
%EMG(1,'goldy.bmp',7);
