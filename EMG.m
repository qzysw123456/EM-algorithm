
%function [response mu EMplot]= EMG(flag,img,k)
close all;
rng(1331);
img = 'stadium.bmp';
k = 4;
flag =0;

%[img cmap] = imread('stadium.bmp');
[img cmap] = imread(img);
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);

[n m] = size(img);
strip = reshape(img_double,[n*m 3]);

%k = 4;
idx = kmeans(strip,k,'MaxIter',3,'EmptyAction','singleton');

response = zeros(n*m,k);

mu = zeros(k,3);
sigma = zeros(3,3,k);
N = zeros(1,k);
pi = zeros(1,k);

for i = 1 : k
    pi(i) = length(strip(idx==i,:))/length(strip);
end

Modify = -0.000005 * flag * eye(3);

for i = 1 : k
    class = strip(idx==i,:);
    mu(i, :) = mean(class);
    sigma(:, :, i) = cov(class) - Modify;
end

saved_log_likelihood = 0;
log_likelihood = 0;
like = [];
response = zeros(n*m,k);
iter = 100;
Eplot = [];
Mplot = [];
EMplot = [];
while iter > 0
    %E-step
    for i = 1 : n*m
        for t = 1 : k
            %den = 0;
            %for j = 1 : k
            %    den = den + pi(j)*(mvnpdf(strip(i,:),mu(:,:,j),sigma(:,:,j))/mvnpdf(strip(i,:),mu(:,:,t),sigma(:,:,t)));
            %end
            %response(i,t) = pi(t)/den;
            response(i,t) = pi(t)*mvnpdf(strip(i,:),mu(t,:),sigma(:,:,t));
        end
        t =sum(response(i,:));
        response(i,:) = response(i,:)/t;
    end

    %N = sum(response);
    %pi = N/sum(N);
    %calc
    logE = 0;
    for i = 1 : n*m
        for t = 1 : k
            logE = logE + response(i,t)*log(N(t))+response(i,t)*log(mvnpdf(strip(i,:),mu(t,:),sigma(:,:,t)));
        end
    end
    Eplot = [Eplot logE];
    EMplot = [EMplot logE];
    
    log_likelihood = 0;
    for i= 1: n*m
        sums = 0;
        for t=1:k
            sums = sums+pi(t)*mvnpdf(strip(i,:),mu(t,:) ,sigma(:,:,t));
        end
        log_likelihood = log_likelihood+log(sums);
    end
    saved_log_likelihood = log_likelihood;
    like = [like log_likelihood];
    
    %M-step
    N = sum(response);
    mu = zeros(k,3);
    sigma = zeros(3,3,k);
    for t = 1 : k
        for i = 1 : n*m
            mu(t,:) = mu(t,:) + response(i,t)*strip(i,:); 
        end
        mu(t,:) = mu(t,:)/N(t);
        for i = 1 : n*m
            sigma(:,:,t) = sigma(:,:,t) + response(i,t)*((strip(i,:)-mu(t,:))'*(strip(i,:)-mu(t,:)));
        end
        sigma(:,:,t) = sigma(:,:,t)/N(t);
        sigma(:,:,t) = sigma(:,:,t) - Modify;
    end
    pi = N/sum(N);
    %calc
    logM = 0;
    for i = 1 : n*m
        for t = 1 : k
            logM = logM + response(i,t)*log(N(t))+response(i,t)*log(mvnpdf(strip(i,:),mu(t,:),sigma(:,:,t)));
        end
    end
    
    log_likelihood = 0;
    for i= 1: n*m
        sums = 0;
        for t=1:k
            sums = sums+pi(t)*mvnpdf(strip(i,:),mu(t,:) ,sigma(:,:,t));
        end
        log_likelihood = log_likelihood+log(sums);
    end
    like = [like log_likelihood];

    
    Mplot = [Mplot logM];
    EMplot = [EMplot logM];
  
    if log_likelihood < 1.00001 * saved_log_likelihood
        break
    end

    iter = iter - 1;
end
[dummy,idx] = max(response,[],2);

for i = 1 : k
    strip(idx==i,:) = repmat(mean(strip(idx==i,:)),length(strip(idx==i,:)),1);
end

img_double_new = reshape(strip,n,m,3);
image(img_double_new);
title(['k = ' num2str(k)]);

figure(2)
hold all
scatter(1:2:2*length(Eplot)-1,Eplot);
scatter(2:2:2*length(Mplot),Mplot);
plot(1:length(EMplot),EMplot);
xlabel('StepE-StepM Alternate Step num');
ylabel('expected complete log-likelihood');
title(['k = ' num2str(k)]);
%end
figure(3)
plot(1:length(like),like);