%function [resp,means,Q,compressed_img] = EMG2(flag, image_path, k)
flag = 0;
image_path = 'stadium.bmp';
k = 4;

lambda = 0;
if(flag==1)
    lambda = 0.0001;
end
rng(1331);

%tweak the threshold
threshold = 3.0;
%convert img into double array
[img,cmap] = imread(image_path);
img_rgb = ind2rgb(img,cmap);
img_double = im2double(img_rgb);
%convert 3D matrix into 2D matrix
X = reshape(img_double,size(img_double,1)*size(img_double,2),3);
%run kmeans by 3 iterations; idx means the clustering index for each point,
%C means the centroid of each clustering
[idx] = kmeans(X,k,'MaxIter',3,'EmptyAction','singleton');

%calculate the initial parameters
pi = zeros(1,k);
covs = cell(1,k);
means = cell(1,k);
for i=1:k
    Xi = X(idx==i,:);
    pi(i) = size(Xi,1)./size(X,1);
    means{i} = mean(Xi,1);
    covs{i} = cov(Xi);
end

Q = [];

iteration_of_EM = 1;
%start the EM algorithm
for iter = 1:iteration_of_EM
    %E step
    %calculate responsibility
    resp = zeros(size(X,1),k);
    for i=1:size(X,1)
        %calculate the denomator
        denominator = 0;
        for p=1:k
            %why denominator can be zero?
            %mvnpdf(X(i,:),means{p},covs{p})
            denominator = denominator+pi(p)*mvnpdf(X(i,:),means{p},covs{p});
        end
        for p=1:k
            resp(i,p) = pi(p)*mvnpdf(X(i,:),means{p},covs{p})./denominator;
        end
        %decide the membership of each pixel
        [maxvalue, index] = max(resp(i,:));
        idx(i) = index;
    end
%{
    log_likelihood = 0;
    for i=1:size(X,1)
        sums = 0;
        for p=1:k
            sums = sums+resp(i,p)*pi(p)*mvnpdf(X(i,:),means{p},covs{p});
        end
        log_likelihood = log_likelihood+log(sums);
    end
    penalty = 0;
    for p=1:k
        target_cov = inv(covs{p});
        for j=1:3
            penalty = penalty+target_cov(j,j);
        end
    end
    penalty = -lambda/2*penalty;
    log_likelihood = log_likelihood+penalty;
    last_likelihood = log_likelihood;
    Q = [Q; log_likelihood];
    %end of E step
    
    %M step
    for p=1:k
        %calculate Nk
        sum_N = sum(resp,1);
        Nk = sum_N(p);
        %calculate mean k
        sums = 0;
        for i=1:size(X,1)
            sums = sums+resp(i,p)*X(i,:);
        end
        means{p} = sums./Nk;
        %calculate covariance matrix k
        sums = 0;
        for i=1:size(X,1)
            sums = sums+resp(i,p).*(X(i,:)-means{p})'*(X(i,:)-means{p});
        end
        %if flag == 1, then lambda != 0, the regularization works
        sums = sums+lambda*eye(3);
        covs{p} = sums./Nk;
        %calculate pi k
        pi(p) = Nk./size(X,1);
    end
    
    %calculate log likelihood
    log_likelihood = 0;
    for i=1:size(X,1)
        sums = 0;
        for p=1:k
            sums = sums+resp(i,p)*pi(p)*mvnpdf(X(i,:),means{p},covs{p});
        end
        log_likelihood = log_likelihood+log(sums);
    end
    
    penalty = 0;
    for p=1:k
        target_cov = inv(covs{p});
        for j=1:3
            penalty = penalty+target_cov(j,j);
        end
    end
    penalty = -lambda/2*penalty;
    log_likelihood = log_likelihood+penalty;
    Q = [Q; log_likelihood];
    
    if(abs(last_likelihood-log_likelihood)<threshold)
        break;
    end
%}
end

compressed_img = zeros(size(X,1),3);
for i=1:size(X,1)
    compressed_img(i,:) = means{idx(i)};
end
%restore the img into m*n*3
compressed_img = reshape(compressed_img,size(img_double,1),size(img_double,2),3);
%show the image after quantization
imshow(compressed_img)
A = 1:length(Q)
figure
hold on
plot(A,Q,'or-','MarkerIndices',1:2:length(A),'DisplayName','Expectation Step')
plot(A,Q,'ob-','MarkerIndices',2:2:length(A),'DisplayName','Maxmization Step')
