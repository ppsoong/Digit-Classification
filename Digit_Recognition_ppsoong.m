%%% Patrick Soong
%%% 916220178
%%% MAT 167 Fall 2019
%%% Handwritten Digit Classification
%%% OS: Windows 10 Pro 64-Bit
%%% MATLAB R2019b

%%%%%%%%%%%%%%%%
%%%% STEP 1 %%%%
%%%%%%%%%%%%%%%%

%%% Step 1A
%%% Setting up the data and renaming arrays

load('USPS.mat')
format long

training_digits = train_patterns;
test_digits = test_patterns;
training_labels = train_labels;


%%% Step 1B
%%% Displays first 16 images in training_digits in a 4x4 plot

for k = 1:16
    subplot(4, 4, k)
    imagesc(reshape(training_digits(:,k),[16, 16])')
end

%%%%%%%%%%%%%%%%
%%%% STEP 2 %%%%
%%%%%%%%%%%%%%%%

%%% Creates an array of zeros for storing the mean digit values
%%%
%%% Calculates the mean digit value for each of the 10 digits
%%% 
%%% Displays the images for the 10 mean digit images

training_averages = zeros(256, 10);
for k = 1:10
        training_averages(:,k) = ...
            mean(training_digits(:, training_labels(k,:) == 1), 2);
        subplot(2, 5, k)
        imagesc(reshape(training_averages(:,k),[16, 16])')
end

%%%%%%%%%%%%%%%%
%%%% STEP 3 %%%%
%%%%%%%%%%%%%%%%

%%% Step 3A
%%% Creates an array of zeros for storing Eucludean distance.
%%%
%%% Computes Eucludean distance between every image in test_digits and the
%%% respective mean digit image in training_averages
%%%
%%% Store results in test_classification

test_classification = zeros(10, 4649);
for k = 1:10
    test_classification(k,:) = ...
        sum((test_digits - repmat(training_averages(:,k),[1 4649])).^2);
end
    
%%% Step 3B
%%% Creates an array of zeros to store classification results
%%%
%%% Computes classification results by finding the position index of the
%%% min Euclidean distance value for every value in test_classification
%%%
%%% Stores the results in test_classification_res

test_classification_res = zeros(1, 4649);
for j = 1:4649
        [tmp, ind] = min(test_classification(:,j));
        test_classification_res(:,j) = ind;
end

%%% Step 3C
%%% Creates an array of zeros to store confusion matrix
%%%
%%% Computes confusion matrix from number of non-k values from
%%% classification results
%%%
%%% Stores the results in test_confusion

test_confusion = zeros(10, 10);
for k = 1:10
    tmp = test_classification_res(test_labels(k,:) == 1);
    for j = 1:10
        test_confusion(k, j) = sum(tmp == j);
    end
end

%%%%%%%%%%%%%%%%
%%%% STEP 4 %%%%
%%%%%%%%%%%%%%%%

%%% Step 4A
%%% Creates an array of zeros to store left singular values
%%%
%%% Pool all images corresponding to kth digit in array and computes the
%%% left singular vectors in a rank 17 SVD of image set
%%%
%%% Stores the results in left_singular_vectors

left_singular_vectors = zeros(256, 17, 10);
for k = 1:10
    [left_singular_vectors(:,:,k), ~, ~] = ...
        svds(training_digits(:,training_labels(k,:) == 1), 17);
end

%%% Step 4B
%%% Creates an array of zeros to store SVD results
%%%
%%% Computes expansion coefficients for each digit image WRT to the k
%%% singular vectors (here k = 17)
%%%
%%% Stores the results in test_svd17

test_svd17 = zeros(17, 4649, 10);
for k = 1:10
    test_svd17(:,:,k) = ...
        left_singular_vectors(:,:,k)' * test_digits;
end

%%% Step 4C
%%% Create an array of zeros to store approximations (errors)
%%%
%%% Computes error between each original test image and it's rank 17
%%% approximation
%%%
%%% Stores the results in test_digits_rank17_approximation
%%%
%%% Computes classification results by finding the position index of the 
%%% min error value in test_digits_rank17_approximation
%%%
%%% Stores the result in svd_classification

test_digits_rank_17_approximation = zeros(10, 4649);
svd_classification = zeros(10, 4649);
for k = 1:10
    for j = 1:4649
        tmp = norm(test_digits(:,j) - left_singular_vectors(:,:,k) * ...
            test_svd17(:,j,k));
        test_digits_rank_17_approximation(k,j) = tmp;
        
        [tmp, ind] = min(test_digits_rank_17_approximation(:,j));
        svd_classification(1,j) = ind;
    end
end

%%% Step 4D
%%% Create a matrix to store classification results
%%%
%%% Computes classification results by finding the position index of the
%%% min value for every value in test_digits_rank17_approximation
%%%
%%% Stores the results in test_classification_res

test_svd_classification_res = zeros(1, 4649);
for j = 1:4649
        [tmp, ind] = ...
            min(test_digits_rank_17_approximation(:,j));
        test_svd_classification_res(:,j) = ind;
end

%%% Create a matrix to store confusion matrix
%%%
%%% Computes confusion matrix from number of non-k values from
%%% classification results
%%%
%%% Stores the results in test_svd17_confusion

test_svd17_confusion = zeros(10, 10);
for k = 1:10
    tmp = test_svd_classification_res(test_labels(k,:) == 1);
    for j = 1:10
        test_svd17_confusion(k, j) = sum(tmp == j);
    end
end

%%% Outputs the two confusion matrices for k-Means and SVD methods to the
%%% terminal

display("The Confusion Matrix for k-Means is:")
display("Note: The first column/row is for 0 and the last is for 9")
display("Note: Columns contain the predicted and rows contain the actual")
display(test_confusion)
display("The Confusion Matrix for SVD is:")
display("Note: The first column/row is for 0 and the last is for 9")
display("Note: Columns contain the predicted and rows contain the actual")
display(test_svd17_confusion)
