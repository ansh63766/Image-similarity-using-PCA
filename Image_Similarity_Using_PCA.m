clear all;
clc;

% Folder containing the images
folder_path = '/MATLAB Drive/New Folder';

image_size = [243, 320]; % height x width

num_images_per_subject = 6;
num_subjects = 15;

images_array = zeros(image_size(1) * image_size(2), num_images_per_subject * num_subjects); % storing like 77760*90

euclidean_distances = zeros(90, 15); % dist between one norm image and every other image in one column

% storing all images as column vectors
for subject = 1:num_subjects
    for image_idx = 1:num_images_per_subject
        % filename for the current image
        filename = sprintf('s%02d%s.png', subject, getImageSuffix(image_idx));
        
        % reading the image
        img = imread(fullfile(folder_path, filename));
        
        % greyscale image into a column vector
        img_vector = img(:);
        
        column_index = (subject - 1) * num_images_per_subject + image_idx;
        images_array(:, column_index) = img_vector;
    end
end

disp('Images have been loaded and stored as column vectors.');

% Calculating eucldean dist
for i = 1:90
    for j = 1:15
        norm_img = images_array(:, 6*j);
        image = images_array(:, i); 
        euclidean_distances(i, j) = norm(norm_img - image);
    end
end

% finding subject of minimum dist for each row
min_dist_index = zeros(90, 1);
for i = 1:90
    [~, min_index] = min(euclidean_distances(i, :));
    min_dist_index(i) = min_index;
end

data1 = cell(90, 3);
count1 = 0; % count for correctly classified without pca

sub_list = repelem(1:15, 6); 
sub_list = sub_list(1:90)'; % just storing number of subjects as first 6 are 1 and so on

for i = 1:90
    if min_dist_index(i) == sub_list(i) % if the index of estimated subject is same as actual then it is correctly classified
        data1{i, 2} = 'correct';
        count1 = count1 + 1;
    else
        data1{i, 2} = 'incorrect';
    end
    data1{i, 1} = i;
    data1{i, 3} = sub_list(i);
end

table1 = table(data1(:, 1), data1(:, 2), data1(:, 3), 'VariableNames', {'Image Number', 'Estimated_Subject', 'Actual_Subject'});

fprintf('\nWithout PCA:\n')
disp(table1);
fprintf('\nWithout PCA correctly classified images:\n')
disp(count1);

variance_percentages_1 = zeros(15, 1); % var % for 1st pc
variance_percentages_2 = zeros(15, 1); % var % for 2nd pc
weighted_images_array_1 = zeros(image_size(1) * image_size(2), num_subjects); 
weighted_images_array_2 = zeros(image_size(1) * image_size(2), num_subjects);

for subject = 1:num_subjects
    % 6 images for the current subject
    subject_images = images_array(:, (subject - 1) * num_images_per_subject + 1 : subject * num_images_per_subject);

    % mean shifting
    S = subject_images;
    Sz = S - mean(S, 1);

    % cov mat
    Szs = cov(Sz);

    % calc eig values and vectors
    [v, d] = eig(Szs);
    lambda = diag(d);

    % sorting lambda
    [sorted_lambda, sorted_idx] = sort(lambda, 'descend');

    variance_percent = (sorted_lambda(1) / sum(sorted_lambda)) * 100; % variance percentage of largest eigenvalue
    variance_percentages_1(subject) = variance_percent; 

    variance_percent = (sorted_lambda(2) / sum(sorted_lambda)) * 100; % variance percentage of 2nd largest eigenvalue
    variance_percentages_2(subject) = variance_percent;

    weights_1 = v(:, sorted_idx(1)) .^ 2;
    weights_2 = v(:, sorted_idx(2)) .^ 2;

    weighted_img_1 = S * weights_1; % compression
    weighted_img_2 = S * weights_2; % compression

    weighted_images_array_1(:, subject) = weighted_img_1;
    weighted_images_array_2(:, subject) = weighted_img_2;
end

disp('PCA has been performed and weighted images have been calculated.');

% calculating euclidean dist for only 1 rep image
euclidean_distances_for_rep_1 = zeros(90, 15);

for i = 1:90
    for j = 1:15
        rep_img = weighted_images_array_1(:, j);
        image = images_array(:, i);
        euclidean_distances_for_rep_1(i, j) = norm(rep_img - image);
    end
end

min_dist_index_for_rep_1 = zeros(90, 1);
for i = 1:90
    [~, min_index_for_rep_1] = min(euclidean_distances_for_rep_1(i, :));
    min_dist_index_for_rep_1(i) = min_index_for_rep_1;
end

data2 = cell(90, 3);
count2 = 0;

for i = 1:90
    if min_dist_index_for_rep_1(i) == sub_list(i)
        data2{i, 2} = 'correct';
        count2 = count2 + 1;
    else
        data2{i, 2} = 'incorrect';
    end
    data2{i, 1} = i;
    data2{i, 3} = sub_list(i);
end

table2 = table(data2(:, 1), data2(:, 2), data2(:, 3), 'VariableNames', {'Image Number', 'Estimated_Subject', 'Actual_Subject'});

fprintf('\nWith only 1st PC:\n')
disp(table2);
fprintf('\nWith 1 PC correctly classified images:\n')
disp(count2);

% calculating euclidean dist for 2 rep images
euclidean_distances_for_rep_2 = zeros(90, 30);

for i = 1:90
    for j = 1:15
        rep_img_1 = weighted_images_array_1(:, j);
        rep_img_2 = weighted_images_array_2(:, j);
        image = images_array(:, i);
        euclidean_distances_for_rep_2(i, j) = norm(rep_img_1 - image);
        euclidean_distances_for_rep_2(i, j+15) = norm(rep_img_2 - image);
    end
end

min_dist_index_for_rep_2 = zeros(90, 1);
for i = 1:90
    [~, min_index_for_rep_2] = min(euclidean_distances_for_rep_2(i, :));
    if min_index_for_rep_2 <= 15
        min_dist_index_for_rep_2(i) = min_index_for_rep_2;
    else
        min_dist_index_for_rep_2(i) = min_index_for_rep_2 - 15;
    end
end

data3 = cell(90, 3);
count3 = 0;

for i = 1:90
    if min_dist_index_for_rep_2(i) == sub_list(i)
        data3{i, 2} = 'correct';
        count3 = count3 + 1;
    else
        data3{i, 2} = 'incorrect';
    end
    data3{i, 1} = i;
    data3{i, 3} = sub_list(i);
end

table3 = table(data3(:, 1), data3(:, 2), data3(:, 3), 'VariableNames', {'Image Number', 'Estimated_Subject', 'Actual_Subject'});

fprintf('\nWith 2 PCs:\n')
disp(table3);
fprintf('\nWith 2 PCs correctly classified images:\n')
disp(count3);

% Calculate euclidean distances between images
euclidean_distances_all = pdist(images_array'); % Pairwise distances between columns

% Reshape distances into a square matrix
n = size(images_array, 2); % Number of images
euclidean_distances_matrix = squareform(euclidean_distances_all);

% Plot heatmap
figure;
imagesc(euclidean_distances_matrix);
colormap jet;
colorbar;
xlabel('Image Index');
ylabel('Image Index');
title('Euclidean Distance Heatmap');

function suffix = getImageSuffix(image_idx)
    switch image_idx
        case 1
            suffix = 'cl';
        case 2
            suffix = 'glass';
        case 3
            suffix = 'happy';
        case 4
            suffix = 'll';
        case 5
            suffix = 'ng';
        case 6
            suffix = 'norm';
        otherwise
            error('Invalid image index.');
    end
end
