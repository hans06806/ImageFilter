% Define dataset folder path
datasetFolder = fullfile('/MATLAB Drive', 'Dataset');

% Create datastores for each category (Human and AI)
humanImds = imageDatastore(fullfile(datasetFolder, 'Human'), 'LabelSource', 'foldernames');
aiImds = imageDatastore(fullfile(datasetFolder, 'AI'), 'LabelSource', 'foldernames');

% Combine both datasets
combinedImds = imageDatastore([humanImds.Files; aiImds.Files], ...
    'Labels', [humanImds.Labels; aiImds.Labels]);

% Display the number of images in each category
labelCount = countEachLabel(combinedImds);
disp(labelCount);

%% Preview random images from the original dataset
perm = randperm(numel(combinedImds.Files), 9);
figure;
for i = 1:9
    subplot(3,3,i);
    img = readimage(combinedImds, perm(i));
    imshow(img);
    title(string(combinedImds.Labels(perm(i))));
end

%% Define target image size and create an augmentedImageDatastore for resizing
imageSize = [224 224]; % Resize to 224x224 (adjust as needed)
augmentedImdsResize = augmentedImageDatastore(imageSize, combinedImds);

%% Preview preprocessed (resized) images
reset(augmentedImdsResize);
figure;
for i = 1:9
    subplot(3,3,i);
    data = read(augmentedImdsResize);
    % Unwrap nested cell arrays or tables until a numeric array is obtained
    while iscell(data) || istable(data)
        if istable(data)
            data = data{1,1};  % Extract from table
        elseif iscell(data)
            data = data{1};    % Extract from cell array
        end
    end
    img = data;
    imshow(img);
    title(string(combinedImds.Labels(i)));
end

%% Define data augmentation settings
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...  % Rotate between -20 to +20 degrees
    'RandXReflection', true, ...    % Flip images horizontally
    'RandXTranslation', [-5 5], ...   % Shift images horizontally
    'RandYTranslation', [-5 5]);      % Shift images vertically

%% Create augmentedImageDatastore with augmentation settings
augmentedImdsAug = augmentedImageDatastore(imageSize, combinedImds, 'DataAugmentation', imageAugmenter);

%% Preview augmented images
reset(augmentedImdsAug);
figure;
for i = 1:9
    subplot(3,3,i);
    data = read(augmentedImdsAug);
    % Unwrap nested cell arrays or tables until a numeric array is obtained
    while iscell(data) || istable(data)
        if istable(data)
            data = data{1,1};
        elseif iscell(data)
            data = data{1};
        end
    end
    img = data;
    imshow(img);
    title(string(combinedImds.Labels(i)));
end

save('matlab.mat', 'augmentedImdsAug', 'augmentedImdsResize', 'combinedImds');
