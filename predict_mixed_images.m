% predict_mixed_images.m
% Classify unlabeled mixed images (AI + Human) using trained ResNet-50 model

% Step 1: Load trained network
load('trained_resnet50.mat', 'trainedNet');

% Step 2: Define test folder path (must contain mixed images only)
mixedFolder = fullfile(pwd, 'MixedTest');  % Corrected path
imageSize = [224 224];                    % Image size for ResNet50

% Step 3: Load images (no labels)
mixedImds = imageDatastore(mixedFolder);

% Step 4: Resize images to model input
augImds = augmentedImageDatastore(imageSize, mixedImds);

% Step 5: Predict labels
predictedLabels = classify(trainedNet, augImds);

% Step 6: Display results
disp('Prediction Results:');
for i = 1:length(mixedImds.Files)
    fprintf('%s ==> %s\n', mixedImds.Files{i}, string(predictedLabels(i)));
end

% Step 7: Plot bar chart of prediction counts
figure;
bar(countcats(predictedLabels));
title('Prediction Summary: Human vs AI');
ylabel('Number of Images');
xlabel('Predicted Class');
xticklabels(categories(predictedLabels));
grid on;

% Step 8 (Optional): Visual preview of predictions
figure;
for i = 1:min(9, length(mixedImds.Files))
    subplot(3,3,i);
    img = readimage(mixedImds, i);
    imshow(img);
    title(string(predictedLabels(i)));
end
