% Train the modified ResNet-50 on AI vs. Human dataset
% -----------------------------------------------------

% Define training options
clear;
load('matlab.mat'); % Load saved dataset variables

options = trainingOptions('adam', ...  % Optimizer (Adam)
    'InitialLearnRate', 1e-4, ...      % Learning rate
    'MaxEpochs', 10, ...               % Number of epochs
    'MiniBatchSize', 32, ...           % Batch size
    'Shuffle', 'every-epoch', ...      % Shuffle data every epoch
    'ValidationFrequency', 10, ...     % Validate every 10 iterations
    'Verbose', true, ...
    'Plots', 'training-progress');     % Show training progress

% Train the network
trainedNet = trainNetwork(augmentedImdsAug, lgraph, options);

% Save the trained network
save('trained_resnet50.mat', 'trainedNet');