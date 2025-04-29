clear;
load('matlab.mat', 'net'); % Load the ResNet-50 model

% Get the number of classes (AI and Human)
numClasses = 2;

% Extract the layer graph from the network
lgraph = layerGraph(net);

% Remove the last 2 layers (Fully Connected and Softmax)
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax'});

% Create a new fully connected layer with 2 output classes
newFC = fullyConnectedLayer(numClasses, 'Name', 'fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);

% Create a new softmax layer
newSoftmax = softmaxLayer('Name', 'softmax');

% Create a new classification layer
newClassLayer = classificationLayer('Name', 'output');

% Add the new layers
lgraph = addLayers(lgraph, newFC);
lgraph = addLayers(lgraph, newSoftmax);
lgraph = addLayers(lgraph, newClassLayer);

% Connect the new layers to the network
lgraph = connectLayers(lgraph, 'avg_pool', 'fc');
lgraph = connectLayers(lgraph, 'fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'output');

% Display the modified network
analyzeNetwork(lgraph);

save('matlab.mat', 'lgraph', '-append');