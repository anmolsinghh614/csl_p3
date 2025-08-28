import torch.nn as nn
import torchvision.models as models
import torch

class ResNet32(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet32, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        # Forward pass through the model up to the penultimate layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_features:
            return self.model.fc(features), features
        
        return self.model.fc(features)

    def get_feature_dim(self):
        """Get the dimension of the feature vector."""
        return self.model.fc.in_features


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        # Forward pass through the model up to the penultimate layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_features:
            return self.model.fc(features), features
        
        return self.model.fc(features)

    def get_feature_dim(self):
        """Get the dimension of the feature vector."""
        return self.model.fc.in_features


class ResNet101(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        # Forward pass through the model up to the penultimate layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_features:
            return self.model.fc(features), features
        
        return self.model.fc(features)

    def get_feature_dim(self):
        """Get the dimension of the feature vector."""
        return self.model.fc.in_features
