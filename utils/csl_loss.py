import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class AdditionalTermLayer(nn.Module):
    def __init__(self, target_class_index, num_classes):
        super(AdditionalTermLayer, self).__init__()
        self.target_class_index = target_class_index
        self.num_classes = num_classes
        self.previous_epoch_class_predictions = None
        self.feature_storage = {i: [] for i in range(num_classes)}
        self.entropies = {i: [] for i in range(num_classes)}

    def compute_entropy(self, class_predictions, num_samples):
        """
        Compute entropy for class i based on its predictions.
        """
        probabilities = class_predictions.float() / num_samples
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs + 1e-6))  # Add small value to avoid log(0)
        return entropy.item()

    def forward(self, inputs, true_labels, epoch):
        inputs = torch.nan_to_num(inputs)  # Replace NaNs with zero
        additional_term = 0.0

        class_predictions = torch.argmax(inputs, dim=-1)

        # Store the current batch's features
        for i in range(self.num_classes):
            class_indices = (true_labels == i).nonzero(as_tuple=True)[0]
            if class_indices.size(0) > 0: 
                self.feature_storage[i].extend(inputs[class_indices].detach().cpu().numpy())

        # Calculate the semantic scale for each class
        semantic_scales = []
        for features in self.feature_storage.values():
            if len(features) > 0:
                features = np.array(features)
                avg_magnitude = np.mean(np.linalg.norm(features, axis=1))
                semantic_scale = avg_magnitude ** 2
                semantic_scales.append(semantic_scale)
            else:
                semantic_scales.append(0.0)

        # Calculate class entropies
        class_entropies = []
        num_samples = len(true_labels)
        for i in range(self.num_classes):
            class_indices = (true_labels == i).nonzero(as_tuple=True)[0]
            class_predictions_i = (class_predictions == i).float()
            entropy = self.compute_entropy(class_predictions_i, num_samples)
            self.entropies[i].append(entropy)
            class_entropies.append(entropy)

        # Calculate dynamic gamma values
        max_semantic_scale = max(semantic_scales) + 1e-6
        dynamic_gammas = [
            scale / (1e-6 + max_semantic_scale * entropy)
            for scale, entropy in zip(semantic_scales, class_entropies)
        ]

        # Calculate the number of predictions for each class
        current_epoch_class_predictions = torch.tensor([
            torch.sum((class_predictions == i).float()).item() for i in range(self.num_classes)
        ])

        # Compute the additional term
        for i, gamma in enumerate(dynamic_gammas):
            class_i_predictions = current_epoch_class_predictions[i]
            if i in self.target_class_index:
                if self.previous_epoch_class_predictions is not None:
                    previous_class_i_predictions = self.previous_epoch_class_predictions[i]
                    reinforcement_term = torch.tensor(0.0)
                    if class_i_predictions > previous_class_i_predictions:
                        reinforcement_term = -2.0
                    elif class_i_predictions < previous_class_i_predictions:
                        reinforcement_term = 2.0
                else:
                    reinforcement_term = torch.tensor(0.0)
            else:
                reinforcement_term = torch.tensor(0.0)

            term = (gamma * class_i_predictions + reinforcement_term) ** 2
            denom = torch.sum((inputs - F.one_hot(torch.tensor(i), num_classes=self.num_classes).float().to(inputs.device)) ** 2) + 1e-6  # Add small value to avoid division by zero
            additional_term += term / denom

        # Normalize the additional term
        additional_term /= self.num_classes
        self.previous_epoch_class_predictions = current_epoch_class_predictions

        return additional_term

class CSLLossFunc(nn.Module):
    def __init__(self, target_class_index, num_classes):
        super(CSLLossFunc, self).__init__()
        self.additional_term_layer = AdditionalTermLayer(target_class_index, num_classes)

    def forward(self, y_true, y_pred, epoch):
        y_true_one_hot = F.one_hot(y_true.squeeze().long(), num_classes=y_pred.size(-1)).float()
        cross_entropy_loss = F.cross_entropy(y_pred, y_true)

        additional_term = self.additional_term_layer(y_pred, y_true, epoch)
        total_loss = cross_entropy_loss + additional_term

        return total_loss