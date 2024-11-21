import torch
from torch import nn
import pandas as pd
import numpy as np

class RegularizedMCE_FF(torch.nn.Module):
    def __init__(self, k):
        super(RegularizedMCE_FF, self).__init__()
        self.k = k
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=k),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Linear(in_features=k, out_features=2)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not torch.is_tensor(x):
            raise ValueError("Input must be a PyTorch tensor or NumPy array.")

        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        return predicted_class.detach().cpu().numpy()

    def get_regularization_terms(self):
        weight_norm = torch.norm(self.hidden_layer[0].weight)

        x1 = torch.linspace(-5, 5, 100)
        x2 = torch.linspace(-5, 5, 100)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)

        hidden_outputs = self.hidden_layer(grid_points)
        decisions = (hidden_outputs > 0.5).float() * 2 - 1

        N = decisions.shape[1]
        orthogonality = 0
        for i in range(N):
            for j in range(i + 1, N):
                dot_product = torch.sum(decisions[:, i] * decisions[:, j])
                orthogonality += (dot_product / decisions.shape[0]) ** 2

        return weight_norm, orthogonality

    def calculate_accuracy(self, features, labels):
        predictions = self.predict(features)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        correct = (predictions == labels).sum()
        total = len(labels)
        accuracy = correct / total

        return accuracy


def find_optimal_params(
    k_list,
    epoch_list,
    learning_list,
    data_file_list,
    weight_decay=0.001,
    orthogonal_penalty=0.001,
):
    all_results = {}

    for train_data, valid_data in data_file_list:
        # Load validation data once
        valid_data_df = pd.read_csv(valid_data)
        valid_labels = valid_data_df.iloc[:, 0].to_numpy()
        valid_features = valid_data_df.iloc[:, 1:].to_numpy()

        for k in k_list:
            for epoch in epoch_list:
                for rate in learning_list:
                    # Train model with these parameters
                    reg_model, unreg_model, history = make_regularized_MCE_FF(
                        k=k,
                        epoch=epoch,
                        learning_rate=rate,
                        train_file=train_data,
                        valid_file=valid_data,
                        weight_decay=weight_decay,
                        orthogonal_penalty=orthogonal_penalty,
                    )

                    # Store accuracies
                    reg_accuracy = reg_model.calculate_accuracy(
                        valid_features, valid_labels
                    )
                    unreg_accuracy = unreg_model.calculate_accuracy(
                        valid_features, valid_labels
                    )

                    all_results[(k, epoch, rate)] = {
                        "reg_accuracy": reg_accuracy,
                        "unreg_accuracy": unreg_accuracy,
                        "history": history,
                    }

                    print(f"(k={k}, epoch={epoch}, learning_rate={rate})")
                    print(f"Regularized Accuracy: {reg_accuracy:.4f}")
                    print(f"Unregularized Accuracy: {unreg_accuracy:.4f}\n")

    return all_results


def make_regularized_MCE_FF(
    k,
    epoch,
    learning_rate,
    train_file,
    valid_file,
    weight_decay=0.001,
    orthogonal_penalty=0.001,
):
    # Load training data
    train_data = pd.read_csv(train_file)
    train_features = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

    # Load validation data
    valid_data = pd.read_csv(valid_file)
    valid_features = torch.tensor(valid_data.iloc[:, 1:].values, dtype=torch.float32)
    valid_labels = torch.tensor(valid_data.iloc[:, 0].values, dtype=torch.long)

    # Initialize models
    reg_model = RegularizedMCE_FF(k=k)
    unreg_model = RegularizedMCE_FF(k=k)

    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    reg_optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)
    unreg_optimizer = torch.optim.SGD(unreg_model.parameters(), lr=learning_rate)

    # Track losses
    reg_train_losses = []
    reg_valid_losses = []
    unreg_train_losses = []
    unreg_valid_losses = []
    weight_norms = []
    orthogonality_values = []

    for rising_epoch in range(epoch):
        # Training
        reg_outputs = reg_model(train_features)
        reg_loss = criterion(reg_outputs, train_labels)
        weight_norm, orthogonality = reg_model.get_regularization_terms()
        reg_total_loss = (
            reg_loss + weight_decay * weight_norm + orthogonal_penalty * orthogonality
        )

        reg_optimizer.zero_grad()
        reg_total_loss.backward()
        reg_optimizer.step()

        # Train unregularized model
        unreg_outputs = unreg_model(train_features)
        unreg_loss = criterion(unreg_outputs, train_labels)

        unreg_optimizer.zero_grad()
        unreg_loss.backward()
        unreg_optimizer.step()

        # Calculate validation loss
        with torch.no_grad():
            reg_valid_outputs = reg_model(valid_features)
            reg_valid_loss = criterion(reg_valid_outputs, valid_labels)

            unreg_valid_outputs = unreg_model(valid_features)
            unreg_valid_loss = criterion(unreg_valid_outputs, valid_labels)

        # Record losses
        reg_train_losses.append(reg_loss.item())
        reg_valid_losses.append(reg_valid_loss.item())
        unreg_train_losses.append(unreg_loss.item())
        unreg_valid_losses.append(unreg_valid_loss.item())
        weight_norms.append(weight_norm.item())
        orthogonality_values.append(orthogonality.item())

    return (
        reg_model,
        unreg_model,
        {
            "reg_train_losses": reg_train_losses,
            "reg_valid_losses": reg_valid_losses,
            "unreg_train_losses": unreg_train_losses,
            "unreg_valid_losses": unreg_valid_losses,
            "weight_norms": weight_norms,
            "orthogonality": orthogonality_values,
        },
    )


if __name__ == "__main__":
    # Example usage
    k_list = [2, 3, 5, 7, 9]
    epoch_list = [10, 25, 50, 75, 100]
    learning_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    data_files = [("xor_train.csv", "xor_valid.csv")]

    results = find_optimal_params(k_list, epoch_list, learning_list, data_files)
