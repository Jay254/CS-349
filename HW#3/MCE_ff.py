import torch
from torch import nn
import pandas as pd
import numpy as np
import graph
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class RegularizedMCE_FF(torch.nn.Module):
    def __init__(self, k):
        super(RegularizedMCE_FF, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=k),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Linear(in_features=k, out_features=2)

    def forward(self, x):
        # Pass through the hidden layer
        x = self.hidden_layer(x)
        # Pass through the output layer
        x = self.output_layer(x)
        return x

    def predict(self, x):
        # Ensure the input is a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not torch.is_tensor(x):
            raise ValueError("Input must be a PyTorch tensor or NumPy array.")

        # Perform forward pass
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        # Return as NumPy array for compatibility with plotting
        return predicted_class.detach().cpu().numpy()

    def calculate_accuracy(self, features, labels):
        """
        Calculates the accuracy of the model given features and true labels.

        Args:
            features: Input features (NumPy array or PyTorch tensor).
            labels: True labels (NumPy array or PyTorch tensor).

        Returns:
            accuracy (float): The accuracy as a value between 0 and 1.
        """
        # Get predictions
        predictions = self.predict(features)

        # Ensure labels are NumPy arrays for comparison
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Calculate accuracy
        correct = (predictions == labels).sum()
        total = len(labels)
        accuracy = correct / total

        return accuracy

    def get_regularization_terms(self):
        # L2 norm of input layer weights
        weight_norm = torch.norm(self.hidden_layer[0].weight)

        # Orthogonality of decision boundaries
        # Get a grid of points to evaluate decision boundaries
        x1 = torch.linspace(-5, 5, 100)
        x2 = torch.linspace(-5, 5, 100)
        X1, X2 = torch.meshgrid(x1, x2)
        grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)

        # Get hidden layer outputs (decision boundaries)
        hidden_outputs = self.hidden_layer(grid_points)
        # Convert to binary decisions (-1 or 1)
        decisions = (hidden_outputs > 0.5).float() * 2 - 1

        # Calculate dot products between decision boundaries
        N = decisions.shape[1]  # number of hidden nodes
        orthogonality = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Dot product between decision boundaries i and j
                dot_product = torch.sum(decisions[:, i] * decisions[:, j])
                orthogonality += (dot_product / decisions.shape[0]) ** 2

        return weight_norm, orthogonality


def make_regularized_MCE_FF(
    k, epoch, learning_rate, file_name, weight_decay, orthogonal_penalty
):
    # Load and prepare data
    cur_train_data = pd.read_csv(file_name)
    labels = cur_train_data.iloc[:, 0]
    features = cur_train_data.iloc[:, 1:]

    # Convert to PyTorch tensors
    labels_tensor = torch.tensor(labels.to_numpy(), dtype=torch.long)
    features_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)

    # Create train tensor
    train_tensor = [
        [label, feature] for label, feature in zip(labels_tensor, features_tensor)
    ]

    # Initialize models
    reg_model = RegularizedMCE_FF(k=k)
    unreg_model = RegularizedMCE_FF(k=k)

    # Initialize loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    reg_optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)
    unreg_optimizer = torch.optim.SGD(unreg_model.parameters(), lr=learning_rate)

    # Track losses
    reg_pure_losses = []
    reg_total_losses = []
    unreg_losses = []
    weight_norms = []
    orthogonality_values = []

    for rising_epoch in range(epoch):
        reg_pure_loss = 0.0
        reg_total_loss = 0.0
        unreg_total_loss = 0.0

        for item in train_tensor:
            feature = item[1].unsqueeze(0)
            actual_result = torch.tensor([item[0]])

            # Train regularized model
            reg_prediction = reg_model(feature)
            reg_loss = criterion(reg_prediction, actual_result)
            weight_norm, orthogonality = reg_model.get_regularization_terms()

            # Store pure loss before adding regularization terms
            reg_pure_loss += reg_loss.item()

            # Add regularization terms for training
            reg_full_loss = (
                reg_loss
                + weight_decay * weight_norm
                + orthogonal_penalty * orthogonality
            )

            reg_optimizer.zero_grad()
            reg_full_loss.backward()
            reg_optimizer.step()

            # Train unregularized model
            unreg_prediction = unreg_model(feature)
            unreg_loss = criterion(unreg_prediction, actual_result)

            unreg_optimizer.zero_grad()
            unreg_loss.backward()
            unreg_optimizer.step()

            reg_total_loss += reg_full_loss.item()
            unreg_total_loss += unreg_loss.item()

        # Record metrics
        reg_pure_losses.append(reg_pure_loss / len(train_tensor))
        reg_total_losses.append(reg_total_loss / len(train_tensor))
        unreg_losses.append(unreg_total_loss / len(train_tensor))
        weight_norms.append(weight_norm.item())
        orthogonality_values.append(orthogonality.item())

        if rising_epoch % 10 == 0:
            print(f"Epoch {rising_epoch}:")
            print(f"  Regularized Pure Loss: {reg_pure_loss/len(train_tensor):.4f}")
            print(f"  Regularized Total Loss: {reg_total_loss/len(train_tensor):.4f}")
            print(f"  Unregularized Loss: {unreg_total_loss/len(train_tensor):.4f}")
            print(f"  Weight Norm: {weight_norm.item():.4f}")
            print(f"  Orthogonality: {orthogonality.item():.4f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(reg_pure_losses, label="Regularized (Pure Loss)")
    plt.plot(reg_total_losses, label="Regularized (Total Loss)")
    plt.plot(unreg_losses, label="Unregularized")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(weight_norms)
    plt.title("Weight Norm")

    plt.subplot(1, 3, 3)
    plt.plot(orthogonality_values)
    plt.title("Orthogonality")

    plt.tight_layout()
    plt.show()

    return (
        reg_model,
        unreg_model,
        {
            "reg_pure_losses": reg_pure_losses,
            "reg_total_losses": reg_total_losses,
            "unreg_losses": unreg_losses,
            "weight_norms": weight_norms,
            "orthogonality": orthogonality_values,
        },
    )


if __name__ == "__main__":
    reg_model, unreg_model, history = make_regularized_MCE_FF(
        k=7,
        epoch=100,
        learning_rate=0.04,
        file_name="xor_train.csv",
        weight_decay=0.001,
        orthogonal_penalty=0.001,
    )

    # Load test data
    test_data = pd.read_csv("xor_test.csv")
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Print data ranges
    print("X range:", X_test[:, 0].min(), "to", X_test[:, 0].max())
    print("Y range:", X_test[:, 1].min(), "to", X_test[:, 1].max())
    print("Number of positive examples:", sum(y_test == 1))
    print("Number of negative examples:", sum(y_test == 0))

    # Define plotting function with explicit bounds
    def plot_decision_boundaries(reg_model, unreg_model, X_test, y_test):
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Common bounds for both plots
        min_x, max_x = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        min_y, max_y = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

        # Create a finer grid for better visualization
        x1grid = np.linspace(min_x, max_x, 200)
        x2grid = np.linspace(min_y, max_y, 200)
        xx, yy = np.meshgrid(x1grid, x2grid)

        # Create grid points
        grid = np.column_stack((xx.ravel(), yy.ravel()))

        # Plot regularized model
        zz1 = reg_model.predict(grid)
        zz1 = zz1.reshape(xx.shape)

        ax1.contourf(
            xx, yy, zz1, cmap=ListedColormap(["#9ce8ff", "#ffc773"]), alpha=0.7
        )
        ax1.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label="negative")
        ax1.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label="positive")
        ax1.set_title("Regularized")
        ax1.legend()
        ax1.set_xlim([min_x, max_x])
        ax1.set_ylim([min_y, max_y])

        # Plot unregularized model
        zz2 = unreg_model.predict(grid)
        zz2 = zz2.reshape(xx.shape)

        ax2.contourf(
            xx, yy, zz2, cmap=ListedColormap(["#9ce8ff", "#ffc773"]), alpha=0.7
        )
        ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label="negative")
        ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label="positive")
        ax2.set_title("Unregularized")
        ax2.legend()
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])

        plt.tight_layout()
        plt.show()

    # Call the plotting function
    plot_decision_boundaries(reg_model, unreg_model, X_test, y_test)