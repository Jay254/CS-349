import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from Regularnn import RegularizedMCE_FF, make_regularized_MCE_FF
# from graph import draw_node_outputs_adapted
from matplotlib.colors import ListedColormap


def plot_decision_surface(reg_model, unreg_model, test_file):
    """Plot decision surfaces for regularized and unregularized models"""
    # Load test data
    test_data = pd.read_csv(test_file)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Create figure for decision surfaces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Common bounds
    min_x, max_x = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    min_y, max_y = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

    # Create grid
    x1grid = np.linspace(min_x, max_x, 200)
    x2grid = np.linspace(min_y, max_y, 200)
    xx, yy = np.meshgrid(x1grid, x2grid)
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    # Plot regularized model
    red_color = "#EE6B6B"
    blue_color = "#4682B4"

    # Plot regularized model
    zz_reg = reg_model.predict(grid).reshape(xx.shape)
    zz_reg = 1 - zz_reg  # Invert the predictions
    ax1.contourf(xx, yy, zz_reg, cmap=plt.cm.RdBu, alpha=0.7)
    ax1.scatter(
        X_test[y_test == 1, 0], X_test[y_test == 1, 1], c=red_color, label="Class 1"
    )
    ax1.scatter(
        X_test[y_test == 0, 0], X_test[y_test == 0, 1], c=blue_color, label="Class 0"
    )
    ax1.set_title("Regularized Model")
    ax1.legend()

    # Plot unregularized model
    zz_unreg = unreg_model.predict(grid).reshape(xx.shape)
    zz_unreg = 1 - zz_unreg  # Invert the predictions
    ax2.contourf(xx, yy, zz_unreg, cmap=plt.cm.RdBu, alpha=0.7)
    ax2.scatter(
        X_test[y_test == 1, 0], X_test[y_test == 1, 1], c=red_color, label="Class 1"
    )
    ax2.scatter(
        X_test[y_test == 0, 0], X_test[y_test == 0, 1], c=blue_color, label="Class 0"
    )
    ax2.set_title("Unregularized Model")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def train_and_visualize(
    dataset_name,
    k=5,
    epochs=100,
    learning_rate=0.04,
    weight_decay=0.001,
    orthogonal_penalty=0.001,
):
    """
    Train both regularized and unregularized models and visualize their performance
    """
    # File paths
    train_file = f"{dataset_name}_train.csv"
    valid_file = f"{dataset_name}_valid.csv"
    test_file = f"{dataset_name}_test.csv"

    # Train models
    reg_model, unreg_model, history = make_regularized_MCE_FF(
        k=k,
        epoch=epochs,
        learning_rate=learning_rate,
        train_file=train_file,
        valid_file=valid_file,
        weight_decay=weight_decay,
        orthogonal_penalty=orthogonal_penalty,
    )

    # Plot learning curves
    plt.figure(figsize=(15, 5))

    # Training and validation losses
    plt.subplot(1, 3, 1)
    plt.plot(history["reg_train_losses"], label="Reg Train")
    plt.plot(history["reg_valid_losses"], label="Reg Valid")
    plt.plot(history["unreg_train_losses"], label="Unreg Train")
    plt.plot(history["unreg_valid_losses"], label="Unreg Valid")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Weight norm over time
    plt.subplot(1, 3, 2)
    plt.plot(history["weight_norms"])
    plt.title("Weight Norm Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")

    # Orthogonality over time
    plt.subplot(1, 3, 3)
    plt.plot(history["orthogonality"])
    plt.title("Orthogonality Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Orthogonality Measure")

    plt.tight_layout()
    plt.show()

    # Plot decision surfaces
    plot_decision_surface(reg_model, unreg_model, test_file)

    # Plot node outputs for both models
    print("\nRegularized Model Node Outputs:")
    draw_node_outputs_adapted(k, reg_model, test_file)
    print("\nUnregularized Model Node Outputs:")
    draw_node_outputs_adapted(k, unreg_model, test_file)

    return reg_model, unreg_model, history

def draw_node_outputs_adapted(nodes, model, test_file):
    """
    Visualizes individual node outputs using the same color scheme as the original plot
    """
    # Read test data
    test_data = pd.read_csv(test_file)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Initialize the figure - now just for nodes, no composite
    fig, axes = plt.subplots(1, nodes, figsize=(15, 5))

    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Prepare grid data
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Get hidden layer outputs
    with torch.no_grad():
        hidden_outputs = model.hidden_layer(grid)

        # Plot individual node outputs
        for i in range(nodes):
            Z_node = hidden_outputs[:, i].numpy().reshape(xx.shape)
            axes[i].contourf(xx, yy, Z_node, cmap=plt.cm.RdBu, alpha=0.6)
            axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu)
            axes[i].set_title(f"Node {i}")

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Run for XOR dataset with best parameters
    reg_model, unreg_model, history = train_and_visualize(
        dataset_name="two_gaussians",
        k=5,
        epochs=100,
        learning_rate=0.04,
        weight_decay=0.001,
        orthogonal_penalty=0.001,
    )

    # Print final accuracies
    test_data = pd.read_csv("two_gaussians_test.csv")
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    reg_accuracy = reg_model.calculate_accuracy(X_test, y_test)
    unreg_accuracy = unreg_model.calculate_accuracy(X_test, y_test)

    print(f"\nFinal Test Accuracies:")
    print(f"Regularized Model: {reg_accuracy:.4f}")
    print(f"Unregularized Model: {unreg_accuracy:.4f}")

