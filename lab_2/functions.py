import numpy as np
from scipy.ndimage import rotate


def partial_gradient(w, w0, example_train, label_train):
    """
    Computes the derivatives of the partial loss Li with respect to each of the classifier parameters.

    Parameters:
    - w: weight vector (same shape as example_train)
    - w0: bias term (scalar)
    - example_train: input image / example (same shape as w)
    - label_train: 0 or 1 (negative or positive example)

    Returns:
    - wgrad: gradient with respect to w
    - w0grad: gradient with respect to w0
    """

    # Write your code here
    example_train_flat = example_train.flatten()
    w_flat = w.flatten()
    output = np.dot(example_train_flat, w_flat) + w0
    prob = np.exp(output) / (1 + np.exp(output))

    wgrad = (prob - label_train) * example_train
    w0grad = prob - label_train
    return wgrad, w0grad


def process_epoch(w, w0, lrate, examples_train, labels_train, random_order=True):
    """
    Performs one epoch of stochastic gradient descent.

    Parameters:
    - w: weight array (same shape as examples)
    - w0: bias term (scalar)
    - lrate: learning rate (scalar)
    - examples_train: list or array of training examples (e.g., shape (N, 35, 35))
    - labels_train: array of labels (shape (N,))

    Returns:
    - Updated w and w0 after one epoch
    """

    # Write your code here
    num_examples = len(examples_train)

    if random_order:
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        examples_train = examples_train[indices]
        labels_train = labels_train[indices]

    for i in range(num_examples):
        wgrad, w0grad = partial_gradient(w, w0, examples_train[i], labels_train[i])
        w = w - lrate * wgrad
        w0 = w0 - lrate * w0grad

    return w, w0


def classify(examples_val, w, w0):
    """
    Applies a classifier to the example data.

    Parameters:
    - examples_val: List of validation examples (each example is a 1D array)
    - w: weight array (same shape as each example in examples_val)
    - w0: bias term (scalar)

    Returns:
    - predicted_labels: Array of predicted labels (0 or 1) for each example
    """

    # Write your code here
    w_flatten = w.flatten()
    predicted_labels = []

    for example in examples_val:
        example_flatten = example.flatten()
        pred = np.dot(w_flatten, example_flatten) + w0
        prob = np.exp(pred) / (1 + np.exp(pred))
        label = 1 if prob > 0.5 else 0
        predicted_labels.append(label)
    return np.array(predicted_labels)


def augment_data(examples_train, labels_train, M):
    """
    Data augmentation: Takes each sample of the original training data and
    applies M random rotations, which result in M new examples.

    Parameters:
    - examples_train: List of training examples (each example is a 2D array)
    - labels_train: Array of labels corresponding to the examples
    - M: Number of random rotations to apply to each training example

    Returns:
    - examples_train_aug: Augmented examples after rotations
    - labels_train_aug: Corresponding labels for augmented examples
    """
    # Write your code here
    examples_train_aug = []
    labels_train_aug = []
    for example, label in zip(examples_train, labels_train):
        for _ in range(M):
            angle = np.random.uniform(-180, 180)
            rotated_example = rotate(example, angle, reshape=False)
            examples_train_aug.append(rotated_example)
            labels_train_aug.append(label)

    examples_train_aug = np.array(examples_train_aug)
    labels_train_aug = np.array(labels_train_aug)

    return examples_train_aug, labels_train_aug

import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(w, sv):
    # Round weights to two decimal places and format in scientific notation
    rounded_w = np.round(w, 2)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(rounded_w, cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap, label='Weight Magnitude')

    # Add title and labels
    plt.title(f"Heatmap of Weights (s = {sv})")
    plt.xlabel("Weight Index (Columns)")
    plt.ylabel("Weight Index (Rows)")

    # Display the heatmap
    plt.tight_layout()
    plt.show()

