# Housing Price Prediction Neural Network

This project demonstrates the creation of a simple neural network using TensorFlow and Keras to predict housing prices based on the number of bedrooms.

## Project Overview

The goal is to build a model that can learn the linear relationship between a single feature (number of bedrooms) and a target (price). The model is intentionally simple to illustrate the fundamental concepts of building and training a neural network for a regression task.

### The Scenario

The training data is generated based on a straightforward rule:
- A house with 1 bedroom costs **100,000**.
- Each additional bedroom increases the cost by **50,000**.

For modeling purposes, the prices are scaled down to "hundreds of thousands" (e.g., 100,000 is represented as `1.0`).

---

## Project Structure

The project is broken down into three main exercises:

### 1. Create Training Data (`create_training_data`)

- **Objective:** Generate the features (number of bedrooms) and targets (prices) for the model to learn from.
- **Implementation:** Two NumPy arrays are created:
  - `n_bedrooms`: A float array `[1., 2., 3., 4., 5., 6.]`.
  - `price_in_hundreds_of_thousands`: A float array `[1. , 1.5, 2. , 2.5, 3. , 3.5]`.

### 2. Define and Compile the Model (`define_and_compile_model`)

- **Objective:** Define the neural network's architecture and configure it for training.
- **Implementation:** A `tf.keras.Sequential` model is created with:
  - An `Input` layer with a shape of `(1,)` to accept the single feature.
  - A `Dense` layer with a single neuron (`units=1`), which will learn the weight and bias to map the input to the output.
  - The model is compiled with the **Stochastic Gradient Descent (SGD)** optimizer and **Mean Squared Error (MSE)** as the loss function, which is suitable for regression problems.

### 3. Train the Model and Predict (`train_model`)

- **Objective:** Train the model on the generated data and then use it to make a new prediction.
- **Implementation:**
  - The `model.fit()` method is called to train the network for 500 epochs.
  - After training, the model is used to predict the price of a 7-bedroom house. The expected prediction should be very close to `4.0` (representing 400,000).

---

## Dependencies

- **TensorFlow:** For building and training the neural network.
- **NumPy:** For efficient numerical operations and data creation.

