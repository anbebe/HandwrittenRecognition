import tensorflow as tf
import tqdm

from model import ConvModel
from data import load_data, preprocess_data
import numpy as np
import matplotlib.pyplot as plt



def training_loop(model, train_ds, val_ds, epochs, save_path):
    train_history = {"loss":[], "acc":[]}
    val_history = {"val_loss":[], "val_acc":[]}
    #1. iterate over epochs
    for e in range(epochs):
        #2. train steps on all batchs in the training data
        for data in tqdm.tqdm(train_ds):
            metrics = model.train_step(data)
        # 3. log and print data metrics
        for key, value in metrics.items():
            train_history[key].append(value)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        #4 reset the metrics
        model.reset_metrics()

        #5. evaluate on validation data
        for data in val_ds:
            metrics = model.test_step(data)
    
        #6. log validation metrics
        for key, value in metrics.items():
            val_history[key].append(value)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])
    
        #7. reset metric objects
        model.reset_metrics()

    #8 save model weights
    model.save_weights(save_path)
    return train_history, val_history


if __name__ == "__main__":
    train_ds, val_ds = load_data()
    train_ds = preprocess_data(train_ds)
    val_ds = preprocess_data(val_ds)

    #1. instantiate model
    model = ConvModel()
    #model(tf.keras.Input((32,32,3)))
    #model.summary()
    epochs=30

    #2. choose a path to save the weights
    save_path = "trained_model_var"

    train_history, val_history = training_loop(model, train_ds, val_ds, epochs, save_path)

    plt.plot(val_history["val_loss"])
    plt.plot(train_history["loss"])
    plt.legend(labels=["Validation Loss", "Loss"])
    plt.savefig("convnet__loss_cnn_opt_reg_40.svg")
    plt.show()

    plt.plot(val_history["val_acc"])
    plt.plot(train_history["acc"])
    plt.legend(labels=["Validation Accuracy", "Accuracy"])
    plt.savefig("convnet__acc_cnn_opt_reg_40.svg")
    plt.show()