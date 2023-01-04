import tensorflow as tf
import tqdm
from datetime import datetime

from model import ConvModel
from data import load_data, preprocess_data
import matplotlib.pyplot as plt



def training_loop(model, train_ds, val_ds, epochs, save_path, train_summary_writer, val_summary_writer):
    train_history = {"loss":[], "acc":[]}
    val_history = {"val_loss":[], "val_acc":[]}
    #1. iterate over epochs
    for e in range(epochs):
        #2. train steps on all batchs in the training data
        for data in tqdm.tqdm(train_ds):
            metrics = model.train_step(data)
        # 3. log and print data metrics
        with train_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
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
        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
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

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_path = f"logs/train/"
    val_log_path = f"logs/val/"
    train_log_dir = train_log_path + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_path + current_time)

    train_ds, val_ds = load_data(train_summary_writer, val_summary_writer)
    train_ds = preprocess_data(train_ds)
    val_ds = preprocess_data(val_ds)

    
    #1. instantiate model
    model = ConvModel(batch_norm=True, dropout=False, dropout_rate=0, L2_reg=0.01)
    model(tf.keras.Input((32,32,3)))
    model.summary()
    tf.summary.trace_on(graph=True, profiler=True)
    model(tf.zeros((1, 32, 32, 3)))
    with train_summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=train_log_dir)
    epochs=10

    #2. choose a path to save the weights
    save_path = "trained_model_opt"



    train_history, val_history = training_loop(model, train_ds, val_ds, epochs, save_path, train_summary_writer, val_summary_writer)

    plt.plot(val_history["val_loss"])
    plt.plot(train_history["loss"])
    plt.legend(labels=["Validation Loss", "Loss"])
    #plt.savefig("convnet__loss_opt_aug_30.svg")
    plt.show()

    plt.plot(val_history["val_acc"])
    plt.plot(train_history["acc"])
    plt.legend(labels=["Validation Accuracy", "Accuracy"])
    #plt.savefig("convnet__acc_opt_aug_30.svg")
    plt.show()
    