import pathlib
import tensorflow as tf
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
import os
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # to shut tf up

# hyperparams
config = {
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "activation": "relu",
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
    "imgHeight": 240,
    "imgWidth": 135
}

def load_images(dataDir):
    # dataDir = "/home/media/simtoon/DATA/dataset/dataset/images/processed/"
    dataDir = pathlib.Path("data")
    imageCount = len(list(dataDir.glob("*.jpg")))
    print(imageCount)

    return dataDir

def check_dataset(dataDir):
    niche = list(dataDir.glob("niche/*"))
    nonNiche = list(dataDir.glob("non_niche/*"))
    print(len(niche))
    print(len(nonNiche))

def setup_wandb():
    wandb.init(project="niche-classifier", config=config)

def setup_datasets(dataDir):
    trainDS = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.15,
        subset="training",
        seed=69,
        image_size=(config["imgHeight"], config["imgWidth"]),
        batch_size=config["batch_size"])

    valDS = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.15,
        subset="validation",
        seed=69,
        image_size=(config["imgHeight"], config["imgWidth"]),
        batch_size=config["batch_size"])

    classNames = trainDS.class_names
    numClasses = len(classNames)

    return trainDS, valDS, classNames, numClasses

def setup_autotune(trainDS, valDS):
    AUTOTUNE = tf.data.AUTOTUNE
    trainDS = trainDS.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valDS = valDS.cache().prefetch(buffer_size=AUTOTUNE)

    return trainDS, valDS

def normalize(trainDS):
    normLayer = layers.Rescaling(1./255)

    normTrainDS = trainDS.map(lambda x, y: (normLayer(x), y))

    return normTrainDS

def augment():
    dataAugmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(config["imgHeight"], config["imgWidth"], 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )

    return dataAugmentation


def setup_model(numClasses, dataAugmentation):
    model = Sequential([
        dataAugmentation,
        layers.Rescaling(1./255, input_shape=(config["imgHeight"], config["imgWidth"], 3)),
        layers.Conv2D(16, 3, padding=config["padding"], activation=config["activation"]),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding=config["padding"], activation=config["activation"]),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding=config["padding"], activation=config["activation"]),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding=config["padding"], activation=config["activation"]),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding=config["padding"], activation=config["activation"]),
        layers.MaxPooling2D(),
        layers.Dropout(config["dropout"]),
        layers.Flatten(),
        layers.Dense(128, activation=config["activation"]),
        layers.Dense(numClasses)
    ])

    model.compile(optimizer=config["optimizer"],
                loss=config["loss"],
                metrics=config["metrics"])

    model.summary()

    return model

def train(model, trainDS, valDS):
    trainedModel = model.fit(
        trainDS,
        validation_data=valDS,
        epochs=config["epochs"],
        callbacks=[
            WandbMetricsLogger(log_freq=10),
            WandbModelCheckpoint("ckpt/niche-classifier-{epoch}.h5"),
            tf.keras.callbacks.ModelCheckpoint(filepath="ckpt/niche-classifier-{epoch}.h5", save_best_only=True, save_weights_only=True, verbose=1, save_freq="epoch"),
            tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
            ]
        )

    return trainedModel

def report(trainedModel, model, epochs):
    acc = trainedModel.history['accuracy']
    val_acc = trainedModel.history['val_accuracy']
    print("acc: " + str(acc), "val_acc: " + str(val_acc))

    top_acc = max(trainedModel.history["accuracy"])
    top_val_acc = max(trainedModel.history["val_accuracy"])
    print("top_acc: " + str(top_acc), "top_val_acc: " + str(top_val_acc))

    loss = trainedModel.history['loss']
    val_loss = trainedModel.history['val_loss']
    print("loss: " + str(loss), "val_loss: " + str(val_loss))

    min_loss = min(trainedModel.history["loss"])
    min_val_loss = min(trainedModel.history["val_loss"])
    print("min_loss: " + str(min_loss), "min_val_loss: " + str(min_val_loss))

    epochs_range = range(epochs)
    print("epochs_range: " + str(epochs_range))

    wandb.log({"top_acc": top_acc, "top_val_acc": top_val_acc, "min_loss": min_loss, "min_val_loss": min_val_loss, "epochs": epochs})

    model.save("model.h5")
    model.save_weights("model_weights.h5")


wandb.finish()


# let's roll, m'fucker
setup_wandb()
dataDir = load_images(dataDir="")
trainDS, valDS, classNames, numClasses = setup_datasets(dataDir)
trainDS, valDS = setup_autotune(trainDS, valDS)
dataAugmentation = augment()
model = setup_model(numClasses, dataAugmentation)
trainedModel = train(model, trainDS, valDS)
report(trainedModel, model, config["epochs"])