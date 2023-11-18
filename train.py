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
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # to shut tf up

# hyperparams
config = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.00001,
    "dropout": 0.2,
    "activation": "relu",
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
    "imgHeight": 135,
    "imgWidth": 240,
    "padding": "valid"
}

def load_images(dataDir):
    dataDir = pathlib.Path(dataDir)
    print(f"Data directory: {dataDir}")
    check_dataset(dataDir)

    return dataDir

def check_dataset(dataDir):
    niche = list(dataDir.glob("like/*"))
    nonNiche = list(dataDir.glob("dislike/*"))
    print(f"niche: {len(niche)}")
    print(f"non-niche: {len(nonNiche)}")

def setup_wandb():
    wandb.init(project="niche-classifier", config=config, entity="simtoonia")

def setup_datasets(dataDir):
    trainDS, valDS = tf.keras.preprocessing.image_dataset_from_directory(
        dataDir,
        validation_split=0.15,
        subset="both",
        shuffle=True,
        seed=69,
        label_mode="int",
        color_mode="rgb",
        image_size=(config["imgHeight"], config["imgWidth"]),
        batch_size=config["batch_size"]
    )


    classNames = trainDS.class_names
    numClasses = len(classNames)

    plt.figure(figsize=(10, 10))
    for images, labels in trainDS.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classNames[int(labels[i])])
            plt.axis("off")

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
        layers.Conv2D(16, 3, padding=config["padding"], activation=config["activation"], input_shape=(config["imgHeight"], config["imgWidth"], 3)),
        layers.BatchNormalization(),
        layers.Conv2D(16, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding=config["padding"], activation=config["activation"]),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(config["dropout"]),
        layers.Flatten(),
        layers.Dense(256, activation="softmax"),
        layers.Dense(128, activation="softmax"),
        layers.Dense(numClasses)
    ])

    model.build((None, config["imgHeight"], config["imgWidth"], 3))

    model.compile(optimizer=config["optimizer"],
                loss=config["loss"],
                metrics=config["metrics"])

    model.summary()

    return model

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def train(model, trainDS, valDS):
    trainedModel = model.fit(
        trainDS,
        validation_data=valDS,
        epochs=config["epochs"],
        callbacks=[
            WandbMetricsLogger(log_freq=10),
            WandbModelCheckpoint("ckpt/niche-classifier-{epoch}.keras"),
            tf.keras.callbacks.ModelCheckpoint(filepath="ckpt/niche-classifier-{epoch}.keras", save_best_only=True, save_weights_only=True, verbose=1, save_freq="epoch"),
            tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1),
            tf.keras.callbacks.LearningRateScheduler(scheduler)
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

    model.save("model.keras")
    model.save_weights("model_weights.keras")


wandb.finish()


# let's roll, m'fucker
setup_wandb()
dataDir = load_images(dataDir="/home/simtoon/git/niche")
trainDS, valDS, classNames, numClasses = setup_datasets(dataDir)
imageBatch, labelsBatch = next(iter(trainDS))
firstImage = imageBatch[0]
print(np.min(firstImage), np.max(firstImage))
trainDS = normalize(trainDS)
imageBatch, labelsBatch = next(iter(trainDS))
firstImage = imageBatch[0]
print(np.min(firstImage), np.max(firstImage))
trainDS, valDS = setup_autotune(trainDS, valDS)
dataAugmentation = augment()

def display_images(dataset, title, num_images=3):
    plt.figure(figsize=(20, 7))
    for images, labels in dataset.take(1):
        random_indices = random.sample(range(images.shape[0]), num_images)
        for i, index in enumerate(random_indices):
            ax = plt.subplot(1, num_images, i + 1)
            if (dataset == trainDS):
                plt.imshow(images[index].numpy().astype("uint8") * 255)
            else:
                plt.imshow(images[index].numpy().astype("uint8"))
            # print filename, too, from the prefetch dataset
            plt.title(f"{title}: {classNames[int(labels[index].numpy())]}\nFilename: ")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

display_images(trainDS, "train")
display_images(valDS, "val")

model = setup_model(numClasses, dataAugmentation)
trainedModel = train(model, trainDS, valDS)
report(trainedModel, model, config["epochs"])