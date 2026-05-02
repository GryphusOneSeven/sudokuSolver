import tensorflow
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_MNIST(path="model/digit_cnn.keras"):
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test  = x_test[..., None]

    model = build_model()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    model.save(path)

def train_from_folder():
    datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
    )

    train_gen = datagen.flow_from_directory(
        "dataset",
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=32,
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        "dataset",
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=32,
        subset="validation"
    )

    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save("model/digit_cnn_sudoku.keras")

if __name__ == "__main__":
    train_from_folder()
