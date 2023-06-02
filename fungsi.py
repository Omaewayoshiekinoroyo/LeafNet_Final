import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.applications.mobilenet import MobileNet

def create_model():
    mobilenet = MobileNet(input_shape=(224, 224, 3),
                          include_top=False,
                          weights='imagenet')

    model = Sequential()
    model.add(mobilenet)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(4, activation="softmax", name="classification"))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()
