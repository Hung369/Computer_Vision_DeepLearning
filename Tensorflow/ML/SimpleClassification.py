import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":
    data = load_breast_cancer()
    print("All features name in this data: ")
    print(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    number, feat = X_train.shape

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(feat,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    r=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

    print("\n Model train score: ", model.evaluate(X_train, y_train))
    print("\n Model test score: ", model.evaluate(X_test, y_test))

    plt.plot(r.history['accuracy'])
    plt.plot(r.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
# print(tf.__version__)