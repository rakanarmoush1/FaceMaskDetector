import tensorflow as tf
import numpy as np
import os
import cv2
import splitfolders


# Splitting the data into a train(0.8)/test(0.2) split
splitfolders.ratio('data', output='train_test', ratio=(0.8, 0.2))

# Optimizing TensorFlow performance to my current machine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Building CNN layers (3 convolutional layers with max pooling, 1 dense layer, and a softmax output layer)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Using generator to input the training and validation images. Also, used image augmentation on the training data.
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory('train_test/train',
                      color_mode='grayscale',
                      target_size=(100, 100),
                      batch_size=128,
                      class_mode='categorical')

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
).flow_from_directory('train_test/val',
                      color_mode='grayscale',
                      target_size=(100, 100),
                      batch_size=16,
                      class_mode='categorical')

# Using checkpoints to save models
checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0,
                                                save_best_only=True, mode='auto')

# Fitting the model on the data. Saved the model in a variable to compare results
history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[checkpoint])

# Loading the best model from training
model = tf.keras.models.load_model('model-015.model')

# Loading CV2 Haar cascade classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Using webcam
source = cv2.VideoCapture(0)


labels_dict = {0: 'Wearing a mask', 1: 'Not wearing a mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Streaming from the webcam
while True:
    # Reading the image
    ret, img = source.read()
    # Rescaling the image to gray
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(grayscaled, 1.3, 5)

    # Processing the image to fit the CNN
    for (x, y, w, h) in faces:
        face_img = grayscaled[y:y + w, x:x + w]
        face_img = cv2.resize(face_img, (100, 100))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, 100, 100, 1))
        result = model.predict(face_img)

        label = np.argmax(result, axis=1)[0]

        # Size of rectangle border around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    # Break when the esc key is pressed
    if key == 27:
        break

cv2.destroyAllWindows()
source.release()
