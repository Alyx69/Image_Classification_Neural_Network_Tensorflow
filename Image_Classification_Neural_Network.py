## *** Image Classification  Using a Convolutional Neural Network ***

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


##To load the data(images)
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()


##To scale data between 0 and 1
training_images, testing_images = training_images / 255, testing_images / 255

##To assign names to labels
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


##To display an overview of the data
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()


##To reduce the amount of images that we are feeding to neuarl network
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


##To build the neural network
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuacry: {accuracy}")

model.save('image_classifier.model')


##To load the model
model = models.load_model('image_classifier.model')


##To resize the image to 32*32
image = Image.open('image.jpg')
new_image = image.resize((32, 32))
new_image.save('new_image.jpg')


image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image, cmap=plt.cm.binary)
plt.show()

print('\n')

img = cv2.imread('new_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)
plt.show()


prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")