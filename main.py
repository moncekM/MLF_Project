import os
from random import randint
from sklearn.model_selection import train_test_split
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import keras
import seaborn as sns

# Load the train data

folder_path = 'Train'
# Get a list of all files in the folder
files = os.listdir(folder_path)
Train_data = [np.load(os.path.join(folder_path, file)) for file in files if file.endswith('.npy')]
#add data to single array
Train_data = np.stack(Train_data, axis=0)
# import the marking for data
Train_marking = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)
#add the marking to one colum array
Train_marking = Train_marking[:, 1]

# print the shape of the data
print (Train_data.shape)
print(Train_marking.shape)
random_index = randint(0, 1491)
plt.figure()
plt.imshow(Train_data[random_index], cmap='gray', interpolation='nearest')
plt.title('Train Data' + str(random_index))
plt.show()


# Load the test data
folder_path = 'Test'
# Get a list of all files in the folder
files = os.listdir(folder_path)
Test_data = [np.load(os.path.join(folder_path, file)) for file in files if file.endswith('.npy')]
#add data to single array
Test_data = np.stack(Test_data, axis=0)
# import the marking for data
Test_marking = np.genfromtxt('test_format.csv', delimiter=',', skip_header=1) 
#add the marking to one colum array
Test_marking = Test_marking[:, 1]  

# print the shape of the data
print (Test_data.shape)
print(Test_marking.shape) 
random_index = randint(0, 121) 
plt.figure()
plt.imshow(Test_data[random_index], cmap='gray', interpolation='nearest')
plt.title('Test Data' + str(random_index))
plt.show()

#data preprocesing
#spliting the data into train and validation dataset
data_train, data_validation, markng_train, marking_validation = train_test_split(
    Train_data, Train_marking, test_size=0.2, random_state=42, stratify=Train_marking
)
#Encodeing marking data ot universal matrix to feed into model.
markng_train = keras.utils.to_categorical(markng_train, 3)
marking_validation = keras.utils.to_categorical(marking_validation, 3)
Test_markings = keras.utils.to_categorical(Test_marking, 3)

#adding the first experimental model it will be changed with more information about dataset
model = keras.models.Sequential()
#I use the Input layer as a conv 2D which devides the image to small subsection do search for diferences
model.add(keras.layers.Conv2D(32,kernel_size=3,activation='relu',padding=1))
#Adding a second convolutional layer with larger filter to pull more form Image
model.add(keras.layers.Conv2D(64,kernel_size=3,activation='relu',padding=1))



#Output layer is 3 because we have 3 classes
model.add(keras.layers.Dense(3, activation='softmax'))

# Compileing the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
# Train the model
history = model.fit(
    data_train,
    markng_train,
    epochs=10,
    batch_size=32,
    validation_data=(data_validation, marking_validation)
)
# Evaluate the model
test_loss, test_accuracy = model.evaluate(Test_data, Test_markings)
print('Test accuracy:', test_accuracy)
print ('Test loss:', test_loss)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

#plot the confusion matrix
#Make a Label for the confusion matrix
labels = ['0', '1', '2']
#Make a prediction from the model
prdictions = model.predict(Test_data)
#Convert predictions to class labels
prediction_classes = np.argmax(prdictions, axis=1)
 #Creating a confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(Test_marking, prediction_classes)
#Plotting the confusion matrix
plt.figure()
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()