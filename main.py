import os
import csv
from random import randint
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
import pandas as pd
#Disable the floating point in tesorfolw to avoid performance issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# Load the train data
def load_npy_folder(folder, id_list):
    return np.stack([
        np.load(os.path.join(folder, f"{i}.npy"))
        for i in id_list
    ], axis=0)

folder_path = 'Train'
#import of the train data wia the csv file to prevent any issues with lixed up labels
# read the true labels from the CSV file
Train_labels = pd.read_csv('label_train.csv',names=['ID', 'target'],header =0)
# build a filename colum to load the data
Train_labels['Train'] = Train_labels['ID'].astype(int).astype(str) + '.npy'
# sort the labels by ID to load the matching data
Train_labels.rename(columns={'target':'label'}, inplace=True)
Train_labels.sort_values('ID', inplace=True, ignore_index=True)
Train_data = load_npy_folder('Train', Train_labels['ID'])
Train_marking = Train_labels['label'].values

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

# adding a class wight to the data because the dataset is highely unbalanced
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Train_marking),
    y=Train_marking
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)


# Load the test data
folder_path = 'Test'
# Get a list of all files in the folder
Test_labels = pd.read_csv('test_format.csv', names=['ID', ' label'],header =0)
Test_labels['Test'] = Test_labels['ID'].astype(int).astype(str) + '.npy'
# sort the labels by ID to load the matching data and append them to one array
Test_data = np.stack([
    np.load(os.path.join('Test', fn))
    for fn in Test_labels['Test']
], axis=0)
# import the marking for data
Test_marking = np.genfromtxt('test_format.csv', delimiter=',', skip_header=1) 
#add the marking to one colum array
Test_marking = Test_marking[:, 1]  

# print the shape of the data
print (Test_data.shape)
print(Test_marking.shape) 
#ptint a class balance of the data
print (np.unique(Train_marking, return_counts=True))
random_index = randint(0, 121) 
plt.figure()
plt.imshow(Test_data[random_index], cmap='gray', interpolation='nearest')
plt.title('Test Data' + str(random_index))
plt.show()

# Add a channel dimension to the data (grayscale images have 1 channel)
Train_data = np.expand_dims(Train_data, axis=-1)  
Test_data = np.expand_dims(Test_data, axis=-1)    
#data preprocesing
#spliting the data into train and validation dataset
data_train, data_validation, markng_train, marking_validation = train_test_split(
    Train_data, Train_marking, test_size=0.3, random_state=42, stratify=Train_marking
)
#Encodeing marking data ot universal matrix to feed into model.
markng_train = keras.utils.to_categorical(markng_train, 3)
marking_validation = keras.utils.to_categorical(marking_validation, 3)
Test_markings = keras.utils.to_categorical(Test_marking, 3)

#adding the first experimental model it will be changed with more information about dataset
model = keras.models.Sequential()
#I use the Input layer as a conv 2D which devides the image to small subsection do search for diferences
model.add(keras.layers.Conv2D(32,kernel_size=3,activation='relu', input_shape=(72, 48, 1)))
#Adding a second convolutional layer with larger filter to pull more form Image
model.add(keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
#Adding a third convolutional layer with larger filter to pull more form Image
model.add(keras.layers.Conv2D(128,kernel_size=3,activation='relu'))
#Adding battch normalization to emphasize faster learning winth less overfitting
model.add(keras.layers.BatchNormalization())
#Adding a max pooling layer to reduce the complexity of the model
model.add(keras.layers.MaxPooling2D(pool_size=2))
# Flatten the output to feed into Dense layers
model.add(keras.layers.GlobalAveragePooling2D())
#adding a linear layer to make a realtion between the data
model.add(keras.layers.Dense(128, activation='relu'))
#adding a second dense layer the learning deeper 
model.add(keras.layers.Dense(256, activation='relu'))
#Adding a dropout layer to reduce overfitting
model.add(keras.layers.Dropout(0.5))
#Output layer is 3 because we have 3 classes
model.add(keras.layers.Dense(3, activation='softmax'))

# Compileing the model
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)
# Adding a learning rate scheduler
Scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
# Adding an early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.summary()
# Train the model
history = model.fit(
    data_train,
    markng_train,
    epochs=15,
    batch_size=32,
    validation_data=(data_validation, marking_validation),
    class_weight=class_weights_dict,
    callbacks=[Scheduler, early_stopping]
)
# Evaluate the model
val_loss, val_acc = model.evaluate(data_validation, marking_validation)
print('Validation accuracy:', val_acc)
print('Validation loss:', val_loss)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

#plot the confusion matrix
#Make a Label for the confusion matrix
labels = ['0', '1', '2']
#Make a prediction from the model
prdictions = model.predict(Test_data)
print(prdictions[:25])  
#Convert predictions to class labels
prediction_classes = np.argmax(prdictions, axis=1)

 #Creating a confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(
    Test_marking, prediction_classes, labels=[0, 1, 2])
#Plotting the confusion matrix
plt.figure()
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# adda an export of the predistion to csv file
with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'target'])
    for i, pred in enumerate(prediction_classes):
        writer.writerow([i, pred])