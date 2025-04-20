import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'D:\\Documents\\____school\\MLF_Project\\Train'

# Get a list of all files in the folder
files = os.listdir(folder_path)
Train_data = [np.load(os.path.join(folder_path, file)) for file in files if file.endswith('.npy')]
# combine all arrays into a single array

plt.figure()
plt.imshow(Train_data[55], cmap='gray', interpolation='nearest')
plt.title('Test Data 0')
plt.show()

