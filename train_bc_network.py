import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from tqdm import tqdm

# Load the data
num_trajectories = len(os.listdir('./data'))
num_trajectories = num_trajectories - 10
x = []
y = []
print("Loading Data...")
for i in tqdm(range(num_trajectories)):
    traj_dir = './data/traj_' + str(i)
    traj_sample_filenames = [traj_dir + '/' + sample for sample in os.listdir(traj_dir)]
    for sample_filename in traj_sample_filenames:
        sample = np.load(sample_filename, allow_pickle=True)
        obs, delta_with_gripper = sample
        # delta_with_gripper = delta_with_gripper.reshape(-1,1)
        x.append(obs)
        y.append(delta_with_gripper)

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x, y))
train_ds = train_ds.batch(16)
print(train_ds)


# num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(512, 512, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['root_mean_squared_error'])
model.summary()

epochs=100
history = model.fit(
  train_ds,
  epochs=epochs
)

model.save('bc_network_v5.h5')

error = history.history['root_mean_squared_error']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, error, label='Training Error')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

sample_idx = 173
features = x[sample_idx]
features = np.expand_dims(features, axis=0)
labels = y[sample_idx]

predictions = model.predict(features)
print("predictions", predictions)
print("labels", labels)
