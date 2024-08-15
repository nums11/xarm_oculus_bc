import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('bc_network_v2.h5')

traj_idx = 3
sample_idx = 3
filename = './data/traj_' + str(traj_idx) + '/sample_' + str(sample_idx) +'.npy'

sample = np.load(filename, allow_pickle=True)
obs, delta_with_gripper = sample
img = obs

obs = tf.convert_to_tensor(obs)
obs = tf.expand_dims(obs, 0) # batch dim
predictions = model.predict(obs)[0]
print("Predictions", predictions)
print("Actual", delta_with_gripper)

title = '\nPredicted ----\n x: ' + str(predictions[0]) + ' y: ' + str(predictions[1]) \
    + ' z: ' + str(predictions[2]) + ' gripper: ' + str(predictions[3]) + '\n' + \
    'Actual --- \n x: ' + str(delta_with_gripper[0]) + ' y: ' + str(delta_with_gripper[1]) \
    + ' z: ' + str(delta_with_gripper[2]) + ' gripper: ' + str(delta_with_gripper[3]) + '\n'

plt.title(title)
plt.imshow(img, interpolation='nearest')
plt.show()
