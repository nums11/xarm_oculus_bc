import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

def getActionSemantics(action):
    if action == 0:
        return "Move Forward"
    elif action == 1:
        return "Move Backward"
    elif action == 2:
        return "Move Left"
    elif action == 3:
        return "Move Right"
    elif action == 4:
        return "Move Up"
    elif action == 5:
        return "Move Down"
    elif action == 6:
        return "Open Gripper"
    elif action == 7:
        return "Close Gripper"

traj_idx = 9
sample_idx = 40
filename = './data/traj_' + str(traj_idx) + '/sample_' + str(sample_idx) +'.npy'


# Show sample ----------------------
sample = np.load(filename, allow_pickle=True)
obs, delta_with_gripper = sample
plt.title('x: ' + str(delta_with_gripper[0]) + ', y: ' + str(delta_with_gripper[1]) \
          + ', z: ' + str(delta_with_gripper[2]) + ' gripper: ' + str(delta_with_gripper[3]))
plt.imshow(obs, interpolation='nearest')
plt.show()

# Show max and min deltas ----------------------
# num_trajectories = len(os.listdir('./data'))
# max_x_delta = float('-inf')
# max_y_delta = float('-inf')
# max_z_delta = float('-inf')
# min_x_delta = float('inf')
# min_y_delta = float('inf')
# min_z_delta = float('inf')
# for i in range(num_trajectories):
#     traj_dir = './data/traj_' + str(i)
#     traj_sample_filenames = [traj_dir + '/' + sample for sample in os.listdir(traj_dir)]
#     for sample_filename in traj_sample_filenames:
#         sample = np.load(sample_filename, allow_pickle=True)
#         obs, delta_with_gripper = sample
#         x_delta, y_delta, z_delta, _ = delta_with_gripper
#         if x_delta > max_x_delta:
#             max_x_delta = x_delta
#         if y_delta > max_y_delta:
#             max_y_delta = y_delta
#         if z_delta > max_z_delta:
#             max_z_delta = z_delta
#         if x_delta < min_x_delta:
#             min_x_delta = x_delta
#         if y_delta < min_y_delta:
#             min_y_delta = y_delta
#         if z_delta < min_z_delta:
#             min_z_delta = z_delta
# print("max_x_delta", max_x_delta)
# print("max_y_delta", max_y_delta)
# print("max_z_delta", max_z_delta)
# print("min_x_delta", min_x_delta)
# print("min_y_delta", min_y_delta)
# print("min_z_delta", min_z_delta)
