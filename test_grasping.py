import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/weirdlab/xarm_bc/xArm-Python-SDK')
from xarm.wrapper import XArmAPI
from time import sleep
import threading

def moveRobot(arm, deltas_with_gripper):
    x_delta, y_delta, z_delta, gripper_close = deltas_with_gripper
    _, pos = arm.get_position()
    x, y, z, roll, pitch, yaw = pos
    new_x = x + x_delta
    new_y = y + y_delta
    new_z = z + z_delta
    arm.set_servo_cartesian(mvpose=[new_x,new_y,new_z,roll,pitch,yaw], speed=0.00001)
    if gripper_close > 0.7:
        arm.close_lite6_gripper
    else:
        arm.open_lite6_gripper()

def readCamera():
    global camera_image
    while True:
        ret, frame = vid.read()
        resized = cv2.resize(frame, (512, 512))
        camera_image = np.asarray(resized)
    
ip = '192.168.1.185'
arm = XArmAPI(port=ip)
arm.connect()
arm.clean_warn()
arm.clean_error()
arm.motion_enable(enable=True)
arm.set_mode(0) # position control mode
arm.set_state(state=0) # sport state
print("Resetting Arm")
arm.reset(wait=True)
arm.set_mode(1) # servo mootion mode
arm.set_state(state=0) # sport state
sleep(2)

print("Initializing Camera")
vid = cv2.VideoCapture(0 + cv2.CAP_V4L2)
print("Reading frames at", int(vid.get(cv2.CAP_PROP_FPS)), "FPS")

print("Loading Model")
model = tf.keras.models.load_model('bc_network_v5.h5')

print("Starting Camera Read")
camera_image = None
camera_thread = threading.Thread(target=readCamera, args=())
camera_thread.start()
sleep(2)

while True:
    obs = camera_image
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, 0) # batch dim
    predictions = model.predict(obs)[0]
    print("Predictions",predictions)
    moveRobot(arm, predictions)
    sleep(0.1)