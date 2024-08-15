import sys
import os
sys.path.append('/home/weirdlab/xarm_bc/xArm-Python-SDK')
from xarm.wrapper import XArmAPI
from time import sleep
from oculus_reader.oculus_reader.reader import OculusReader
import numpy as np
import threading
import cv2
import numpy as np
from pynput.keyboard import Key, Listener
import random

def getArmPosition():
    code, [arm_x, arm_y, arm_z, roll, pitch, yaw] = arm.get_position()
    return np.array([arm_x, arm_y, arm_z, roll, pitch, yaw])

def getControllerPositionAndTrigger():
    transformations, buttons = oculus.get_transformations_and_buttons()
    positions = transformations['r'][:,3]
    return (positions[0:3], buttons['RTr'])

def saveSample(data_dir, timestep, image, delta_with_gripper):
    sample = np.array([image, delta_with_gripper,], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

def getDatasetDirectory():
    num_trajectories = len(os.listdir('./data'))
    dir_name = './data/traj_' + str(num_trajectories) + '/'
    os.mkdir(dir_name)
    return dir_name

def getCameraImage():
    global camera_image
    while True:
        ret, frame = camera.read()
        resized = cv2.resize(frame, (512, 512))
        camera_image = np.asarray(resized)

def createDatasetFromTrajectory(trajectory):
    data_dir = getDatasetDirectory()
    traj_length = len(trajectory)
    for timestep, (image, position, _) in trajectory.items():
        if not timestep == traj_length-1:
            _, next_position, next_gripper_status = trajectory[timestep+1]
            delta = next_position - position
            # Remove rotation
            translational_delta = delta[0:3]
            delta_with_gripper = np.append(translational_delta, next_gripper_status)
            saveSample(data_dir, timestep, image, delta_with_gripper)
            print("Created sample", timestep)
    print("Finished creating dataset:", data_dir)
    arm.open_lite6_gripper()

def getGripperStatus():
    global gripper_closed
    if gripper_closed:
        return 1
    else:
        return 0

# Record the trajcetory at 10 FPS
def recordTrajectory():
    global stop_recording
    sleep(0.5)
    trajectory = {}
    timestep = 0
    while not stop_recording:
        image = camera_image
        position = getArmPosition()
        gripper_status = getGripperStatus()
        trajectory[timestep] = (image, position, gripper_status)
        timestep += 1
        print("t:", timestep)
        sleep(0.1)
    print("Finished recording trajectory. Creating Dataset")
    createDatasetFromTrajectory(trajectory)

def onPress(key):
    global stop_recording
    if key.char == 'q':
        print("Stopping Recording!")
        stop_recording = True

# x range: 120 - 200
# y range: -200 - 200
# z range: 150 - 200
def getRandomInitialPosition():
    x = random.randint(120, 200)
    y = random.randint(-200, 200)
    z = random.randint(150, 200)
    return (x,y,z)

robot_ip = '192.168.1.185'
arm = XArmAPI(port=robot_ip)
arm.connect()
arm.clean_warn()
arm.clean_error()
arm.motion_enable(enable=True)
arm.set_mode(0) # position control mode
arm.set_state(state=0) # sport state
print("Resetting arm")
arm.reset(wait=True)
initial_position = getRandomInitialPosition()
print("Setting arm to random initial position", initial_position)
arm.set_position(x=initial_position[0], y=initial_position[1],
                 z=initial_position[2], wait=True)
arm.set_mode(1) # servo motion mode
arm.set_state(state=0) # sport state
sleep(2)

oculus = OculusReader()
print("Initialized Oculus Reader")
sleep(1)

camera = cv2.VideoCapture(0 + cv2.CAP_V4L2)
print("Initialized Camera. Reading Frames at", int(camera.get(cv2.CAP_PROP_FPS)), "FPS")
sleep(1)

quit_listener = Listener(on_press=onPress)
quit_listener.start()

camera_image = None
camera_thread = threading.Thread(target=getCameraImage, args=())
camera_thread.start()
sleep(1)

stop_recording = False
gripper_closed = False
trajectory_thread = threading.Thread(target=recordTrajectory, args=())
trajectory_thread.start()

orig_arm_pos = getArmPosition()
roll, pitch, yaw = orig_arm_pos[3:]
arm_pos = [orig_arm_pos[0], orig_arm_pos[1], orig_arm_pos[2]]
controller_pos, _ = getControllerPositionAndTrigger()
print("Begin Teleoperation")
while not stop_recording:
    new_controller_pos, right_trigger = getControllerPositionAndTrigger()

    controller_deltas = np.subtract(new_controller_pos,controller_pos) * 1000
    arm_deltas = [-1 * controller_deltas[2],
        -1 * controller_deltas[0], controller_deltas[1]]
    
    new_arm_pos = np.add(arm_pos, arm_deltas)
    arm.set_servo_cartesian(mvpose=[new_arm_pos[0],
        new_arm_pos[1], new_arm_pos[2], roll, pitch, yaw])
    
    if right_trigger:
        arm.close_lite6_gripper()
        gripper_closed = True
    else:
        arm.open_lite6_gripper()
        gripper_closed = False
    
    controller_pos = new_controller_pos
    arm_pos = new_arm_pos
    sleep(0.005)