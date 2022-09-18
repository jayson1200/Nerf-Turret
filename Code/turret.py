from tflite_runtime.interpreter import Interpreter
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import RPi.GPIO as GPIO
import time
import threading
import sys
import signal

GPIO.setmode(GPIO.BCM)

DIR_MTR_ONE = 5
STEP_MTR_ONE = 6

DIR_MTR_TWO = 26
STEP_MTR_TWO = 13

FLY_WHEEL_CONTROL = 21

GPIO.setwarnings(False)

GPIO.setup(DIR_MTR_TWO, GPIO.OUT)
GPIO.setup(STEP_MTR_TWO, GPIO.OUT)

GPIO.setup(DIR_MTR_ONE, GPIO.OUT)
GPIO.setup(STEP_MTR_ONE, GPIO.OUT)

GPIO.setup(FLY_WHEEL_CONTROL, GPIO.OUT)

class StepperController:
    def __init__(self, kP, screen_width, scren_height):
        self.kP = kP

    def get_speed(self, pixel_distance):
        return self.kP * pixel_distance

    def get_steps(self, x_distance_from_target, y_distance_from_target):
        x_steps = 2*int(0.0463*x_distance_from_target + 0.37)
        y_steps = 2*int(0.0463*y_distance_from_target + 0.37)

        return (x_steps, y_steps)

#The interpreter represents the model
interpreter = Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')

#loads tensors (something we need to do when using a tf-lite model)
interpreter.allocate_tensors()

cont = StepperController(0.99999999999999999, 640, 480)


def rotate_abt_y_ccw(speed, steps):
    curr_steps = 0
    speed = 1/speed
    
    stepper_curr_state = True

    GPIO.output(DIR_MTR_ONE, True)

    while curr_steps < steps:
        GPIO.output(STEP_MTR_ONE, stepper_curr_state)

        stepper_curr_state = not stepper_curr_state

        curr_steps+=1

        time.sleep(speed)

def rotate_abt_y_cw(speed, steps):
    curr_steps = 0
    speed = 1/speed
    
    stepper_curr_state = True
    GPIO.output(DIR_MTR_ONE, False)

    while curr_steps < steps:
        GPIO.output(STEP_MTR_ONE, stepper_curr_state)

        stepper_curr_state = not stepper_curr_state

        curr_steps+=1

        time.sleep(speed)

def rotate_abt_x_down(speed, steps):
    curr_steps = 0
    speed = 1/speed
    
    stepper_curr_state = True

    GPIO.output(DIR_MTR_TWO, True)

    while curr_steps < steps:
        GPIO.output(STEP_MTR_TWO, stepper_curr_state)

        stepper_curr_state = not stepper_curr_state

        curr_steps+=1

        time.sleep(speed)


def rotate_abt_x_up(speed, steps):
    curr_steps = 0
    speed = 1/speed
    
    stepper_curr_state = True
    GPIO.output(DIR_MTR_TWO, False)

    while curr_steps < steps:
        GPIO.output(STEP_MTR_TWO, stepper_curr_state)

        stepper_curr_state = not stepper_curr_state

        curr_steps+=1

        time.sleep(speed)

# Draw keypoint on screen calculates the average then moves the turret to position
def apply_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    curr_avg = []
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)
            curr_avg.append([kx, ky])

    if(len(curr_avg) == 0):
        return

    curr_avg_converted = np.array(curr_avg)
    
    y_avg = np.average(curr_avg_converted, axis=0)[1]
    x_avg = np.average(curr_avg_converted, axis=0)[0]

    cv2.circle(frame, (int(x_avg), int(y_avg)), 4, (255,0,0), -1)

    move_to_target(int(y_avg), int(x_avg), y, x)

        


distance_threshold = 50
resolution = 20

move_up = threading.Thread(target=rotate_abt_x_up, args=(1, 0), daemon=True)
move_down = threading.Thread(target=rotate_abt_x_down, args=(1, 0), daemon=True)
move_cw = threading.Thread(target=rotate_abt_y_cw, args=(1, 0), daemon=True)
move_ccw = threading.Thread(target=rotate_abt_y_ccw, args=(1, 0), daemon=True)


def move_to_target(y_target, x_target, y_size, x_size):
    y_distance_to_target = y_target - (y_size/2)
    x_distance_to_target = x_target - (x_size/2)

    speed_x = abs(cont.get_speed(x_distance_to_target))
    speed_y = abs(cont.get_speed(y_distance_to_target))

    print("Distance to y target {} || Distance to x target: {} || Frame Dimensions: {}x{}", y_distance_to_target, x_distance_to_target, x_size, y_size)

    global move_up
    global move_down
    global move_cw
    global move_ccw

    steps = cont.get_steps(abs(x_distance_to_target), abs(y_distance_to_target))

    if(y_distance_to_target < -distance_threshold):
         if(not move_up.is_alive()):
            move_up = threading.Thread(target=rotate_abt_x_up, args=(speed_y, steps[1]), daemon=True) 
            move_up.start()
    elif(y_distance_to_target > distance_threshold):
        if(not move_down.is_alive()):
           move_down = threading.Thread(target=rotate_abt_x_down, args=(speed_y, steps[1]), daemon=True)
           move_down.start()
    else:
        speed_y = 0

    if(x_distance_to_target > distance_threshold):
        if(not move_cw.is_alive()):
            move_cw = threading.Thread(target=rotate_abt_y_cw, args=(speed_x, steps[0]), daemon=True)
            move_cw.start()
    elif(x_distance_to_target < -distance_threshold):
        if(not move_ccw.is_alive()):
            move_ccw = threading.Thread(target=rotate_abt_y_ccw, args=(speed_x, steps[0]), daemon=True)
            move_ccw.start()
    else:
        speed_x = 0

    if(not move_cw.is_alive() and not move_ccw.is_alive() and not move_up.is_alive() and not move_down.is_alive()):
        GPIO.output(FLY_WHEEL_CONTROL, True)
    else:
        GPIO.output(FLY_WHEEL_CONTROL, False)
    


#connects to our camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    #unpack a single frame from the feed
    ret, frame = cap.read()

    if(frame is None):
        print("skipping execution")
        continue

    img = frame.copy()

    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    # physically runs our input data through the model
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    apply_keypoints(frame, keypoints_with_scores, 0.6)
    
    #shows the feed
    cv2.imshow('MoveNet Lightning', frame)
    
    #provides a pathway for the user to break out of the program
    if cv2.waitKey(10) & 0xFF==ord('q'):
        GPIO.output(FLY_WHEEL_CONTROL, False)
        break


#close the program      
cap.release()
cv2.destroyAllWindows()
