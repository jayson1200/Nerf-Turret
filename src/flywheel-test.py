import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

FLY_WHEEL_CONTROL = 21

GPIO.setup(FLY_WHEEL_CONTROL, GPIO.OUT)

flywheel_state = True

while True: 
    GPIO.output(FLY_WHEEL_CONTROL, flywheel_state)

    time.sleep(3)

    flywheel_state = not flywheel_state

    