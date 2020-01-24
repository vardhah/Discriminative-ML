'''
Main scripts which calls the controller and the TorcsEnv scripts
Invokes the drive_example function which has both the leader car and the follower car functionalities
Includes random generation of leader car speeds and its position on the track
'''

import Controller
import TorcsEnv
import random #for generating the random traget speed and position
import csv
from keras.models import model_from_json
import numpy as np

#Variables for the random behavior of the leader car

targetSpeed = 70.0
targetStrE = 0.0
PI = 3.14159265359
acc = 0.0
brakes =0.0
csvfile = open("data0916.csv", "w")
acc =0.0
brake = 0.0

def steeringControl(S):
    steering = S['angle']*10 / PI
    steering -= (S['trackPos']) *.10
    return steering

def ACCspeedControl(S, R, targetSpeed):
    if S['speedX'] < targetSpeed - (R['steer']*60):
        acc = R['accel'] + .01# Add the accelarction code here
    else:
        acc = R['accel'] - .01 #Brake code goes here
    if S['speedX']<10:
        acc = R['accel'] + 1/(S['speedX']+.1)
    accMin = 0.0
    accMax = 1.0
    acc=max(accMin, acc)
    acc=min(accMax, acc)
    return acc

# Speed control of the leader car
def speedControl(S, R, targetSpeed):
    global acc, brake
    if S['speedX'] < targetSpeed - (R['steer']*60):
         acc = R['accel']+ .01
         brake = 0.0  # Add the accelarction code here
    else:
         brake = R['brake']+ 0.000005
         acc =0.0 #Brake code goes here
    if S['speedX']<10:
        acc = R['accel'] + 1/(S['speedX']+.1)
    accMin = 0.0
    accMax = 1.0
    acc=max(accMin, acc)
    acc=min(accMax, acc)
    brakeMin = 0.0
    brakeMax = 1.0
    brake=max(brakeMin, brake)
    brake=min(brakeMax, brake)
    return acc, brake

def automaticGear(S):
    gear=1
    if S['speedX']>50:
        gear=2
    if S['speedX']>80:
        gear=3
    if S['speedX']>110:
        gear=4
    if S['speedX']>140:
        gear=5
    return gear

def drive_example(c):
    global targetSpeed, acc, brakes
    #S => sensor data (input) , R => actuator data (output)
    S, R = c.S.d, c.R.d

    # Change here to call the function of controller to give actuator function based on given input sensor data
    # -------------------------------------------------------
    #--------------------------------------------------------
    R['accel'], R['brake'] = speedControl(S,R,targetSpeed)
    R['steer'] = Controller.ACCSteeringController(S)
    R['gear'] = automaticGear(S)
    
    # -------------------------------------------------------
    # -------------------------------------------------------
    collectData(S, R)

            

def collectData(S, R):
        #Data being collected in CSV
        car1 = []
        car1.append(S['angle'])         
        car1.append(S['speedX'])
        car1.append(S['speedY'])
        car1.append(S['speedZ']) 
        car1.append(S['rpm'])
        car1.append(S['track'])
        car1.append(S['trackPos'])
        car1.append(R['steer'])
        car1.append(R['accel'])
        car1.append(R['brake'])
        car1.append(S['gear'])

        writer = csv.writer(csvfile)
        writer.writerow(car1)


if __name__ == "__main__":
    #Loading the nn model
    '''
    with open('model.json', 'r') as jfile:
        model = model_from_json(jfile.read())
    model.load_weights('model.h5')
    '''
    #Loading the normalized parameters
    # dataNormalization = np.load('normalizeParameters.npz')

    #Adding different clients to the simulation
    CS=[TorcsEnv.Client(p=P) for P in [3001]]
    for step in range(CS[0].maxSteps, 0, -1):#The simulation steps
        num = 0
        for C in CS:
            C.get_servers_input()#Get sensor values from the simulator
            drive_example(C)#Invoke the python client for the car controller
            C.respond_to_server()#Send the actuator control signals to the simulator
            #num+=1
    C.shutdown()
