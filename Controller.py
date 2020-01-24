'''
Script which has all the functions of the car controller
Leader car controller is built to drive at the center of the track and does a defensive driving of slight acceleration and no brake
Follower car is based on Siyun's controller equations
'''
import math
import numpy as np
import keras

#Constants used in the PD controller
PI = 3.14159265359
k_siP = 0.4
k_siD = 5.0
k_sd = 0.0001
M_c2 = 0.0001

Xr = 0.0
errorL = 0.0

# steering control of the leader car
def steeringControl(S, targetPos):
    steering = S['angle']*10 / PI
    steering -= (S['trackPos']-targetPos) *.10
    return steering

# Speed control of the leader car
def speedControl(S, R, targetSpeed):
    #acc=0
    #brake=0
    if S['speedX'] < targetSpeed - (R['steer']*50):
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

#Speed control of the follower car
def ACCVelocityController(Vl, S):
    global Xr
    distance = min(S['opponents'])
    V = S['speedX']
    Xmin = 20.0
    Vd = Vl + (distance - Xmin - 0.5*V*1000/3600)*0.3*3600.0/1000.0
    Vr = Vd - V
    Xr += Vr*1000/3600*0.02
    action = 0.04*Xr + 0.04*Vr
    #print("rawAction:", action, "Xr:", Xr, "Vd:", Vd, "Distance", distance)
    actionMin = -1.0
    actionMax = 1.0
    action = max(actionMin, action)
    action = min(actionMax, action)
    if (action >= 0.0):
        return [action, 0.0, Xr]
    else:
        return [0.0, -1.0*action, Xr]


#Steering control of the follower car
def ACCSteeringController(S):
    global M_c2, errorL
    trackPos = trackPosCalc(S['track'])
    vx = S['speedX']
    vy = S['speedY']
    if (trackPos >=0):
        sign = -1.0
    else:
        sign = 1.0
    steering = sign * (math.fabs(k_siP*(trackPos)+k_siD*(trackPos-errorL)) -k_sd*math.fabs(vy) - M_c2*vx)
    errorL = trackPos
    if (steering > 1):
       steering = 1.0
    elif (steering < -1):
       steering = -1.0
    return steering

#track position calculation from LIDAR2 (track)
def trackPosCalc(trackLidar):
    edges=trackLidar
    rightc=edges[15]
    rightp=edges[18]

    #Calculating the Lateral_Distance from the sidewall, using the LIDAR2 (track) values
    rightalpha=math.atan((1.0*rightc*math.cos(30.0*PI/180) - rightp)/(1.0*rightc*math.sin(30.0*PI/180)))
    rightd=rightp * math.cos(rightalpha)
    trackPos = (rightd - 5.0)/5.0

    return trackPos

# NN controller for ACC
def nncontroller(row, model):
    inputs = np.array(row)[np.newaxis]
    outputs = model.predict(inputs, batch_size=1)
    return [float(outputs[0][0]), float(outputs[0][1]), float(outputs[0][2])]

#Gear change according to the speed of the car
def automaticGear(S):
    gear=1
    if S['speedX']>80:
        gear=2
    if S['speedX']>120:
        gear=3
    if S['speedX']>140:
        gear=4
    if S['speedX']>160:
        gear=5
    if S['speedX']>170:
        gear=6
    return gear
