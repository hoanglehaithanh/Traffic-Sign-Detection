import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math

from classification import training, getLabel

SIGNS = ["ERROR",
        "STOP",
        "TURN LEFT",
        "TURN RIGHT",
        "DO NOT TURN LEFT",
        "DO NOT TURN RIGHT",
        "ONE WAY",
        "SPEED LIMIT",
        "OTHER"]

def main(file_name):
    vidcap = cv2.VideoCapture(file_name)
    file = open("Output.txt", "r")
    values = file.readlines()
    print(values)
    values = values[1:]
    current_frame = 0

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_2.avi',fourcc, fps , (640,480))

    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        frame = cv2.resize(frame, (640,int(height/(width/640))))

        #print(values[0])
        #print(values[0].split())
        if len(values) > 0:
            frame_no, sign_type, left, top, right, bottom = values[0].split(" ")
            if current_frame == int(frame_no):
                coordinate = [(int(left),int(top)),(int(right),int(bottom))]
                cv2.rectangle(frame, coordinate[0],coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                text = SIGNS[int(sign_type)]
                cv2.putText(frame,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
                values = values[1:]
        print(current_frame)
        current_frame += 1
        #cv2.imshow("Result", frame)
        out.write(frame)
main("./Videos/MVI_1049.avi")