from flask import Flask, redirect, url_for, render_template, request
from IPython.display import HTML
from fnc.extractFeature import extractFeature
from fnc.matching import matching
from scipy.io import savemat
from time import time

import argparse, os
import scipy.io as sio
import cv2
import mysql.connector
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,default = ".\\test_iris.jpg",
                    help="Path to the file that you want to verify.")

parser.add_argument("--temp_dir", type=str, default=".\\templates\\temp\\",
                    help="Path to the directory containing templates.")

parser.add_argument("--thres", type=float, default=0.38,
                    help="Threshold for matching.")

args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
# Extract feature

video = cv2.VideoCapture(0)
while True:
    ret,frame=video.read()
    cv2.rectangle(frame,pt1= (100,50),pt2 = (480,430),color = (0,255,0),thickness=5)

    cv2.imshow("iris image",frame)
    k = cv2.waitKey(1)
    if k == 13:
        test = (".\\test_iris.jpg")
        cv2.imwrite("img.jpg", frame)
        img = cv2.imread("img.jpg",0)
        img = img[50:430, 100:480]
        cv2.imwrite("test_iris.jpg", img)
        break

#release camera
video.release()

start = time()
print('>>> Start verifying {}\n'.format(args.file))
template, mask, file = extractFeature(args.file)


# Matching
result,match_dist = matching(template, mask, args.temp_dir, args.thres)
x= result.split(".")
if result == -1:
    print('>>> No registered sample.')

elif result == 0:
    print('>>> No sample matched.')

else:
    print(x[0])
    print("matching distance : ",match_dist)


# Time measure
end = time()
print('\n>>> Verification time: {} [s]\n'.format(end - start))