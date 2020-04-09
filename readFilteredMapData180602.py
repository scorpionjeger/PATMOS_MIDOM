import numpy
import matplotlib.pyplot as plt
import pylab
import sys
import datetime as dt
import pickle
import os
import random
import math
import cv2
#import Tkinter
from Tkinter import Tk
from tkFileDialog import askopenfilename,askdirectory
import tkMessageBox
from Tkinter import *



#Map = open("F:/belize_nov_trip/maps/example_tom_GoPro243_refind.txt")
#csvout="F:/belize_nov_trip/maps/tomGP243_ltp0001.xyz"

#Map = open('C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/example_GoProTom_243_refined.txt')
Map = open('F:/Australia/ORB_Maps/metadata_Nik_827_190525_H_refind.txt')

csvout="F:/Australia/ORB_Maps/metadata_Nik_827_190525_H_rgb.txt"


csvtxt = open(csvout, "w")

MapLines = Map.readlines()

for i in range(len(MapLines)):
    if i>0:
        #if float(MapLines[i].split(";")[4])<.01 and float(MapLines[i].split(";")[5])>5  :
        if 1:
#            csvstring=MapLines[i].split(";")[1]+" "+MapLines[i].split(";")[2]+" "+MapLines[i].split(";")[3]
            #csvstring = MapLines[i].split(";")[1] + " " + MapLines[i].split(";")[2] + " " + MapLines[i].split(";")[3]+" " +MapLines[i].split(";")[6] + " " + MapLines[i].split(";")[7] + " " + MapLines[i].split(";")[8]
            csvstring = MapLines[i].split(";")[1] + " " + MapLines[i].split(";")[2] + " " + MapLines[i].split(";")[3] + " " + \
                        MapLines[i].split(";")[8] + " " + MapLines[i].split(";")[7] + " " + MapLines[i].split(";")[6]

            csvtxt.write(csvstring + "\n")

