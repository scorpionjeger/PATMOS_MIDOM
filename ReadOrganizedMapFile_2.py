import cv2 as cv

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

from Tkinter import Tk
from tkFileDialog import askopenfilename
import tkMessageBox
from Tkinter import *
import scipy.misc
import h5py

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

import alignment_by_row_channels_pb

file1 = 'F:/belize_nov_trip/maps/example_tom_GoPro243_MapEmph.h5'
file1 = 'C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/example_GoProTom_243_MapEmph.h5'
#file1 = "G:/belize_nov_trip/maps/example_Nik658_2_MapEmph.h5"
file1 = "G:/belize_nov_trip/maps/example_Nik658_2_MapEmph.h5"
f = h5py.File(file1, 'r')

errors = []
itter = 0
#csvout = "F:/belize_nov_trip/maps/example_tom_GoPro243_refind.txt"
#csvout = 'C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/example_GoProTom_243_refined.txtff'
csvout = "G:/belize_nov_trip/maps/example_Nik658_2_refind_2.txt"
csvtxt = open(csvout, "w")
csvstring = "Map Point Index;3D Map x;3D Map y;3D Map z;Map point error;Map point angular spred;Blue;Green;Red;number of points\n"
csvtxt.write(csvstring)
print len(f['mapPoints'].keys())

Keysmin = min(int(s.split("F")[1]) for s in f['frames'].keys())
Keysmax = max(int(s.split("F")[1]) for s in f['frames'].keys())

CamVectBank=numpy.zeros((Keysmax-Keysmin+1,3))

RGBMask=numpy.zeros((255+1))
for i in range(255+1):
    RGBMask[i]=i

RGBMask[0]=numpy.nan

RGBMask[255]=numpy.nan
RGBMask[254]=numpy.nan
RGBMask[253]=numpy.nan


if 1:
    for i in f['frames'].keys():
        CamVectBank[int(int(i.split("F")[1])-Keysmin)][0]= f['frames'][i][9]
        CamVectBank[int(int(i.split("F")[1]) - Keysmin)][1] = f['frames'][i][10]
        CamVectBank[int(int(i.split("F")[1]) - Keysmin)][2] = f['frames'][i][11]

print CamVectBank
print Keysmin,Keysmax

for MapPoints in f['mapPoints']:

    #    print MapPoints
    #    MapData = f['mapPoints'][MapPoints]["MapData"]
    #    print MapData.shape
    if itter > -1:
        # try:
        MapData = f['mapPoints'][MapPoints]["MapData"]
        MapDataSH = f['mapPoints'][MapPoints]["SharedMapData"]
        sf2, sf1 = MapData.shape
        #print itter, sf2, sf1
        mapPointError = numpy.zeros(sf2)
        #RGBpoints = numpy.zeros(sf2)
        RGBpoints = [0]*sf2

        MapVector = numpy.array([MapDataSH[0], MapDataSH[1], MapDataSH[2]])
        Map2CamAve = numpy.zeros(3)
        MapThetVect = numpy.zeros(sf2)
        mapPointError[:] = MapData[:, 8]
#        RGBpoints[:]=MapData[:, 5] + MapData[:, 6] + MapData[:, 7]
        RGBpoints[:] = map(int,MapData[:, 6])

        RGBpoints[:]=RGBMask[RGBpoints[:]]
        RGBpoints=numpy.array(RGBpoints)
        RGBmedian = numpy.nanpercentile(RGBpoints, 50, interpolation="nearest")
        #i_near = numpy.where(RGBpoints == RGBmedian).argmin()
        try:
            RGBindex = numpy.nanargmin(abs(RGBpoints - RGBmedian))
        except:
            RGBindex=0
                #print RGBpoints
        CamOrgn=numpy.zeros((sf2,3))
        Map2Cam = numpy.zeros((sf2, 3))
        MapIndex=map(int,MapData[:, 0]-Keysmin)

        CamOrgn[:][:] = CamVectBank[MapIndex[:]][:]
        #CamOrgn[:][0] = CamVectBank[MapIndex[:]][0]
        #CamOrgn[:][0] = CamVectBank[MapIndex[:]][0]
        Map2Cam[:][:] = CamOrgn[:][:] - MapVector[:]
        #print Map2Cam[0],"first"
        Map2CamNorm=numpy.zeros((sf2))
        Map2CamNorm[:]=numpy.linalg.norm(Map2Cam[:])
        #Map2Cam[:][:] = Map2Cam[:] / numpy.linalg.norm(Map2Cam[:])
        #Map2Cam[:][:] = Map2Cam[:] / numpy.linalg.norm(Map2Cam[:])
        for i in range(sf2):
            Map2Cam[i][:] = Map2Cam[i][:] / numpy.linalg.norm(Map2Cam[i][:])
        #print Map2Cam[0],"second"
        #print Map2Cam
        #Map2Cam[:][:] = Map2Cam[:][:] / numpy.linalg.norm(Map2Cam[:][:])
        #print Map2Cam[0], "thrid"


        Map2CamAve=numpy.average(Map2Cam,axis=0)
        if 0:
            Map2CamAve2=numpy.zeros(3)
            Map2Cam2=numpy.zeros(3)
            for i in range(sf2):

                CamVect = f['frames']["F" + str(int(MapData[i, 0]))]
                CamOrgn2 = numpy.array([CamVect[9], CamVect[10], CamVect[11]])


                Map2Cam2 = CamOrgn2 - MapVector
                Map2Cam2 = Map2Cam2 / numpy.linalg.norm(Map2Cam2)

                print Map2Cam2,"Map2Cam2"
                print Map2Cam[i]
                Map2CamAve2 += Map2Cam2
            Map2CamAve2 = Map2CamAve2 / sf2

            print Map2CamAve
            print Map2CamAve2

        for i in range(sf2):
            MapThetVect[i] = (180 / 3.14159) * numpy.arccos(
                numpy.dot(Map2Cam[i][:], Map2CamAve) / (numpy.linalg.norm(Map2Cam[i][:]) * numpy.linalg.norm(Map2CamAve)))




        #print MapThetVect
        MapThetaSDV = numpy.std(MapThetVect)
        #print MapThetaSDV, "MapThetaSDV"



        csvstring = MapPoints + ";" + str(MapDataSH[0]) + ";" + str(MapDataSH[1]) + ";" + str(
            MapDataSH[2]) + ";" + str(numpy.std(mapPointError)) + ";"
        csvstring = csvstring + str(MapThetaSDV) + ";" + str(MapData[RGBindex, 5]) + ";" + str(
            MapData[RGBindex, 6]) + ";" + str(MapData[RGBindex, 7]) + ";" + str(sf2)
        print itter
        #print csvstring
        csvtxt.write(csvstring+"\n")

        del MapData, mapPointError, RGBpoints

    itter += 1



