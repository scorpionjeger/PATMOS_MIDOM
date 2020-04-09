
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


def theVectors(dat, pixelarray, cameraMatrix, distCoeffs):
    xvect = [dat[0], dat[1], dat[2]]

    yvect = [dat[3], dat[4], dat[5]]
    zvect = [dat[6], dat[7], dat[8]]
    CamOrgn = [dat[9], dat[10], dat[11]]


    test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
    test[0][0][0] = pixelarray[0]
    test[0][0][1] = pixelarray[1]

    undistort = cv2.undistortPoints(test, cameraMatrix, distCoeffs)
    # print undistort

    UdistortVect = numpy.zeros(2)

    UdistortVect[0] = undistort[0][0][0]
    UdistortVect[1] = undistort[0][0][1]

    pixvectMag = numpy.linalg.norm(UdistortVect)
    pixvect = UdistortVect

    GoalVect = [UdistortVect[0], UdistortVect[1], 1]

    GoalVect = numpy.array(GoalVect)

    GoalVect = GoalVect / numpy.linalg.norm(GoalVect)

    # changing it to the goal np goal vector.

    GoalVectNP = GoalVect

    rotMatrix = [[dat[0], dat[1], dat[2]], [dat[3], dat[4], dat[5]], [dat[6], dat[7], dat[8]]]
    rotMatrix = numpy.array(rotMatrix)

    GoalVectNP = numpy.linalg.inv(rotMatrix).dot(GoalVectNP)  # -numpy.array(CamOrgn))

    CamOrgn = numpy.array(CamOrgn)
    return CamOrgn, GoalVectNP


def findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, WorldPoint):
    cloPo1 = numpy.zeros(3)
    x0 = numpy.zeros(3)
    x1 = numpy.zeros(3)
    x2 = numpy.zeros(3)
    line=numpy.zeros(3)

    x0= numpy.array(WorldPoint)
    x1= numpy.array(CamOrgn1)
    x2=numpy.array(pointvect1)
    x2=x2+x1

    line=x2
    cloPo1=numpy.dot(line,x0-x1)*line/((numpy.linalg.norm(line)**2))
    dis=numpy.linalg.norm(numpy.cross((x0-x1),(x0-x2)))/numpy.linalg.norm((x2-x1))
    #print dis, numpy.linalg.norm((cloPo1-x0))

    length=numpy.sqrt(numpy.linalg.norm(x0-x1)**2-dis**2) ##11/24/2018  changed dis to dis**2
    x2norm=numpy.array(pointvect1)/numpy.linalg.norm(numpy.array(pointvect1))
    cloPo1=x1+length*x2norm
    return dis,cloPo1








def ReturnMeshProjection(World3D,  dat1, cameraMatrix1, distCoeffs1):

    zvect1 = numpy.zeros(3)

    xvect1 = numpy.zeros(3)

    yvect1 = numpy.zeros(3)

    CamOrgn1 = numpy.zeros(3)

    rotVect1 = numpy.zeros(3)

    rot = numpy.zeros(3)
    tran = numpy.zeros(3)

    xvect1[0] = dat1[0]
    xvect1[1] = dat1[1]
    xvect1[2] = dat1[2]

    yvect1[0] = dat1[3]
    yvect1[1] = dat1[4]
    yvect1[2] = dat1[5]

    zvect1[0] = dat1[6]
    zvect1[1] = dat1[7]
    zvect1[2] = dat1[8]

    CamOrgn1[0] = dat1[9]
    CamOrgn1[1] = dat1[10]
    CamOrgn1[2] = dat1[11]

    CamOrgn1 = numpy.array(CamOrgn1)

    zvect1 = numpy.array(zvect1)
    xvect1 = numpy.array(xvect1)
    yvect1 = numpy.array(yvect1)

    cop2 = CamOrgn1 - World3D  # work

    d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

    d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
    d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
    d3test[0][0][2] = 1

    imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

    #print imagePoints2,int(imagePoints2[0][0][0])
    Worldtopixel=[0,0] #numpy.array(2)
    Worldtopixel[0] = int(imagePoints2[0][0][0])
    Worldtopixel[1] = int(imagePoints2[0][0][1])

    return Worldtopixel






###########   orb_slam2 data file
file1 = "G:/belize_nov_trip/maps/example_Glo1804_2_H.h5"
#file1 = 'C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/example_GoProTom_243_H.h5'
#file1 = 'F:/belize_nov_trip/maps/example_GoProTom_159_H.h5'
#file2 = 'F:/belize_nov_trip/maps/example_GoProTom_159_MapEmph.h5'
#file2 = 'C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/example_GoProTom_243_MapEmph.h5'
file2 = "G:/belize_nov_trip/maps/example_Glo1804_2_MapEmph.h5"

#PC = open(filteredfilename1)
#PClines = PC.readlines()
f = h5py.File(file1, 'r')
if os.path.isfile(file2):
    print "file exists"
    fw = h5py.File(file2, 'a')
else:
    fw = h5py.File(file2, 'w')

    fw.create_group("header")
    fw.create_group("frames")
    fw.create_group("mapPoints")

    fw['header'].create_dataset('CameraCalibrationHeader',data=f['header']['CameraCalibrationHeader'])
    fw['header'].create_dataset('CameraCalibrationValue',data=f['header']['CameraCalibrationValue'])
    fw['header'].create_dataset('camera header',data=f['header']['camera header'])
    MapHeader=f['header']['map header']
    MapHeader=numpy.array(MapHeader)
    MapHeader2=numpy.array(MapHeader)

    NewMapHeader=numpy.delete(MapHeader,[3,4,5,6,7],axis=0)

    NewMapHeader[0]="frame number"
    NewMapHeader=numpy.append(NewMapHeader,["Distance to point","ClosestPoint x","ClosestPoint y","ClosestPoint z","Reprojected Pixel x","Reprojected Pixel y"]
    )





    fw['header'].create_dataset('map header',data=NewMapHeader)

    NewMapHeader2 = numpy.delete(MapHeader2, [0,1,2,3,4,8,9,10,11,12], axis=0)

    fw['header'].create_dataset('shared map header',data=NewMapHeader2)

cameraMatrix1 = numpy.zeros((3, 3))
distCoeffs1 = numpy.zeros(5)

cameraMatrix1[0, 0] = f['header']['CameraCalibrationValue'][0]
cameraMatrix1[0, 2] = f['header']['CameraCalibrationValue'][2]
cameraMatrix1[1, 1] = f['header']['CameraCalibrationValue'][1]
cameraMatrix1[1, 2] = f['header']['CameraCalibrationValue'][3]
cameraMatrix1[2, 2] = 1

distCoeffs1[0] = f['header']['CameraCalibrationValue'][4]
distCoeffs1[1] = f['header']['CameraCalibrationValue'][5]
distCoeffs1[2] = f['header']['CameraCalibrationValue'][6]
distCoeffs1[3] = f['header']['CameraCalibrationValue'][7]
distCoeffs1[4] = f['header']['CameraCalibrationValue'][8]




for keylist in f.keys():
    if keylist!="header":

        #if int(keylist.split("F")[1])<30000:
        if int(keylist.split("F")[1]) >= 80000 and int(keylist.split("F")[1]) <100000:
            print keylist
            fw["frames"].create_dataset(keylist,data=f[keylist]["CameraPos"])
            try:
                maplist=f[keylist]["MapData"]

            except KeyError:
                continue
            sh2,sh1=maplist.shape
            #NewList=numpy.zeros(((sh1-3),1))

            dat1 =numpy.array(f[keylist]['CameraPos'][:])

            for j in range(sh2-1):
                NewList = numpy.zeros(((sh1)))

                NewList2 = numpy.zeros((sh1))

                NewList[:]=maplist[j,:]
                NewList2[:]=maplist[j,:]

                #print NewList
                NewList=numpy.delete(NewList,[3,4,5,6,7,13,14,15,16,17],axis=0)

                NewList[0]=int(keylist.split("F")[1])

                NewList2 = numpy.delete(NewList2, [0,1,2,3,4,8,9,10,11,12], axis=0)
                World3D=numpy.array([NewList2[0],NewList2[1],NewList2[2]])

                point1 = numpy.array([NewList[3], NewList[4]])
                CamOrgn1, pointvect1 = theVectors(dat1, point1, cameraMatrix1, distCoeffs1)



                dis, CloPo1=findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, World3D)
                World2pixel=ReturnMeshProjection(World3D, dat1, cameraMatrix1, distCoeffs1)
                NewList=numpy.append(NewList,[dis,CloPo1[0],CloPo1[1],CloPo1[2],World2pixel[0],World2pixel[1]])
                NewLista = numpy.zeros((1, (NewList.shape[0])))
                NewLista[0,:]=NewList[:]
                NewListb=[]
                NewListb.append(NewList)
                #print NewListb

                try:



                    fw["mapPoints"].create_group(str(int(maplist[j,3])))
                    fw["mapPoints"][str(int(maplist[j, 3]))].create_dataset("SharedMapData",data=NewList2)
                    #fw["mapPoints"][str(int(maplist[j, 3]))].create_dataset("MapData", data=NewLista,maxshape=(None, NewList.shape[0]), chunks=True)
                    fw["mapPoints"][str(int(maplist[j, 3]))].create_dataset("MapData", data=NewListb)


                except:
                    #print "was here"
                    #ss1=fw["mapPoints"][str(int(maplist[j, 3]))].shape
                    #print ss1
                    #dataList=numpy.zeros((ss1))
                    #dataList[:]=fw["mapPoints"][str(int(maplist[j, 3]))][:]
                    dataList = fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"][:]


                    dataListb=dataList.tolist()
                    #print  NewList
                    dataListb.append(NewList)
                    #print dataListb
                    #numpy.append(dataList,NewList[0,:],axis=1)
                    #dataList= numpy.vstack((dataList, NewList))
                    #dataList =numpy.vstack((dataList, NewList))


                    del fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"]
                    fw["mapPoints"][str(int(maplist[j, 3]))].create_dataset("MapData", data=dataListb)
                    if 0:
                        fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"].resize((fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"].shape[0]+1),axis=0)
                        #fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"].resize(
                        #    (fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"].shape[0] + NewList.shape[0]), axis=0)

                        fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"][-1:]=NewList
                        #print fw["mapPoints"][str(int(maplist[j, 3]))]["MapData"][:]

                        #print keylist



#numpy.save("mapdic3",d)

