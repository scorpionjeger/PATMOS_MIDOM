import numpy
from datetime import datetime
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
import shutil
import os
import cv2
print cv2.__version__
import h5py
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
from scipy.optimize import minimize  # pip install scipi after removing it from anaconda
from stl import mesh #conda install -c conda-forge numpy-stl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
import vtk
#from scipy.signal import savgol_filter
import platform
from scipy import interpolate
if 1:#MIDOM only
    from shapely.geometry import Polygon
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class animateFromVideo:
    #################################################################################################################3

    def __init__(self):

        self.point = numpy.zeros(2)
        self.point[0] = 700
        self.point[1] = 700
        self.point1 = numpy.zeros(2)
        self.point1[0] = 360
        self.point1[1] = 640
        self.point2 = numpy.zeros(2)
        self.point2[0] = 360
        self.point2[1] = 640
        self.clicked1 = numpy.zeros(3)
        self.clicked1[0] = 0
        self.clicked1[1] = 0
        self.clicked1[2] = 0

        self.clicked2 = numpy.zeros(3)
        self.clicked2[0] = 0
        self.clicked2[1] = 0
        self.clicked2[2] = 0

        self.KeepTrack1 = numpy.array([0L, 960.0, 540.0, 2, 0, 0.44427083333333334, 0, 0, 0, 0, 0, 0])
        self.KeepTrack2 = numpy.array([0L, 960.0, 540.0, 2, 0, 0.44427083333333334, 0, 0, 0, 0, 0, 0])

        self.datOrdAltGlobe = numpy.zeros(12)







    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################


    #  Both squid and VSLAM


    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################







    ###############################################################################################################3

    #
    #                    This trasfrom
    #

    #   framecp= The frame
    #   pointg= the center of clicked point
    #   magg = magnification


    def TransformFrame(self, framecp, pointg, magg, windowsize):

        as_array = framecp  # no
        yf, xf, zf = numpy.shape(as_array)

        # center of the frame

        xcenter = int(xf / 2)
        ycenter = int(yf / 2)
        xcenter = pointg[0]
        ycenter = pointg[1]

        # mag greater than one does work
        mag = magg

        # this seems like a good standard size
        # xDimFinal = 848
        # yDimFinal = 480
        xDimFinal, yDimFinal = windowsize
        FullBlank = numpy.zeros((yDimFinal, xDimFinal, zf), numpy.uint8)

        # linearly changing it from full to the Final size
        xRsize = int(mag * xf)
        yRsize = int(mag * yf)

        if yRsize < yDimFinal:
            yRsize = yDimFinal
            xRsize = int(xf * yDimFinal / float(yf))

        resized_image_uncrop = cv2.resize(framecp, (xRsize, yRsize))
        resizedy, resizedx, resizedz = resized_image_uncrop.shape

        # change the center values.

        xcrop1 = int(xcenter * xRsize / float(xf) - xDimFinal / 2)
        if xcrop1 < 0:
            xcrop1 = 0
        xcrop2 = int(xcenter * xRsize / float(xf) + xDimFinal / 2)
        ycrop1 = int(ycenter * yRsize / float(yf) - yDimFinal / 2)
        if ycrop1 < 0:
            ycrop1 = 0
        ycrop2 = int(ycenter * yRsize / float(yf) + yDimFinal / 2)
        if xcrop2 > xDimFinal and xcrop2 > resizedx:
            xcrop2 = resizedx
        if ycrop2 > yDimFinal and ycrop2 > resizedy:
            ycrop2 = resizedy

        ######################   rounding error
        if xcrop2 - xcrop1 == xDimFinal + 1:
            xcrop2 = xcrop2 - 1
        if ycrop2 - ycrop1 == yDimFinal + 1:
            ycrop2 = ycrop2 - 1


        resized_image = resized_image_uncrop[ycrop1:ycrop2, xcrop1:xcrop2]

        bitmap = resized_image
        ybf, xbf, zbf = bitmap.shape
        FBy1 = ycrop1
        FBy2 = ycrop2
        FBx1 = xcrop1
        FBx2 = xcrop2

        if int(ycrop1) == 0:
            FBy1 = yDimFinal - ycrop2 + ycrop1
            FBy2 = yDimFinal - ycrop2 + ycrop2

        if int(xcrop1) == 0:
            FBx1 = xDimFinal - xcrop2 + xcrop1
            FBx2 = xDimFinal - xcrop2 + xcrop2

        if int(ycrop2) >= yDimFinal:
            FBy1 = -(ycrop1) + ycrop1
            FBy2 = - ycrop1 + ycrop2

        if int(xcrop2) >= xDimFinal:
            FBx1 = -(xcrop1) + xcrop1
            FBx2 = - xcrop1 + xcrop2
        FullBlank[FBy1:FBy2, FBx1:FBx2, 0:3] = bitmap

        KeepTrack = [mag, xcenter, ycenter, xcrop1, ycrop1, xRsize / float(xf), xDimFinal, yDimFinal, xcrop2, ycrop2,
                     resizedx, resizedy]
        del as_array
        del framecp
        del resized_image
        return FullBlank, KeepTrack

    #################################################################################################################


    def unTransformPoint(self, x, y, KeepTrack):
        mag = KeepTrack[0]
        xcenter = KeepTrack[1]
        ycenter = KeepTrack[2]
        xcrop1 = KeepTrack[3]
        ycrop1 = KeepTrack[4]
        Ratio = KeepTrack[5]
        xDimFinal = KeepTrack[6]
        yDimFinal = KeepTrack[7]
        xcrop2 = KeepTrack[8]
        ycrop2 = KeepTrack[9]
        resizedx = KeepTrack[10]
        resizedy = KeepTrack[11]
        if xcrop1 != 0:
            xf0 = (xcrop1 + x) / Ratio
        if xcrop1 == 0:
            x = x - (xDimFinal - xcrop2)
            xf0 = (xcrop1 + x) / Ratio
        if int(xcrop2) > xDimFinal:
            xf0 = (xcrop1 + x) / Ratio

        if ycrop1 != 0:
            yf0 = (ycrop1 + y) / Ratio
        if ycrop1 == 0:
            y = y - (yDimFinal - ycrop2)
            yf0 = (ycrop1 + y) / Ratio
        if int(ycrop2) > yDimFinal:
            yf0 = (ycrop1 + y) / Ratio

        return xf0, yf0

    ###############################################################################################################


    def TransformPoint(self, x, y, KeepTrack):
        mag = KeepTrack[0]
        xcenter = KeepTrack[1]
        ycenter = KeepTrack[2]
        xcrop1 = KeepTrack[3]
        ycrop1 = KeepTrack[4]
        Ratio = KeepTrack[5]
        xDimFinal = KeepTrack[6]
        yDimFinal = KeepTrack[7]
        xcrop2 = KeepTrack[8]
        ycrop2 = KeepTrack[9]
        resizedx = KeepTrack[10]
        resizedy = KeepTrack[11]

        if xcrop1 != 0:
            xf0 = (-xcrop1 + Ratio * x)
        if xcrop1 == 0:
            xf0 = (-xcrop1 + Ratio * x) + (xDimFinal - xcrop2)
        if int(xcrop2) >= xDimFinal:
            xf0 = (-xcrop1 + Ratio * x)  #
        if int(xcrop2) > xDimFinal and xcrop2 > resizedx:
            xf0 = (-xcrop1 + Ratio * x)  #
        if ycrop1 != 0:
            yf0 = (-ycrop1 + Ratio * y)
        if ycrop1 == 0:
            yf0 = (-ycrop1 + Ratio * y) + (yDimFinal - ycrop2)
        if int(ycrop2) >= yDimFinal:
            yf0 = (-ycrop1 + Ratio * y)  # -ycrop1

        return xf0, yf0

    ###############################################################################################################








    ####################################################################################################################

    def select_point(self, event, x, y, flags, param):
        self.point1, self.KeepTrackm, self.clicked = param
        theapp = animateFromVideo()
        if event == cv2.EVENT_FLAG_SHIFTKEY:
            xx, yy = theapp.unTransformPoint(x, y, self.KeepTrackm)

            print "mousing", xx, yy
        if event == cv2.EVENT_LBUTTONDBLCLK:
        #if event == cv2.EVENT_RBUTTONDOWN:
            xx, yy = theapp.unTransformPoint(x, y, self.KeepTrackm)
            self.point1[0] = xx
            self.point1[1] = yy
            print "mousing", xx, yy

        if event == cv2.EVENT_MOUSEMOVE:
            xx, yy = theapp.unTransformPoint(x, y, self.KeepTrackm)
            self.clicked[1] = xx
            self.clicked[2] = yy
            self.clicked[0] = 1

            #########################################################################

    def rewind(self, num, cap):
        self.rewindMSEC = cap.get(cv2.CAP_PROP_POS_FRAMES)

        self.rewindMSEC = max(self.rewindMSEC - num - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.rewindMSEC)

    ############################################################################


    def fastforward(self, num, cap):
        self.rewindMSEC = cap.get(cv2.CAP_PROP_POS_FRAMES)

        self.rewindMSEC = self.rewindMSEC + num - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.rewindMSEC)

    #################################################################################################33









    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################


    # VSLAM  only


    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################


    def ConvertMapDataintoH5(self,filename,filewrite):

        Moredata = False
        print "PC = open(filename)"
        PC = open(filename)

        Header = PC.readline()
        print Header
        CameraCalibration = PC.readline()
        print CameraCalibration
        line = CameraCalibration

        hf = h5py.File(filewrite, 'w')

        mapdataheader = 18  # was 13
        data = numpy.zeros(12)
        mapdata = numpy.zeros((1, mapdataheader))
        mapdataInsert = numpy.zeros((1, mapdataheader))
        kIt = 0
        kStart = 0

        g1 = hf.create_group('header')

        datas = [""] * 12

        for i in range(len(Header.split("|"))):
            if i > mapdataheader:
                datas[i - (mapdataheader + 1)] = Header.split("|")[i]

        g1.create_dataset('camera header', data=datas)
        print datas

        datas = [""] * mapdataheader

        for i in range(len(Header.split("|"))):
            if i > 0 and i < mapdataheader + 1:
                datas[i - 1] = Header.split("|")[i]

        g1.create_dataset('map header', data=datas)
        print datas

        datas = [""] * 9

        for i in range(len(CameraCalibration.split("|")) - 1):
            print numpy.floor(i / 2), float(i) / 2
            if float(i) / 2 == numpy.floor(i / 2):
                datas[i / 2] = CameraCalibration.split("|")[i]

        g1.create_dataset('CameraCalibrationHeader', data=datas)
        print datas

        data = numpy.zeros(9)

        for i in range(len(CameraCalibration.split("|"))):
            if float(i) / 2 != numpy.floor(i / 2):
                data[i / 2] = float(CameraCalibration.split("|")[i])

        g1.create_dataset('CameraCalibrationValue', data=data)
        print data

        data = numpy.zeros(12)

        line = PC.readline()
        lineFormer = line
        while line:

            # if k>=kStart:
            if 1:


                if line.split("|")[1] == "":
                    print 'F' + line.split("|")[0]
                    g1 = hf.create_group('F' + line.split("|")[0])
                    for i in range(len(Header.split("|")) + 1):
                        if i > mapdataheader + 1:
                            data[i - (mapdataheader + 2)] = float(line.split("|")[i])

                    g1.create_dataset("CameraPos", data=data)

                    data = numpy.zeros(12)

                if Moredata:
                    if lineFormer.split("|")[1] != "" and line.split("|")[1] != "":
                        for i in range(len(Header.split("|"))):
                            if i > 0 and i < mapdataheader + 1:
                                mapdataInsert[0][i - 1] = float(lineFormer.split("|")[i])

                        if kIt == 0:
                            mapdata[0][:] = mapdataInsert[0][:]

                        else:

                            mapdata = numpy.append(mapdata, mapdataInsert, axis=0)

                        kIt += 1

                    if lineFormer.split("|")[1] != "" and line.split("|")[1] == "":

                        if 1:
                            for i in range(len(Header.split("|"))):
                                if i > 0 and i < mapdataheader + 1:
                                    mapdataInsert[0][i - 1] = float(lineFormer.split("|")[i])

                            if kIt == 0:
                                mapdata[0][:] = mapdataInsert[0][:]

                            else:

                                mapdata = numpy.append(mapdata, mapdataInsert, axis=0)

                            g1.create_dataset("MapData", data=mapdata)  # non compressed
                            kIt = 0
                            mapdata = numpy.zeros((1, mapdataheader))
                            mapdataInsert = numpy.zeros((1, mapdataheader))

                lineFormer = line
                line = PC.readline()

    ##########################################################33
    ##########################################################33
    ##########################################################33
    ##########################################################33
##########################################################33
    def MakeYaml(self,YAMLprojDic,YAMLpath,DicORBdetails):

        File2ORB=DicORBdetails["FiletoORB_SLAM2_output"]

        PathWrite = open(YAMLpath, "w")
        print YAMLprojDic["Camera.fx"][0]

        PathWrite.write("%YAML:1.0\n")
        PathWrite.write("\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("# Camera Parameters. Adjust them!\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("\n")
        PathWrite.write("# Camera calibration and distortion parameters (OpenCV) \n")
        PathWrite.write("Camera.fx: "+str(YAMLprojDic["Camera.fx"][0])+"\n")
        PathWrite.write("Camera.fy: "+str(YAMLprojDic["Camera.fy"][0])+"\n")
        PathWrite.write("Camera.cx: "+str(YAMLprojDic["Camera.cx"][0])+"\n")
        PathWrite.write("Camera.cy: "+str(YAMLprojDic["Camera.cy"][0])+"\n")
        PathWrite.write("\n")
        PathWrite.write("Camera.k1: "+str(YAMLprojDic["Camera.k1"][0])+"\n")
        PathWrite.write("Camera.k2: "+str(YAMLprojDic["Camera.k2"][0])+"\n")
        PathWrite.write("Camera.p1: "+str(YAMLprojDic["Camera.p1"][0])+"\n")
        PathWrite.write("Camera.p2: "+str(YAMLprojDic["Camera.p2"][0])+"\n")
        PathWrite.write("Camera.k3: "+str(YAMLprojDic["Camera.k3"][0])+"\n")
        PathWrite.write("\n")
        PathWrite.write("# Camera frames per second \n")
        PathWrite.write("Camera.fps: "+str(YAMLprojDic["Camera.fps"][0])+" \n")
        PathWrite.write("\n")
        PathWrite.write("# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)\n")
        PathWrite.write("Camera.RGB: "+str(int(YAMLprojDic["Camera.rgb"][0]))+"\n")
        PathWrite.write("\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("# ORB Parameters\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("\n")
        PathWrite.write("# ORB Extractor: Number of features per image\n")
        PathWrite.write("ORBextractor.nFeatures: "+str(int(YAMLprojDic["ORBextractor.nFeatures"][0]))+"\n")
        PathWrite.write("\n")
        PathWrite.write("# ORB Extractor: Scale factor between levels in the scale pyramid 	\n")
        PathWrite.write("ORBextractor.scaleFactor: "+str(YAMLprojDic["ORBextractor.scaleFactor"][0])+"\n")
        PathWrite.write("\n")
        PathWrite.write("# ORB Extractor: Number of levels in the scale pyramid	\n")
        PathWrite.write("ORBextractor.nLevels: "+str(int(YAMLprojDic["ORBextractor.nLevels"][0]))+"\n")
        PathWrite.write("\n")
        PathWrite.write("# ORB Extractor: Fast threshold\n")
        PathWrite.write("# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.\n")
        PathWrite.write("# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST\n")
        PathWrite.write("# You can lower these values if your images have low contrast\n")
        PathWrite.write("ORBextractor.iniThFAST: "+str(int(YAMLprojDic["ORBextractor.iniThFAST"][0]))+"\n")
        PathWrite.write("ORBextractor.minThFAST: "+str(int(YAMLprojDic["ORBextractor.minThFAST"][0]))+"\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("# Viewer Parameters\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("Viewer.KeyFrameSize: "+str(YAMLprojDic["Viewer.KeyFrameSize"][0])+"\n")
        PathWrite.write("Viewer.KeyFrameLineWidth: "+str(int(YAMLprojDic["Viewer.KeyFrameLineWidth"][0]))+"\n")
        PathWrite.write("Viewer.GraphLineWidth: "+str(int(YAMLprojDic["Viewer.GraphLineWidth"][0]))+"\n")
        PathWrite.write("Viewer.PointSize: "+str(int(YAMLprojDic["Viewer.PointSize"][0]))+"\n")
        PathWrite.write("Viewer.CameraSize: "+str(YAMLprojDic["Viewer.CameraSize"][0])+"\n")
        PathWrite.write("Viewer.CameraLineWidth: "+str(int(YAMLprojDic["Viewer.CameraLineWidth"][0]))+"\n")
        PathWrite.write("Viewer.ViewpointX: "+str(int(YAMLprojDic["Viewer.ViewpointX"][0]))+"\n")
        PathWrite.write("Viewer.ViewpointY: "+str(int(YAMLprojDic["Viewer.ViewpointY"][0]))+"\n")
        PathWrite.write("Viewer.ViewpointZ: "+str(int(YAMLprojDic["Viewer.ViewpointZ"][0]))+"\n")
        PathWrite.write("Viewer.ViewpointF: "+str(int(YAMLprojDic["Viewer.ViewpointF"][0]))+"\n")
        PathWrite.write("\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("# Map Parameters\n")
        PathWrite.write("#--------------------------------------------------------------------------------------------\n")
        PathWrite.write("Map.mapfile: " + File2ORB + "/"+DicORBdetails["map_in"] + "\n")
        PathWrite.write("Map.mapfileOut: " + File2ORB + "/" + DicORBdetails["map_out"] + "\n")
        PathWrite.write("Map.KFTfile: " + File2ORB + "/" + DicORBdetails["ORB_pointcloud_output"] + "\n")
        PathWrite.write("Map.FirstFrame: " + str(DicORBdetails["Starting_frame"]) + "\n")
        PathWrite.write("Map.Camera1Placement: " + File2ORB +"/" + DicORBdetails["Camera1_metadata"] + "\n")
        PathWrite.write("Map.Camera2Placement: " + File2ORB + "/" + DicORBdetails["Camera2_metadata"] + "\n")
        PathWrite.write("\n")
        PathWrite.write("\n")
        PathWrite.write("\n")
        PathWrite.write("\n")






        PathWrite.close()




####################################################################################################3
    ####################################################################################################3
    ####################################################################################################3
    ####################################################################################################3
    # Plotting 3D plots for insects



    def Plot3Dinsect(self,fwData, ResultRot, ResultTran, inkey):
        cameraNames = ["camera1", "camera2", "combined"]

        #plotting with fiducial
        if 1:
            try:
                try:
                    ResultRot = numpy.array(
                        fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])
                    ResultRot = ResultRot.T

                    ResultTran = numpy.array(
                        fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                except:
                    ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])#l;l

                    ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])
            except:
                None



        #actual points
        inX = []
        inY = []
        inZ = []

        for insectCom in fwData[cameraNames[2]]:
            if insectCom == "insect"+str(inkey)+"_"+"insect"+str(inkey):
                print insectCom, "insectCom"
                for jj in fwData[cameraNames[2]][insectCom]:
                    if jj != "FrameDelay":
                        if 1:
                            InterPoint = numpy.array(fwData[cameraNames[2]][insectCom][jj]['3Dpoint'])
                            InterPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran

                            inX.append(InterPoint[0])
                            inY.append(InterPoint[1])
                            inZ.append(InterPoint[2])


        #moving average points
        inXk = []
        inYk = []
        inZk = []
        inXkR = []
        inYkR = []
        inZkR = []
        fullPoints=[]
        if 1:
            for insectCom in fwData[cameraNames[2]]:
                if insectCom == "insect"+str(inkey)+"_"+"insect"+str(inkey):
                    for jj in fwData[cameraNames[2]][insectCom]:
                        if jj != "FrameDelay":
                            if 1:
                                try:
                                    InterPointRaw = numpy.array(fwData[cameraNames[2]][insectCom][jj]['MovingAveragePoint'])
                                    print InterPointRaw[0], InterPointRaw[1], InterPointRaw[2]
                                    InterPoint = numpy.array([InterPointRaw[0], InterPointRaw[1], InterPointRaw[2]])

                                    InterPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran

                                    fullPoints.append(InterPoint)
                                    print InterPoint

                                    try:
                                        fwData[cameraNames[2]][insectCom][jj]['IsMirror']
                                        inXkR.append(InterPoint[0])
                                        inYkR.append(InterPoint[1])
                                        inZkR.append(InterPoint[2])
                                    except:

                                        inXk.append(InterPoint[0])
                                        inYk.append(InterPoint[1])
                                        inZk.append(InterPoint[2])

                                except:
                                    None

        #############################################################################33
        #??????
        fullPoints=numpy.array(fullPoints)
        print fullPoints
        saveFile = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/"
        numpy.save(saveFile + "blenderPathTrial.npy", fullPoints)


        print inXk
        fig = plt.figure()
        ax = fig.gca(projection='3d')



        if 0:#do reflection part
            inkey2 = 60
            inkey = 41
            point1 = numpy.zeros(2)
            point1r = numpy.zeros(2)

            inkeyName1 = "insect" + str(inkey) + "_" + "insect" + str(inkey)
            inkeyName2 = "insect" + str(inkey2) + "_" + "insect" + str(inkey2)

            insect = "insect" + str(inkey)
            insectR = "insect" + str(inkey2)
            inXr=[]
            inYr=[]
            inZr=[]
            for i in fwData[cameraNames[2]][insect + "_R_" + insectR].keys():

                realPoint=fwData[cameraNames[2]][insect + "_R_" + insectR][
                    str(int(i))]['pointFromReflection']
                InterPoint = numpy.array([realPoint[0], realPoint[1], realPoint[2]])

                realPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran

                inXr.append(realPoint[0])
                inYr.append(realPoint[1])
                inZr.append(realPoint[2])

            sd = ax.scatter(inXr, inYr, inZr, c=(.0, .1, .2), marker=".", s=2, lw=5)


        if 0:# geting fixed points for caloperix, nik_869_Glo_1915
            femaledecoy=[]
            inkey=7
            for insectCom in fwData[cameraNames[2]]:
                if insectCom == "insect" + str(inkey) + "_" + "insect" + str(inkey):
                    print insectCom, "insectCom"
                    for jj in fwData[cameraNames[2]][insectCom]:
                        if jj != "FrameDelay":
                            if 1:
                                InterPoint = numpy.array(fwData[cameraNames[2]][insectCom][jj]['3Dpoint'])
                                InterPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran
                                femaledecoy.append(InterPoint)

            branch=[]
            inkey=8
            for insectCom in fwData[cameraNames[2]]:
                if insectCom == "insect" + str(inkey) + "_" + "insect" + str(inkey):
                    print insectCom, "insectCom"
                    for jj in fwData[cameraNames[2]][insectCom]:
                        if jj != "FrameDelay":
                            if 1:
                                InterPoint = numpy.array(fwData[cameraNames[2]][insectCom][jj]['3Dpoint'])
                                InterPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran
                                branch.append(InterPoint)

            Shadow=[]
            inkey=9
            for insectCom in fwData[cameraNames[2]]:
                if insectCom == "insect" + str(inkey) + "_" + "insect" + str(inkey):
                    print insectCom, "insectCom"
                    for jj in fwData[cameraNames[2]][insectCom]:
                        if jj != "FrameDelay":
                            if 1:
                                InterPoint = numpy.array(fwData[cameraNames[2]][insectCom][jj]['3Dpoint'])
                                InterPoint = numpy.matmul(ResultRot.T, InterPoint) + ResultTran
                                Shadow.append(InterPoint)
            Shadow=numpy.array(Shadow)
            branch = numpy.array(branch)
            femaledecoy = numpy.array(femaledecoy)

            Shadowave=numpy.mean(Shadow,axis=0)
            branchave=numpy.mean(branch,axis=0)
            femaledecoy=numpy.mean(femaledecoy, axis=0)

            print Shadowave,branchave,femaledecoy

            sunpoint=((branchave-Shadowave)/numpy.linalg.norm(branchave-Shadowave))*15+femaledecoy


        sdk = ax.scatter(inXk, inYk, inZk, c=(.9, .1, .8), marker=".", s=2, lw=10)
        sdk = ax.scatter(inXkR, inYkR, inZkR, c=(0,0,0), marker=".", s=2, lw=10)
        #sdf = ax.scatter(femaledecoy[0], femaledecoy[1], femaledecoy[2], c=(0, 0, 0), marker=".", s=10, lw=10)

        #sds = ax.scatter(sunpoint[0], sunpoint[1], sunpoint[2], c=(.85, .85, 0), marker=".", s=10, lw=10)

        #sdl =ax.plot([sunpoint[0],femaledecoy[0]],[sunpoint[1],femaledecoy[1]], [sunpoint[2],femaledecoy[2]], c=(1, 1, 0.0), linewidth=5)

        inX = numpy.array(inX)
        inY = numpy.array(inY)
        inZ = numpy.array(inZ)

        inXmax = numpy.max(inX)
        inXmin = numpy.min(inX)

        inYmax = numpy.max(inY)
        inYmin = numpy.min(inY)

        inZmax = numpy.max(inZ)
        inZmin = numpy.min(inZ)

        rangeins = numpy.array([inXmax - inXmin, inYmax - inYmin, inZmax - inZmin])
        rangeinsMax = numpy.max(rangeins)
        rangeinsMaxInd = numpy.argmax(rangeins)
        print rangeins

        if rangeinsMaxInd == 0:
            ax.set_xlim3d(inXmin, inXmax)
            ax.set_ylim3d(numpy.mean(inY) - (inXmax - inXmin) / 2, numpy.mean(inY) + (inXmax - inXmin) / 2)
            ax.set_zlim3d(numpy.mean(inZ) - (inXmax - inXmin) / 2, numpy.mean(inZ) + (inXmax - inXmin) / 2)

        if rangeinsMaxInd == 1:
            ax.set_ylim3d(inYmin, inYmax)
            ax.set_xlim3d(numpy.mean(inX) - (inYmax - inYmin) / 2, numpy.mean(inX) + (inYmax - inYmin) / 2)
            ax.set_zlim3d(numpy.mean(inZ) - (inYmax - inYmin) / 2, numpy.mean(inZ) + (inYmax - inYmin) / 2)

        if rangeinsMaxInd == 2:
            ax.set_zlim3d(inZmin, inZmax)
            ax.set_ylim3d(numpy.mean(inY) - (inZmax - inZmin) / 2, numpy.mean(inY) + (inZmax - inZmin) / 2)
            ax.set_xlim3d(numpy.mean(inX) - (inZmax - inZmin) / 2, numpy.mean(inX) + (inZmax - inZmin) / 2)


        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        print "here"
        plt.show()
        print "stuck"
        #plt.close()






    def Plot3Dinsect2(self,PlotDesciptiveDic,DictA,DictErr,thePlatformNum):
        cameraNames = ["camera1", "camera2", "combined"]

        #print PlotDesciptiveDic["insectNumGroupNumber"],"PlotDesciptiveDic[insectNumGroupNumber]"
        fig3D = plt.figure()
        ax = fig3D.gca(projection='3d')

        PlotEllipsoid=False
        if PlotDesciptiveDic["OutputCVSMeshLab"]=="y":
            PathWrite = open(PlotDesciptiveDic["Path2SaveImages"] + "/" + PlotDesciptiveDic["SaveFileName"] + "_CSV_MeshLab.txt",
                             "w")
        def plot_ellipsoid_3d(ellctr,ellaxes, ax):
            """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

            # points on unit sphere
            u = numpy.linspace(0.0, 2.0 * numpy.pi, 40)
            v = numpy.linspace(0.0, numpy.pi, 40)
            z = numpy.outer(numpy.cos(u), numpy.sin(v))
            y = numpy.outer(numpy.sin(u), numpy.sin(v))
            x = numpy.outer(numpy.ones_like(u), numpy.cos(v))

            # transform points to ellipsoid
            for i in range(len(x)):
                for j in range(len(x)):
                    x[i, j], y[i, j], z[i, j] = ellctr + numpy.dot(ellaxes,
                                                                 [x[i, j], y[i, j], z[i, j]])

            ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='#2980b9', alpha=0.2)




        if PlotDesciptiveDic["DisplaySTLin3Dplot"]=="y":
            #your_mesh = mesh.Mesh.from_file('C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/tomGP318mapcroptetcWater.stl')

            #your_mesh = mesh.Mesh.from_file(
            #    'C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/KeyFrameTrajectory_tom_318_jill_261_190115.stl')
            your_mesh = mesh.Mesh.from_file(PlotDesciptiveDic["PathOfSTLtoDisplay"])
            #ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, edgecolor="b",facecolors="none",alpha=.2))
            ax.add_collection3d(
                mplot3d.art3d.Line3DCollection(your_mesh.vectors,linewidths=0.2, linestyles=':', colors="r"))

        inX=[]
        inY=[]
        inZ=[]
        StumpKnob=PlotDesciptiveDic["PointofReference"]
        StumpKnob=numpy.array(StumpKnob)
        colorVarString=PlotDesciptiveDic["PathColorGradientString"]
        LinecolorVarString=PlotDesciptiveDic["ConnectLinesColorGradientString"]
        if numpy.linalg.norm(StumpKnob)>0.0000001:
            sdk = ax.scatter(StumpKnob[0], StumpKnob[1], StumpKnob[2], c=(0,0,0), marker=".", s=2,
                             lw=15)

        ##############################################################################
        if colorVarString!="":
            inColorTotal=[]
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                print str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)]),"PlotDesciptiveDic"
                for i in range(len(PlotDesciptiveDic["Insectnumbers1"])):
                    inColor=DictA[colorVarString + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                    inColorTotal+=list(inColor)

            inColorTotal=numpy.array(inColorTotal)
            inColorMax=numpy.nanmax(inColorTotal)
            inColorMin = numpy.nanmin(inColorTotal)
            print inColorMax,inColorMin,"max min"
            aCol=1/(inColorMax-inColorMin)
            bCol=-inColorMin/(inColorMax-inColorMin)
        #######################################################################################33
        if LinecolorVarString!="" and PlotDesciptiveDic["insectNumGroupNumber"]==2:
            inColorTotal=[]
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                print str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)]),"PlotDesciptiveDic"
                for i in range(len(PlotDesciptiveDic["Insectnumbers1"])):
                    inColor=DictA[
                        LinecolorVarString.split("-")[0] + LinecolorVarString.split("-")[1] + "-" + LinecolorVarString.split("-")[
                            2] + "__" + str(
                            PlotDesciptiveDic["Insectnumbers" + LinecolorVarString.split("-")[1]][i]) + "-" + str(
                            PlotDesciptiveDic["Insectnumbers" + LinecolorVarString.split("-")[2]][i])]

                    inColorTotal+=list(inColor)

            inColorTotal=numpy.array(inColorTotal)
            inColorMax=numpy.nanmax(inColorTotal)
            inColorMin = numpy.nanmin(inColorTotal)
            print inColorMax,inColorMin,"max min"
            LineaCol=1/(inColorMax-inColorMin)
            LinebCol=-inColorMin/(inColorMax-inColorMin)

        for i in range(len(PlotDesciptiveDic["Insectnumbers1"])):
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):

                inXk =DictA["xt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                inYk =DictA["yt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                inZk =DictA["zt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                #print len(inXk),"len inXk"
                if PlotDesciptiveDic["InsectSizes" + str(ii + 1)][i]!="":
                    Isize=float(PlotDesciptiveDic["InsectSizes" + str(ii + 1)][i])
                else:
                    Isize=4
                if colorVarString == "":
                    if 1:
                        if PlotDesciptiveDic["ColorGradientDistinguishingMultiplePaths"] != "":
                            if thePlatformNum == 0:
                                sdk = ax.scatter(inXk, inYk, inZk,
                                                 c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i],
                                                 marker=".",
                                                 s=4, lw=Isize)

                            elif thePlatformNum == 1:
                                sdk = ax.scatter(inXk, inYk, inZk, c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i], edgecolors=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i],marker=".", s=4, lw=Isize)

                        else:
                            #
                            if thePlatformNum == 0:
                                sdk = ax.scatter(inXk, inYk, inZk,
                                                 c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i], marker="o",
                                                 s=Isize, lw=0)
                            elif thePlatformNum == 1:
                                sdk = ax.scatter(inXk, inYk, inZk, c=PlotDesciptiveDic["InsectColors" + str(ii + 1)][i],
                                                 marker="o",
                                                 s=Isize, lw=0)

                    #sdk = ax.scatter(inXk, inYk, inZk, c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i], marker="o",
                    #                 s=20,edgecolors="r", lw=Isize)

                    if 0:
                        for juu in range(0, len(inXk), 1):
                            sdk = ax.scatter(inXk[juu], inYk[juu], inZk[juu], c="r", marker=".",
                                             s=4, lw=8)

                        sdk = ax.scatter(inXk, inYk, inZk, c="b", marker="o",
                                         s=40, lw=1)

                    if PlotDesciptiveDic["OutputCVSMeshLab"]=="y":
                        for j in range(len(inXk)):
                            if str(inXk[j]) != "nan":
                                PathWrite.write(
                                    str(inXk[j]) + " " + str(inYk[j]) + " " + str(inZk[j]) +" "+str(PlotDesciptiveDic["InsectColors" + str(ii + 1)][i][2])
                                    +" "+str(PlotDesciptiveDic["InsectColors" + str(ii + 1)][i][1])+" " +str(PlotDesciptiveDic["InsectColors" + str(ii + 1)][i][0])+ "\n")
                else:
                    # colorVarString="vavg"
                    inColor = DictA[
                        colorVarString + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                    inColor = numpy.array(inColor)

                    if len(inColor) < len(inXk):
                        inColor = numpy.insert(inColor, 1, inColor[0])
                    if len(inColor) < len(inXk):
                        inColor = numpy.insert(inColor, len(inColor), inColor[len(inColor) - 1])
                    if thePlatformNum == 0:
                        sdk = ax.scatter(inXk, inYk, inZk, c=plt.cm.rainbow(aCol * inColor + bCol), marker=".",
                                         s=Isize, lw=2)

                    elif thePlatformNum == 1:
                        sdk = ax.scatter(inXk, inYk, inZk, c=plt.cm.rainbow(aCol * inColor + bCol), marker="o",
                                         s=Isize, lw=0)
                    #print len(inXk), len(inColor),"3dplot"
                    #print inColor
                    if PlotDesciptiveDic["OutputCVSMeshLab"]=="y":
                        for j in range(len(inXk)):
                            if str(inXk[j]) != "nan":
                                #print "plt.cm.rainbow(aCol*inColor[j]+bCol)[2]",plt.cm.rainbow(aCol*inColor[j]+bCol)[0]
                                PathWrite.write(
                                    str(inXk[j]) + " " + str(inYk[j]) + " " + str(inZk[j]) + " "+str(plt.cm.rainbow(aCol*inColor[j]+bCol)[2])
                                                                                            +" "+str(plt.cm.rainbow(aCol*inColor[j]+bCol)[1])
                                                                                            +" "+str(plt.cm.rainbow(aCol*inColor[j]+bCol)[0])+"\n")
                if 0:
                    for ju in range(len(inXk)):
                        if str(inXk[ju])=="nan":
                            None
                        else:
                            sdkk = ax.scatter(inXk[ju], inYk[ju], inZk[ju], c="k", marker=".", s=10, lw=Isize+10)
                            break


                #############
                #Camera drawing
                DrawCamera=False
                if DrawCamera==True:
                    if ii==0:
                        #print "PlotDesciptiveDic str(ii + 1)][i]",PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i]
                        C1O=DictA[
                                "C1O" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                        C1x=DictA[
                                "C1x" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                        C1y=DictA[
                                "C1y" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                        C1z=DictA[
                                "C1z" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]


                        C2O=DictA[
                                "C2O" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]

                        C2x=DictA[
                                "C2x" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                        C2y=DictA[
                                "C2y" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                        C2z=DictA[
                                "C2z" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]

                        #aqwq
                        #print "camera angle;", numpy.arccos(-C1z[2])*180/3.14159,";",numpy.arccos(-C2z[2])*180/3.14159

                        bxo=20
                        bxw=10
                        bxh=6

                        P10=C1O+bxo*C1z+bxw*C1x+bxh*C1y
                        P11 = C1O + bxo * C1z - bxw * C1x + bxh * C1y
                        P12 = C1O + bxo * C1z - bxw * C1x - bxh * C1y
                        P13 = C1O + bxo * C1z + bxw * C1x - bxh * C1y

                        cam1X=[]
                        cam1Y=[]
                        cam1Z=[]

                        cam1X.append([C1O[0], P10[0]])
                        cam1Y.append([C1O[1], P10[1]])
                        cam1Z.append([C1O[2], P10[2]])

                        cam1X.append([C1O[0], P11[0]])
                        cam1Y.append([C1O[1], P11[1]])
                        cam1Z.append([C1O[2], P11[2]])

                        cam1X.append([C1O[0], P12[0]])
                        cam1Y.append([C1O[1], P12[1]])
                        cam1Z.append([C1O[2], P12[2]])

                        cam1X.append([C1O[0], P13[0]])
                        cam1Y.append([C1O[1], P13[1]])
                        cam1Z.append([C1O[2], P13[2]])

                        cam1X.append([P13[0], P10[0]])
                        cam1Y.append([P13[1], P10[1]])
                        cam1Z.append([P13[2], P10[2]])

                        cam1X.append([P10[0], P11[0]])
                        cam1Y.append([P10[1], P11[1]])
                        cam1Z.append([P10[2], P11[2]])

                        cam1X.append([P11[0], P12[0]])
                        cam1Y.append([P11[1], P12[1]])
                        cam1Z.append([P11[2], P12[2]])

                        cam1X.append([P12[0], P13[0]])
                        cam1Y.append([P12[1], P13[1]])
                        cam1Z.append([P12[2], P13[2]])



                        P20 = C2O + bxo * C2z + bxw * C2x + bxh * C2y
                        P21 = C2O + bxo * C2z - bxw * C2x + bxh * C2y
                        P22 = C2O + bxo * C2z - bxw * C2x - bxh * C2y
                        P23 = C2O + bxo * C2z + bxw * C2x - bxh * C2y

                        cam2X = []
                        cam2Y = []
                        cam2Z = []

                        cam2X.append([C2O[0], P20[0]])
                        cam2Y.append([C2O[1], P20[1]])
                        cam2Z.append([C2O[2], P20[2]])

                        cam2X.append([C2O[0], P21[0]])
                        cam2Y.append([C2O[1], P21[1]])
                        cam2Z.append([C2O[2], P21[2]])

                        cam2X.append([C2O[0], P22[0]])
                        cam2Y.append([C2O[1], P22[1]])
                        cam2Z.append([C2O[2], P22[2]])

                        cam2X.append([C2O[0], P23[0]])
                        cam2Y.append([C2O[1], P23[1]])
                        cam2Z.append([C2O[2], P23[2]])

                        cam2X.append([P23[0], P20[0]])
                        cam2Y.append([P23[1], P20[1]])
                        cam2Z.append([P23[2], P20[2]])

                        cam2X.append([P20[0], P21[0]])
                        cam2Y.append([P20[1], P21[1]])
                        cam2Z.append([P20[2], P21[2]])

                        cam2X.append([P21[0], P22[0]])
                        cam2Y.append([P21[1], P22[1]])
                        cam2Z.append([P21[2], P22[2]])

                        cam2X.append([P22[0], P23[0]])
                        cam2Y.append([P22[1], P23[1]])
                        cam2Z.append([P22[2], P23[2]])

                        if 0:
                            sdkkk = ax.scatter(C2O[0], C2O[1], C2O[2],
                                               c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i],
                                               marker="o", s=10, lw=0)
                            sdkkk = ax.scatter(C1O[0], C1O[1], C1O[2],
                                               c=PlotDesciptiveDic["InsectMultColors" + str(ii + 1)][i],
                                               marker="o", s=10, lw=0)

                        if 1:
                            inColor = DictA[
                                colorVarString + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)]
                            inColor = numpy.array(inColor)

                            #sdkkk = ax.scatter(C2O[0], C2O[1],C2O[2], c=plt.cm.rainbow(aCol * inColor[0] + bCol),
                            #                   marker="o", s=10, lw=0)

                            for ic in range(len(cam1X)):
                                ax.plot(cam1X[ic], cam1Y[ic], cam1Z[ic],  c=plt.cm.rainbow(aCol * inColor[0] + bCol), linewidth=1)

                            for ic in range(len(cam2X)):
                                ax.plot(cam2X[ic], cam2Y[ic], cam2Z[ic],  c=plt.cm.rainbow(aCol * inColor[0] + bCol), linewidth=1)

                            #sdkkk = ax.scatter(C1O[0], C1O[1],C1O[2], c=plt.cm.rainbow(aCol * inColor[0] + bCol),
                            #                   marker="o", s=10, lw=0)

                if PlotEllipsoid==True:
                    for po in range(0,len(DictA["PCAsigma" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                            ii + 1)]),20):
                        PCAsigma=DictA[
                            "PCAsigma" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                ii + 1)][po]
                        PCAmu=DictA[
                            "PCAmu" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                ii + 1)][po]
                        PCAwt=DictA[
                            "PCAwt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                ii + 1)][po]

                        PCAsigma=numpy.array(PCAsigma)
                        PCAmu = numpy.array(PCAmu)
                        PCAwt = numpy.array(PCAwt)
                        if 0:
                            PCAwt[:][0]=PCAwt[:][0]*PCAsigma[0]
                            PCAwt[:][1] = PCAwt[:][1] * PCAsigma[1]
                            PCAwt[:][2] = PCAwt[:][2] * PCAsigma[2]
                        if 1:
                            PCAwt[0][:]=PCAwt[0][:]*PCAsigma[0]
                            PCAwt[1][:] = PCAwt[1][:] * PCAsigma[1]
                            PCAwt[2][:] = PCAwt[2][:] * PCAsigma[2]


                        plot_ellipsoid_3d(PCAmu,PCAwt, ax)

                if PlotDesciptiveDic["UseLines"]=="y" :
                    if PlotDesciptiveDic["insectNumGroupNumber"]==2:
                        if ii == 0:

                            if LinecolorVarString != "":

                                # colorVarString="vavg"

                                LineinColor = DictA[
                                    LinecolorVarString.split("-")[0] + LinecolorVarString.split("-")[1] + "-" +
                                    LinecolorVarString.split("-")[
                                        2] + "__" + str(
                                        PlotDesciptiveDic["Insectnumbers" + LinecolorVarString.split("-")[1]][
                                            i]) + "-" + str(
                                        PlotDesciptiveDic["Insectnumbers" + LinecolorVarString.split("-")[2]][i])]

                                LineinColor = numpy.array(LineinColor)

                                if len(LineinColor) < len(inXk):
                                    LineinColor = numpy.insert(LineinColor, 1, LineinColor[0])
                                if len(LineinColor) < len(inXk):
                                    LineinColor = numpy.insert(LineinColor, len(LineinColor), LineinColor[len(LineinColor) - 1])


                            for tt in range(0,len(inXk),3):
                                px1 = [inXk[tt], DictA["xt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(1 + 1)][i]) + "_" + str(1 + 1)][tt]]
                                py1 = [inYk[tt], DictA["yt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(1 + 1)][i]) + "_" + str(1 + 1)][tt]]
                                pz1 = [inZk[tt], DictA["zt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(1 + 1)][i]) + "_" + str(1 + 1)][tt]]
                                distancePath=numpy.sqrt((px1[0]-px1[1])**2+(py1[0]-py1[1])**2+(pz1[0]-pz1[1])**2)
                                distancePathMax=100
                                #ax.plot(px1, py1, pz1, color=plt.cm.rainbow(distancePath/distancePathMax), linewidth=1)
                                if LinecolorVarString != "":
                                    if numpy.isnan(LineinColor[tt])==False:
                                        ax.plot(px1, py1, pz1, color=plt.cm.rainbow(LineaCol * LineinColor[tt] + LinebCol),
                                                linewidth=1)
                                    else:
                                        ax.plot(px1, py1, pz1, color=(0, 0, 0),
                                                linewidth=1)
                                else:
                                    ax.plot(px1, py1, pz1, color=(0,0,0),
                                            linewidth=1)

                inX+=list(inXk)
                inY += list(inYk)
                inZ += list(inZk)

        inX = numpy.array(inX)
        inY = numpy.array(inY)
        inZ = numpy.array(inZ)

        inXmax = numpy.nanmax(inX)
        inXmin = numpy.nanmin(inX)

        inYmax = numpy.nanmax(inY)
        inYmin = numpy.nanmin(inY)

        inZmax = numpy.nanmax(inZ)
        inZmin = numpy.nanmin(inZ)

        rangeins = numpy.array([inXmax - inXmin, inYmax - inYmin, inZmax - inZmin])
        rangeinsMax = numpy.max(rangeins)
        rangeinsMaxInd = numpy.argmax(rangeins)
        #print rangeins

        if rangeinsMaxInd == 0:
            ax.set_xlim3d(inXmin, inXmax)
            ax.set_ylim3d(numpy.nanmean(inY) - (inXmax - inXmin) / 2, numpy.nanmean(inY) + (inXmax - inXmin) / 2)
            ax.set_zlim3d(numpy.nanmean(inZ) - (inXmax - inXmin) / 2, numpy.nanmean(inZ) + (inXmax - inXmin) / 2)

        if rangeinsMaxInd == 1:
            ax.set_ylim3d(inYmin, inYmax)
            ax.set_xlim3d(numpy.nanmean(inX) - (inYmax - inYmin) / 2, numpy.nanmean(inX) + (inYmax - inYmin) / 2)
            ax.set_zlim3d(numpy.nanmean(inZ) - (inYmax - inYmin) / 2, numpy.nanmean(inZ) + (inYmax - inYmin) / 2)

        if rangeinsMaxInd == 2:
            ax.set_zlim3d(inZmin, inZmax)
            ax.set_ylim3d(numpy.nanmean(inY) - (inZmax - inZmin) / 2, numpy.nanmean(inY) + (inZmax - inZmin) / 2)
            ax.set_xlim3d(numpy.nanmean(inX) - (inZmax - inZmin) / 2, numpy.nanmean(inX) + (inZmax - inZmin) / 2)


        ax.set_xlabel('X in cm')
        ax.set_ylabel('Y in cm')
        ax.set_zlabel('Z in cm')
        if PlotDesciptiveDic["3DPlotViewpoint"]!="":
            ax.view_init(int(PlotDesciptiveDic["3DPlotViewpoint"].split(",")[0]),int(PlotDesciptiveDic["3DPlotViewpoint"].split(",")[1]))



        #scale = your_mesh.points.flatten(-1)
        #ax.auto_scale_xyz(scale, scale, scale)

        #mngr = plt.get_current_fig_manager()
        #mngr.window.SetPosition((800, 20))
        #plt.show()
        #plt.close()

        if PlotDesciptiveDic["SaveImages"]=="y":
            fig3D.savefig(PlotDesciptiveDic["Path2SaveImages"] + '/' + PlotDesciptiveDic["SaveFileName"] +'_3D' + '.png', bbox_inches='tight',
                        dpi=600)





    def Plot3DSquid(self,fwData):
        cameraNames = ["camera1", "camera2", "combined"]


        inX=[]
        inY=[]
        inZ=[]
        inN=[]
        #pointssquidreal = numpy.array(
         #   [[0, 3.55998, -0.02321], [-0.56313, -0.89247, 0.22152], [0.56313, -0.89247, 0.22152],[0,0,0]])
        pointssquidreal = numpy.array(
            [[0, 3.55998, -0.02321], [0.56313, -0.89247, 0.22152], [-0.56313, -0.89247, 0.22152],[0,0,0]])
        SolarSquidVect = numpy.array([0.2808512, 0.05381697, -0.95824127])
        SolarSquidVectxy = numpy.array([0.2808512, 0.05381697])

        PlaceNum=1
        eqRangeArr=[11,21,19,16]
        eqRange=eqRangeArr[PlaceNum]
        plotInsects3="C:/Users/Parrish/Documents/aOTIC/put on a hard drive/SquidMovie1/"
        if 1:
            try:
                ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])

                ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

                RotMatMinClas=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                TranMatMinClas=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                MinClassData=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatParams'])

            except:
                None
        eq=0


        inNDict={}

        fwDataKeysminArr=[46608, 49008, 51648,65808]
        anglesToSun=[]
        SquidOrder = ["12", "1", "10", "6", "2", "11", "3", "8", "13", "5", "7", "4", "9", "", "16", "14", "15", "18",
                      "19", "20", "17"]
        if 1:
            SquidOrder = [2,
                5,
                7,
                12,
                10,
                4,
                11,
                8,
                13,
                3,
                6,
                1,
                9,
                16,
                17,
                15,
                21,
                18,
                19,
                20]
        if 0:
            SquidOrder = [6,
                7,
                4,
                5,
                3,
                2,
                8,
                10,
                1,
                9,
                16,
                12]

        for eq in range(eqRange):
            inNDict[eq]=[]
            if 0:
                fwDataKeys = fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)].keys()
                fwDataKeysInt = []
                # print len(fwDataKeys), "fwDataKeys"
                # if len(fwDataKeys) < 2:
                #    continue
                for s in fwDataKeys:
                    if s != "FrameDelay":
                        fwDataKeysInt.append(int(s))


                fwDataKeysmin = min(fwDataKeysInt)
                print str(int(fwDataKeysmin))
            if 1:
                fwDataKeysmin=fwDataKeysminArr[PlaceNum]

            try:
                TotRotSQ=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                    str(int(fwDataKeysmin))]['TotRotSQ'][:]
                TotTrans=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                    str(int(fwDataKeysmin))]['TotTrans'][:]
                FitDatawithFiducial=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                    str(int(fwDataKeysmin))]['FitDatawithFiducial'][:]
            except:
                continue


            for i in range(len(pointssquidreal)):

                print  pointssquidreal[i], numpy.matmul(TotRotSQ, pointssquidreal[i])
                inNDict[eq].append(numpy.matmul(TotRotSQ,pointssquidreal[i])*FitDatawithFiducial[3]+ TotTrans)


            #don't know is plue y or minus y
            SquidAxis=numpy.matmul(TotRotSQ,numpy.array([0,1,0]))
            SquidAxisxy=numpy.array([SquidAxis[0],SquidAxis[1]])

            anglesToSun.append((numpy.arctan2(SquidAxisxy[1],SquidAxisxy[0])-numpy.arctan2(SolarSquidVectxy[1],SolarSquidVectxy[0]))*180/3.1415926)

        print "anglesToSun"+str(fwDataKeysmin)+"=",anglesToSun


        if 0:
            MinSquidDistance = []
            for eq in range(eqRange):


                SquidDistance = []
                for eq2 in range(eqRange):

                    if eq2 != eq:
                        clx = [inNDict[eq][3][0], inNDict[eq2][3][0]]
                        cly = [inNDict[eq][3][1], inNDict[eq2][3][1]]
                        clz = [inNDict[eq][3][2], inNDict[eq2][3][2]]
                        clx1 = numpy.array([inNDict[eq][3][0], inNDict[eq][3][1], inNDict[eq][3][2]])
                        clx2 = numpy.array([inNDict[eq2][3][0], inNDict[eq2][3][1], inNDict[eq2][3][2]])
                        # clxDiff=clx2-clx1#c1x1 is the center point
                        clxDiff = clx1 - clx2  # c1x1 is the center point

                        SquidDistance.append(numpy.linalg.norm(clxDiff))
                print SquidDistance
                MinSquidDistance.append(min(SquidDistance))
            print "MinSquidDistance"+str(fwDataKeysmin)+"=", MinSquidDistance

        if 1:
            for eqi in range(len(SquidOrder)):
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                for eq in range(eqRange):

                    #print inN
                    #print inN[:][0], inN[:][1], inN[:][2]
                    #print inN[0][:], inN[1][:], inN[2][:]
                    for ia in range(len(pointssquidreal)):
                        sdk = ax.scatter(inNDict[eq][ia][0], inNDict[eq][ia][1], inNDict[eq][ia][2], c='b', marker=".", s=2, lw=2)
                    #sdk = ax.scatter(inN[0][:], inN[1][:], inN[2][:], c='b', marker=".", s=2, lw=2)

                    xx=[]
                    yy=[]
                    zz=[]
                    for ia in range(3):
                        xx.append(inNDict[eq][ia][0])
                        yy.append(inNDict[eq][ia][1])
                        zz.append(inNDict[eq][ia][2])

                    verts = [list(zip(xx, yy, zz))]
                    # verts = [x, y, z]

                    pc = Poly3DCollection(verts, linewidths=1)
                    pc.set_alpha(.6)  # Order reversed
                    pc.set_facecolor("C0")

                    ax.add_collection3d(pc)

                #print "anglesToSun",anglesToSun

                #for eq in range(eqRange):

            #if 1:


                if 1:
                #if SquidOrder[eqi]!="":
                    for eqii in range(eqi):
                        eq = int(SquidOrder[eqii]) - 1
                        #for eq in range(eqRange):
                        if 1:
                            # print inN
                            # print inN[:][0], inN[:][1], inN[:][2]
                            # print inN[0][:], inN[1][:], inN[2][:]
                            for ia in range(len(pointssquidreal)):
                                sdk = ax.scatter(inNDict[eq][ia][0], inNDict[eq][ia][1], inNDict[eq][ia][2], c='r',
                                                 marker=".", s=2, lw=2)
                            # sdk = ax.scatter(inN[0][:], inN[1][:], inN[2][:], c='b', marker=".", s=2, lw=2)

                            xx = []
                            yy = []
                            zz = []
                            for ia in range(3):
                                xx.append(inNDict[eq][ia][0])
                                yy.append(inNDict[eq][ia][1])
                                zz.append(inNDict[eq][ia][2])

                            verts = [list(zip(xx, yy, zz))]
                            # verts = [x, y, z]

                            pc = Poly3DCollection(verts, linewidths=1)
                            pc.set_alpha(.6)  # Order reversed
                            pc.set_facecolor("R0")

                            ax.add_collection3d(pc)

                    eq=int(SquidOrder[eqi])-1
                    SquidDistance = []
                    for eq2 in range(eqRange):

                        if eq2!=eq:
                            clx = [inNDict[eq][3][0], inNDict[eq2][3][0]]
                            cly = [inNDict[eq][3][1], inNDict[eq2][3][1]]
                            clz = [inNDict[eq][3][2], inNDict[eq2][3][2]]
                            clx1=numpy.array([inNDict[eq][3][0],inNDict[eq][3][1],inNDict[eq][3][2]])
                            clx2=numpy.array([inNDict[eq2][3][0],inNDict[eq2][3][1],inNDict[eq2][3][2]])
                            #clxDiff=clx2-clx1#c1x1 is the center point
                            clxDiff=clx1-clx2#c1x1 is the center point

                            SquidDistance.append(numpy.linalg.norm(clxDiff))
                            theta =  0.5 *numpy.arccos(clxDiff.dot(SolarSquidVect) / (
                                    numpy.linalg.norm(clxDiff) * numpy.linalg.norm(SolarSquidVect)))

                            if 1:
                                # this is for a the fresnel reflection for a dielectric.
                                n1 = 1.33
                                n2 = 1.52

                                # s polarizaiton :::
                                Rs = n1 * numpy.cos(theta) - n2 * numpy.sqrt(
                                    1 - ((n1 / n2) * numpy.sin(theta)) ** 2)
                                Rs = Rs / (n1 * numpy.cos(theta) + n2 * numpy.sqrt(
                                    1 - ((n1 / n2) * numpy.sin(theta)) ** 2))
                                Rs = Rs ** 2
                                # p polarization :::
                                Rp = n1 * numpy.sqrt(1 - ((n1 / n2) * numpy.sin(theta)) ** 2) - n2 * numpy.cos(
                                    theta)
                                Rp = Rp / (n1 * numpy.sqrt(
                                    1 - ((n1 / n2) * numpy.sin(theta)) ** 2) + n2 * numpy.cos(theta))
                                Rp = Rp ** 2


                            Dolp = numpy.abs(Rs - Rp) / (Rs + Rp)
                            if 0:
                                if 0: #eq sees it eq2
                                    TotRotSQ = fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                                   str(int(fwDataKeysmin))]['TotRotSQ'][:]
                                    TotTrans = fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                                   str(int(fwDataKeysmin))]['TotTrans'][:]
                                    FitDatawithFiducial = fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                                              str(int(fwDataKeysmin))]['FitDatawithFiducial'][:]

                                    #numpy.matmul(TotRotSQ, pointssquidreal[i]) * FitDatawithFiducial[3] + TotTrans
                                    clx2Trans=numpy.matmul(numpy.linalg.inv(TotRotSQ) ,(clx2-TotTrans))/FitDatawithFiducial[3]
                                    clx2Trans=clx2Trans/numpy.linalg.norm(clx2Trans)
                                    if clx2Trans.dot(numpy.array([0,-1,0]))>.9:
                                        ColorLook=(0,1,0)
                                    else:
                                        ColorLook = (1, 1, 0)
                                if 1:#eq2 seas it
                                    TotRotSQ = fwData[cameraNames[2]]["Squid" + str(eq2 + 1) + "_" + "Squid" + str(eq2 + 1)][
                                                   str(int(fwDataKeysmin))]['TotRotSQ'][:]
                                    TotTrans = fwData[cameraNames[2]]["Squid" + str(eq2 + 1) + "_" + "Squid" + str(eq2 + 1)][
                                                   str(int(fwDataKeysmin))]['TotTrans'][:]
                                    FitDatawithFiducial = \
                                    fwData[cameraNames[2]]["Squid" + str(eq2 + 1) + "_" + "Squid" + str(eq2 + 1)][
                                        str(int(fwDataKeysmin))]['FitDatawithFiducial'][:]

                                    # numpy.matmul(TotRotSQ, pointssquidreal[i]) * FitDatawithFiducial[3] + TotTrans
                                    clx2Trans = numpy.matmul(numpy.linalg.inv(TotRotSQ), (clx1 - TotTrans)) / \
                                                FitDatawithFiducial[3]
                                    clx2Trans = clx2Trans / numpy.linalg.norm(clx2Trans)
                                    if clx2Trans.dot(numpy.array([0, 1, 0])) > .9:
                                        ColorLook = (0, 1, 0)
                                    else:
                                        ColorLook = (1, 1, 0)
                                #print thetag*180/3.1415926
                                #ax.plot(clx, cly, clz, color=plt.cm.jet(Dolp), linewidth=2)
                                ax.plot(clx, cly, clz, color=ColorLook, linewidth=1)
                            else:
                                ax.plot(clx, cly, clz, color=plt.cm.jet(Dolp), linewidth=2)



                if eqi==0:
                    for eq in range(eqRange):
                        for i in range(len(pointssquidreal)):
                            inX.append(inNDict[eq][i][0])
                            inY.append(inNDict[eq][i][1])
                            inZ.append(inNDict[eq][i][2])


                    print inX
                    print inY
                    print inZ
                    inX = numpy.array(inX)
                    inY = numpy.array(inY)
                    inZ = numpy.array(inZ)

                    inXmax = numpy.max(inX)
                    inXmin = numpy.min(inX)

                    inYmax = numpy.max(inY)
                    inYmin = numpy.min(inY)

                    inZmax = numpy.max(inZ)
                    inZmin = numpy.min(inZ)
                if 1:
                    rangeins = numpy.array([inXmax - inXmin, inYmax - inYmin, inZmax - inZmin])
                    rangeinsMax = numpy.max(rangeins)
                    rangeinsMaxInd = numpy.argmax(rangeins)
                    print rangeins
                    scale=-70
                    if rangeinsMaxInd == 0:
                        ax.set_xlim3d(inXmin-scale, inXmax+scale)
                        ax.set_ylim3d(numpy.mean(inY) - (inXmax - inXmin) / 2-scale, numpy.mean(inY) + (inXmax - inXmin) / 2+scale)
                        ax.set_zlim3d(numpy.mean(inZ) - (inXmax - inXmin) / 2-scale, numpy.mean(inZ) + (inXmax - inXmin) / 2+scale)
                        centerPoint=[(inXmin+inXmax)/2,(numpy.mean(inY) - (inXmax - inXmin) / 2 +(numpy.mean(inY) + (inXmax - inXmin) / 2))/2,
                                     (numpy.mean(inZ) - (inXmax - inXmin) / 2+(numpy.mean(inZ) + (inXmax - inXmin) / 2))/2]

                    if rangeinsMaxInd == 1:
                        ax.set_ylim3d(inYmin-scale, inYmax+scale)
                        ax.set_xlim3d(numpy.mean(inX) - (inYmax - inYmin) / 2-scale, numpy.mean(inX) + (inYmax - inYmin) / 2+scale)
                        ax.set_zlim3d(numpy.mean(inZ) - (inYmax - inYmin) / 2-scale, numpy.mean(inZ) + (inYmax - inYmin) / 2+scale)
                        centerPoint=[(numpy.mean(inX) - (inYmax - inYmin) / 2+ (numpy.mean(inX) + (inYmax - inYmin) / 2))/2,(inYmin+inYmax)/2,
                                     (numpy.mean(inZ) - (inYmax - inYmin) / 2+ numpy.mean(inZ) + (inYmax - inYmin) / 2)/2]


                    if rangeinsMaxInd == 2:
                        ax.set_zlim3d(inZmin-scale, inZmax+scale)
                        ax.set_ylim3d(numpy.mean(inY) - (inZmax - inZmin) / 2-scale, numpy.mean(inY) + (inZmax - inZmin) / 2+scale)
                        ax.set_xlim3d(numpy.mean(inX) - (inZmax - inZmin) / 2-scale, numpy.mean(inX) + (inZmax - inZmin) / 2+scale)
                        centerPoint=[(numpy.mean(inX) - (inZmax - inZmin) / 2+ numpy.mean(inX) + (inZmax - inZmin) / 2)/2,
                                     (numpy.mean(inY) - (inZmax - inZmin) / 2+ numpy.mean(inY) + (inZmax - inZmin) / 2)/2,
                                     (inZmin+ inZmax)/2]

                length2=20
                if 0:
                    ax.quiver(centerPoint[0], centerPoint[1], centerPoint[2]+40, length2 * SolarSquidVect[0], length2 * SolarSquidVect[1],
                              length2 * SolarSquidVect[2], color='y', alpha=.8, lw=4, arrow_length_ratio=.2)
                    ax.quiver(centerPoint[0], centerPoint[1], centerPoint[2]+40, length2 * SolarSquidVect[0], length2 * SolarSquidVect[1],
                              0, color='y', alpha=.8, lw=4, arrow_length_ratio=.2)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

                ax.set_axis_off()
                if PlaceNum==3:
                    ax.view_init(130,69)
                if PlaceNum==1:
                    ax.view_init(115,104)
                plt.tight_layout()
                plt.savefig(plotInsects3+"/fig" + str(eqi).zfill(4) + ".jpg", dpi=300)
                #plt.show()
                plt.close()





    ########################################################################
    ################################################################################################################
    # linestring is the camera info, and pixelString is the pixel info

    # THis is returning the vector that originates from the camera origin and goes in the direction of the pixel.

    def theVectors(self, dat, pixelarray, cameraMatrix, distCoeffs):

        CamOrgn = [dat[9], dat[10], dat[11]]

        # so the new goalvect will probably just be this:
        test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
        test[0][0][0] = pixelarray[0]
        test[0][0][1] = pixelarray[1]

        undistort = cv2.undistortPoints(test, cameraMatrix, distCoeffs)
        UdistortVect = numpy.zeros(2)
        UdistortVect[0] = undistort[0][0][0]
        UdistortVect[1] = undistort[0][0][1]

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





        ##################################################################################################################

    def returnMutualPointProjection(self, dat1, dat2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                    width, height, point1, point2):

        zvect1 = numpy.zeros(3)
        zvect2 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        xvect2 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        yvect2 = numpy.zeros(3)

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        boundary = .01 * width

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        rotVect1 = numpy.zeros(3)
        rotVect2 = numpy.zeros(3)


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

        xvect2[0] = dat2[0]
        xvect2[1] = dat2[1]
        xvect2[2] = dat2[2]

        yvect2[0] = dat2[3]
        yvect2[1] = dat2[4]
        yvect2[2] = dat2[5]

        zvect2[0] = dat2[6]
        zvect2[1] = dat2[7]
        zvect2[2] = dat2[8]

        CamOrgn1[0] = dat1[9]
        CamOrgn1[1] = dat1[10]
        CamOrgn1[2] = dat1[11]

        CamOrgn2[0] = dat2[9]
        CamOrgn2[1] = dat2[10]
        CamOrgn2[2] = dat2[11]

        CamOrgn2 = numpy.array(CamOrgn2)
        CamOrgn1 = numpy.array(CamOrgn1)

        zvect1 = numpy.array(zvect1)
        zvect2 = numpy.array(zvect2)

        xvect1 = numpy.array(xvect1)
        xvect2 = numpy.array(xvect2)

        yvect1 = numpy.array(yvect1)
        yvect2 = numpy.array(yvect2)


        line1to2 = [[0, 0]]
        line2to1 = [[0, 0]]
        k1 = 0
        k2 = 0

        CamOrgn1, pointvect1 = self.theVectors(dat1, point1, cameraMatrix1, distCoeffs1)
        CamOrgn2, pointvect2 = self.theVectors(dat2, point2, cameraMatrix2, distCoeffs2)


        test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
        test[0][0][0] = boundary
        test[0][0][1] = boundary

        undistort1 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        test[0][0][0] = width - boundary
        test[0][0][1] = height - boundary

        undistort2 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        bound1 = [undistort1[0][0][0], undistort1[0][0][1], undistort2[0][0][0], undistort2[0][0][1]]

        test[0][0][0] = boundary
        test[0][0][1] = boundary

        undistort1 = cv2.undistortPoints(test, cameraMatrix2, distCoeffs2)

        test[0][0][0] = width - boundary
        test[0][0][1] = height - boundary

        undistort2 = cv2.undistortPoints(test, cameraMatrix2, distCoeffs2)

        bound2 = [undistort1[0][0][0], undistort1[0][0][1], undistort2[0][0][0], undistort2[0][0][1]]

        cloPo1_1, cloPo2_1 = self.findClosestPointsOfnonParallelLines(CamOrgn2, pointvect2, CamOrgn1,
                                                                      zvect1)  # does not work

        cloPo1_2, cloPo2_2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, CamOrgn2, zvect2)

        maxrange = 50
        for i in range(maxrange):
            x = .1 * (i - maxrange / 2)
            #x = 10 * (i - maxrange / 2)
            #x = 5 * (i - maxrange / 2)

            line1point1 = cloPo1_2 + x * pointvect1  # works

            line1point2 = cloPo1_1 + x * pointvect2  # work
            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            cop = CamOrgn2 - line1point1  # works

            d3test[0][0][0] = xvect2.dot(cop) / zvect2.dot(cop)
            d3test[0][0][1] = yvect2.dot(cop) / zvect2.dot(cop)
            d3test[0][0][2] = 1

            if d3test[0][0][0] >= bound2[0] and d3test[0][0][0] <= bound2[2] and d3test[0][0][1] >= bound2[1] and \
                            d3test[0][0][1] <= bound2[3]:

                imagePoints, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix2, distCoeffs2)

                if k1 == 0:
                    line1to2[0][0] = int(imagePoints[0][0][0])
                    line1to2[0][1] = int(imagePoints[0][0][1])
                    k1 += 1
                else:
                    line1to2.append([int(imagePoints[0][0][0]), int(imagePoints[0][0][1])])

            cop2 = CamOrgn1 - line1point2  # work

            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
            d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
            d3test[0][0][2] = 1

            if d3test[0][0][0] >= bound1[0] and d3test[0][0][0] <= bound1[2] and d3test[0][0][1] >= bound1[1] and \
                            d3test[0][0][1] <= bound1[3]:

                imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

                if k2 == 0:
                    line2to1[0][0] = int(imagePoints2[0][0][0])
                    line2to1[0][1] = int(imagePoints2[0][0][1])
                    k2 += 1
                else:
                    line2to1.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])

        return line1to2, line2to1

    ########################################################################################






        ##################################################################################################################
    #191113

    def returnMutualPointProjection2(self, dat1, dat2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                    width, height, point1, point2):

        #print cameraMatrix1,distCoeffs1
        zvect1 = numpy.zeros(3)
        zvect2 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        xvect2 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        yvect2 = numpy.zeros(3)

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        boundary = .01 * width

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        rotVect1 = numpy.zeros(3)
        rotVect2 = numpy.zeros(3)


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

        xvect2[0] = dat2[0]
        xvect2[1] = dat2[1]
        xvect2[2] = dat2[2]

        yvect2[0] = dat2[3]
        yvect2[1] = dat2[4]
        yvect2[2] = dat2[5]

        zvect2[0] = dat2[6]
        zvect2[1] = dat2[7]
        zvect2[2] = dat2[8]

        CamOrgn1[0] = dat1[9]
        CamOrgn1[1] = dat1[10]
        CamOrgn1[2] = dat1[11]

        CamOrgn2[0] = dat2[9]
        CamOrgn2[1] = dat2[10]
        CamOrgn2[2] = dat2[11]

        CamOrgn2 = numpy.array(CamOrgn2)
        CamOrgn1 = numpy.array(CamOrgn1)

        zvect1 = numpy.array(zvect1)
        zvect2 = numpy.array(zvect2)

        xvect1 = numpy.array(xvect1)
        xvect2 = numpy.array(xvect2)

        yvect1 = numpy.array(yvect1)
        yvect2 = numpy.array(yvect2)


        line1to2 = [[0, 0]]
        line2to1 = [[0, 0]]
        k1 = 0
        k2 = 0

        CamOrgn1, pointvect1 = self.theVectors(dat1, point1, cameraMatrix1, distCoeffs1)
        CamOrgn2, pointvect2 = self.theVectors(dat2, point2, cameraMatrix2, distCoeffs2)


        test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
        test[0][0][0] = boundary
        test[0][0][1] = boundary

        undistort1 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        test[0][0][0] = width - boundary
        test[0][0][1] = height - boundary

        undistort2 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        bound1 = [undistort1[0][0][0], undistort1[0][0][1], undistort2[0][0][0], undistort2[0][0][1]]

        test[0][0][0] = boundary
        test[0][0][1] = boundary

        undistort1 = cv2.undistortPoints(test, cameraMatrix2, distCoeffs2)

        test[0][0][0] = width - boundary
        test[0][0][1] = height - boundary

        undistort2 = cv2.undistortPoints(test, cameraMatrix2, distCoeffs2)

        bound2 = [undistort1[0][0][0], undistort1[0][0][1], undistort2[0][0][0], undistort2[0][0][1]]


        #find closest point on the pixel ray to the other camera zvector

        cloPo1_1, cloPo2_1 = self.findClosestPointsOfnonParallelLines(CamOrgn2, pointvect2, CamOrgn1,
                                                                      zvect1)  # does not work

        cloPo1_2, cloPo2_2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, CamOrgn2, zvect2)

        def linout1(xx):
            line1point1 = cloPo1_2 + xx * pointvect1  # works

            ################################################
            cop = CamOrgn2 - line1point1  # works
            return line1point1, cop
        ###########################################################
        maxrange = 50
        for i in range(maxrange):
            #x = .1 * (i - maxrange / 2)
            #x = 10 * (i - maxrange / 2)
            x = 50 * (i - maxrange / 2)

            line1point1, cop=linout1(x)

            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            d3test[0][0][0] = xvect2.dot(cop) / zvect2.dot(cop)
            d3test[0][0][1] = yvect2.dot(cop) / zvect2.dot(cop)
            d3test[0][0][2] = 1

            # seeing if it is within bounds.
            if d3test[0][0][0] >= bound2[0] and d3test[0][0][0] <= bound2[2] and d3test[0][0][1] >= bound2[1] and \
                            d3test[0][0][1] <= bound2[3]:

                imagePoints, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix2, distCoeffs2)

                if k1 == 0:
                    line1to2[0][0] = int(imagePoints[0][0][0])
                    line1to2[0][1] = int(imagePoints[0][0][1])
                    k1 += 1
                else:
                    line1to2.append([int(imagePoints[0][0][0]), int(imagePoints[0][0][1])])




            ################################################
            line1point2 = cloPo1_1 + x * pointvect2  # work
            cop2 = CamOrgn1 - line1point2  # work

            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
            d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
            d3test[0][0][2] = 1

            if d3test[0][0][0] >= bound1[0] and d3test[0][0][0] <= bound1[2] and d3test[0][0][1] >= bound1[1] and \
                            d3test[0][0][1] <= bound1[3]:

                imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

                if k2 == 0:
                    line2to1[0][0] = int(imagePoints2[0][0][0])
                    line2to1[0][1] = int(imagePoints2[0][0][1])
                    k2 += 1
                else:
                    line2to1.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])

        return line1to2, line2to1

    ########################################################################################




    def findClosestPointsOfnonParallelLines(self, CamOrgn1, pointvect1, CamOrgn2, pointvect2):

        cloPo1 = numpy.zeros(3)
        cloPo2 = numpy.zeros(3)
        n1 = numpy.cross(pointvect1, numpy.cross(pointvect2, pointvect1))#changed 11/24/18
        n2 = numpy.cross(pointvect2, numpy.cross(pointvect1, pointvect2))
        c2scale = (numpy.dot((CamOrgn1 - CamOrgn2), n1) / numpy.dot(pointvect2, n1))
        c1scale = (numpy.dot((CamOrgn2 - CamOrgn1), n2) / numpy.dot(pointvect1, n2))
        cloPo1 = CamOrgn1 + c1scale * pointvect1

        cloPo2 = CamOrgn2 + c2scale * pointvect2
        return cloPo1, cloPo2











        ##########################################################################################
        ####  here we are bringing together the two camera2 to combine the tracks.

    def MergeAndComputeInsectTracks(self, fwData, frameDelay, f1, cameraMatrix1, distCoeffs1, f2, cameraMatrix2,
                                    distCoeffs2, Tobject,Box,Water_Surface_Plane,h5SquidDelay,delayfraction):
        #initialization parameters
        print "MergeAndComputeInsectTracks"
        cameraNames = ["camera1", "camera2", "combined"]


        point1 = numpy.zeros(2)
        point2 = numpy.zeros(2)
        pp11=numpy.zeros(2)
        pp12 = numpy.zeros(2)
        #delayfraction=.7  #Was before191105


        #delay fraction information
        #delayfraction = 0.125 #200329
        if delayfraction==0:
            useDelatyFraction=False
        else:
            useDelatyFraction = True
        dat1 = numpy.zeros(12)
        dat2 = numpy.zeros(12)
        ThereIsAReflection = [0,0]
        #loading the mirror dictionaries.
        mirror={}
        try:
            mirror[0] = fwData[cameraNames[0]]["mirrorCamera1"].value
            ThereIsAReflection[0]=1
            print "mirror 1", mirror[0]
        except:
            None

        try:
            mirror[1] = fwData[cameraNames[1]]["mirrorCamera2"].value
            ThereIsAReflection[1]=1
            print "mirror 2", mirror[1]
        except:
            None

        if ThereIsAReflection[0]==1 or ThereIsAReflection[1]==1:
            print Water_Surface_Plane,"Water_Surface_Plane"
            abcd = []
            for ic in range(4):
                abcd.append(float(Water_Surface_Plane.split(";")[0].split(",")[ic]))

        for insect in fwData[cameraNames[0]]:  # calling the first camera tracks insects

            if Tobject in insect:  # unising only the right kind of object as defined by Tobject


                if 1:
                    TheBug = 0
                    BugDifferenceArrayName = [insect]
                    useCombination = True

                ### do the iteration again for the chosen one and call it insect-num-num
                print insect + "_" + BugDifferenceArrayName[TheBug]

                if useCombination == True:

                    # here we are creating the H5 group to put the data in or deleting it to create a knew one
                    try:
                        fwData[cameraNames[2]].create_group(insect + "_" + BugDifferenceArrayName[TheBug])
                    except:
                        del fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]]
                        fwData[cameraNames[2]].create_group(insect + "_" + BugDifferenceArrayName[TheBug])


                    # we are setting up the araingement to select the correct measurements to compare... adding a frame delay

                    # Here are the keys of the frame numbers for the various measurements
                    try:
                        insectkey = fwData[cameraNames[0]][insect].keys()
                        bugkey = fwData[cameraNames[1]][BugDifferenceArrayName[TheBug]].keys()
                    except:
                        continue

                    ########################################3
                    ###          Finding coincident points

                    # turning the insect key from u to string
                    insectkey = map(str, insectkey)

                    # then tunring it into a set
                    insectkey = set(insectkey)

                    # making bugkeyF into an array of integers to get it into a map.
                    bugkeyF = numpy.array(map(int, bugkey))


                    # so it is at a different delay and we want to add the frame delay to get it to where it matches the insect
                    # as apposed to subtracting it which is done above
                    bugkeyF = bugkeyF + frameDelay

                    # turning it into s string set
                    bugkeyF = map(str, bugkeyF)
                    bugkeyF = set(bugkeyF)
                    bugkey = set(bugkey)



                    # I am doing this insted
                    bugkey = bugkeyF

                    inter = bugkey.intersection(insectkey)

                    inter = list(inter)

                    DifferenceNorm = []


                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]].create_dataset(
                        'FrameDelay', data=[frameDelay])
                    for i in inter:
                        try:
                            if useDelatyFraction==True:
                                pp11[0] = fwData[cameraNames[0]][insect][str(int(i))][
                                    0]
                                pp11[1] = fwData[cameraNames[0]][insect][str(int(i))][
                                    1]
                                pp12[0] = fwData[cameraNames[0]][insect][str(int(i) + 1)][
                                    0]
                                pp12[1] = fwData[cameraNames[0]][insect][str(int(i) + 1)][
                                    1]
                                point1 = pp11 + delayfraction * (pp12 - pp11)

                            else:

                                point1[0] = fwData[cameraNames[0]][insect][str(int(i))][0]
                                point1[1] = fwData[cameraNames[0]][insect][str(int(i))][1]


                            point2[0] = \
                            fwData[cameraNames[1]][BugDifferenceArrayName[TheBug]][str(int(i) - frameDelay)][0]
                            point2[1] = \
                            fwData[cameraNames[1]][BugDifferenceArrayName[TheBug]][str(int(i) - frameDelay)][1]

                            #dat1 = numpy.array(f1['F' + str(int(i))]['CameraPos'][:])

                            #dat2 = numpy.array(f2['F' + str(int(i)- frameDelay)]['CameraPos'][:])

                            dat1 = numpy.array(f1['F' + str(int(i)-h5SquidDelay)]['CameraPos'][:])

                            dat2 = numpy.array(f2['F' + str(int(i)-h5SquidDelay - frameDelay)]['CameraPos'][:])

                            CamOrgn1, pointvect1 = self.theVectors(dat1, point1, cameraMatrix1, distCoeffs1)
                            CamOrgn2, pointvect2 = self.theVectors(dat2, point2, cameraMatrix2, distCoeffs2)

                            cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, CamOrgn2,
                                                                                      pointvect2)

                            cloPoAve = [(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                        (cloPo1[2] + cloPo2[2]) / 2]

                            cloPoError = numpy.linalg.norm(cloPo1 - cloPo2) / 2
                            Error1 = numpy.array([fwData[cameraNames[0]][insect][str(int(i))][2],
                                                  fwData[cameraNames[0]][insect][str(int(i))][3]])
                            Error2 = numpy.array(
                                [fwData[cameraNames[1]][BugDifferenceArrayName[TheBug]][str(int(i) - frameDelay)][2],
                                 fwData[cameraNames[1]][BugDifferenceArrayName[TheBug]][str(int(i) - frameDelay)][3]])


                            #######################################################################
                            # entering into the H5
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]].create_group(
                                str(int(i)))
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset('point1', data=point1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset('point2', data=point2)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset('3Dpoint', data=cloPoAve)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset('3DpointError', data=[cloPoError])
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset('cloPo1', data=cloPo1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'cloPo2', data=cloPo2)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'point1Error', data=Error1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'point2Error', data=Error2)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'CameraOriginError1', data=numpy.array([0.001, 0.001, 0.001]))
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'CameraOriginError2', data=numpy.array([0.001, 0.001, 0.001]))
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'CameraAngularError1', data=numpy.array([0.001, 0.001, 0.001]))
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'CameraAngularError2', data=numpy.array([0.001, 0.001, 0.001]))
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'pointvect1', data=pointvect1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'pointvect2', data=pointvect2)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'cameraData1', data=dat1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'cameraData2', data=dat2)

                            if Error1[0] < 0.0000001:
                                Error1[0] = 40
                                Error1[1] = 40
                            if Error2[0] < 0.0000001:
                                Error2[0] = 40
                                Error2[1] = 40

                            Epoint1x = numpy.array([Error1[0] + point1[0], point1[1]])
                            Epoint1y = numpy.array([point1[0], Error1[1] + point1[1]])

                            CamOrgn1, Epointvect1x = self.theVectors(dat1, Epoint1x, cameraMatrix1, distCoeffs1)
                            CamOrgn1, Epointvect1y = self.theVectors(dat1, Epoint1y, cameraMatrix1, distCoeffs1)


                            PixelErrorAngle1 = numpy.array([numpy.arccos(numpy.dot(Epointvect1x, pointvect1) / (
                                numpy.linalg.norm(Epointvect1x) * numpy.linalg.norm(pointvect1))), numpy.arccos(
                                numpy.dot(Epointvect1y, pointvect1) / (
                                    numpy.linalg.norm(Epointvect1y) * numpy.linalg.norm(pointvect1)))])

                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'PixelErrorAngle1', data=PixelErrorAngle1)

                            Epoint2x = numpy.array([Error2[0] + point2[0], point2[1]])

                            Epoint2y = numpy.array([point2[0], Error2[0] + point2[1]])

                            CamOrgn1, Epointvect2x = self.theVectors(dat2, Epoint2x, cameraMatrix2, distCoeffs2)
                            CamOrgn1, Epointvect2y = self.theVectors(dat2, Epoint2y, cameraMatrix2, distCoeffs2)
                            PixelErrorAngle2 = numpy.array([numpy.arccos(numpy.dot(Epointvect2x, pointvect2) / (
                                numpy.linalg.norm(Epointvect2x) * numpy.linalg.norm(pointvect2))), numpy.arccos(
                                numpy.dot(Epointvect2y, pointvect2) / (
                                    numpy.linalg.norm(Epointvect2y) * numpy.linalg.norm(pointvect2)))])

                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'PixelErrorAngle2', data=PixelErrorAngle2)

                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'Error1', data=Error1)
                            fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                str(int(i))].create_dataset(
                                'Error2', data=Error2)

                        except:
                            print "missing frame", i

                ############
                #  Reflection
                for Ri in range(2):
                    if ThereIsAReflection[Ri]==1:
                        mirrorIndicator=0
                        for mr in range(len(mirror[Ri])):
                            if Tobject+str(mirror[Ri][mr][1])== insect:
                                print "yay reflection"
                                insectR = "insect" + str(mirror[Ri][mr][0])
                                mirrorIndicator+=1
                                break

                        if mirrorIndicator==0:
                            continue

                        point1 = numpy.zeros(2)
                        point1r = numpy.zeros(2)



                        insectRkeys = fwData[cameraNames[Ri]][insectR].keys()


                        if 1:
                            #Checking to see if there is a reflection match and skiping the calculation if not.
                            for i in insectRkeys:
                                try:
                                    point1[0] = fwData[cameraNames[Ri]][insect][str(int(i))][0]
                                    point1[1] = fwData[cameraNames[Ri]][insect][str(int(i))][1]

                                    point1r[0] = fwData[cameraNames[Ri]][insectR][str(int(i))][0]
                                    point1r[1] = fwData[cameraNames[Ri]][insectR][str(int(i))][1]
                                except:
                                    continue

                                #checking to see if there is already a 3D point from triangulating and skipping if so
                                try:
                                    fwData[cameraNames[2]][insect + "_" + insect][str(int(i))]['3Dpoint']
                                    continue
                                except:
                                    None


                                #the calculation
                                if Ri==0:
                                    dat1 = numpy.array(f1['F' + str(int(i)-h5SquidDelay)]['CameraPos'][:])

                                    CamOrgn1, pointvect1 = self.theVectors(dat1, point1, cameraMatrix1,
                                                                           distCoeffs1)
                                    CamOrgn1, pointvect2 = self.theVectors(dat1, point1r, cameraMatrix1,
                                                                           distCoeffs1)
                                elif Ri==1:

                                    dat1 = numpy.array(f2['F' + str(int(i)-h5SquidDelay - frameDelay)]['CameraPos'][:])
                                    CamOrgn1, pointvect1 = self.theVectors(dat2, point1, cameraMatrix2,
                                                                           distCoeffs1)
                                    CamOrgn1, pointvect2 = self.theVectors(dat2, point1r, cameraMatrix2,
                                                                           distCoeffs2)

                                # abcd = [0.166865, -3.466464, -1, 1.416294]
                                #print abcd,"abcd"
                                abssin = 1
                                COdist = abssin * (
                                    CamOrgn1[0] * abcd[0] + CamOrgn1[1] * abcd[1] + CamOrgn1[2] * abcd[2] +
                                        abcd[3]) / numpy.sqrt(abcd[0] ** 2 + abcd[1] ** 2 + abcd[2] ** 2)

                                ReflectZeroHypot = abssin * -1 * (
                                    CamOrgn1[0] * abcd[0] + CamOrgn1[1] * abcd[1] + CamOrgn1[2] * abcd[2] + abcd[
                                        3]) / (pointvect2[0] * abcd[0] + pointvect2[1] * abcd[1] + pointvect2[2] *
                                               abcd[2])

                                RealZeroHypot = abssin * -1 * (
                                    CamOrgn1[0] * abcd[0] + CamOrgn1[1] * abcd[1] + CamOrgn1[2] * abcd[2] + abcd[
                                        3]) / (pointvect1[0] * abcd[0] + pointvect1[1] * abcd[1] + pointvect1[2] *
                                               abcd[2])

                                ThetaReal = numpy.arcsin(COdist / RealZeroHypot)
                                ThetaReflect = numpy.arcsin(COdist / ReflectZeroHypot)

                                ReflectZeroX = ReflectZeroHypot * numpy.cos(ThetaReflect)
                                RealZeroX = RealZeroHypot * numpy.cos(ThetaReal)

                                D = (RealZeroX - ReflectZeroX) / (
                                    1 / numpy.tan(ThetaReal) + 1 / numpy.tan(ThetaReflect))

                                RealT = (D * (numpy.sqrt(abcd[0] ** 2 + abcd[1] ** 2 + abcd[2] ** 2)) - abssin * (
                                    CamOrgn1[0] * abcd[0] + CamOrgn1[1]
                                    * abcd[1] + CamOrgn1[2] * abcd[2] + abcd[3])) / (
                                            pointvect1[0] * abcd[0] + pointvect1[1] * abcd[1] + pointvect1[2] * abcd[2])


                                #this is the reflected 3D point.
                                RealPoint = CamOrgn1 + RealT * pointvect1

                                if 1:
                                    #######################################################################
                                    # entering into the H5
                                    fwData[cameraNames[2]][insect + "_" + insect].create_group(
                                        str(int(i)))
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset('point1', data=point1)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset('point2', data=point1r)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset('3Dpoint', data=RealPoint)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset('IsMirror', data="yes")

                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset(
                                        'pointvect1', data=pointvect1)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset(
                                        'pointvect2', data=pointvect2)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset(
                                        'cameraData1', data=dat1)
                                    fwData[cameraNames[2]][insect + "_" + insect][
                                        str(int(i))].create_dataset(
                                        'cameraData2', data=dat2)

                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'point1Error', data=0.001)
                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'point2Error', data=0.001)
                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'CameraOriginError1', data=numpy.array([0.001, 0.001, 0.001]))
                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'CameraOriginError2', data=numpy.array([0.001, 0.001, 0.001]))
                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'CameraAngularError1', data=numpy.array([0.001, 0.001, 0.001]))
                                    fwData[cameraNames[2]][insect + "_" + BugDifferenceArrayName[TheBug]][
                                        str(int(i))].create_dataset(
                                        'CameraAngularError2', data=numpy.array([0.001, 0.001, 0.001]))





                                # now what to record.

        #####Here we make the transformation
        if Tobject == "fiducial0_":


            BoxArray = []

            fidArray2 = []

            for combs in fwData[cameraNames[2]]:
                if combs.split("_")[0] == "fiducial0":
                    for difpoints in fwData[cameraNames[2]][combs]:
                        if difpoints != "FrameDelay" and difpoints != "Fiducial_Rotation" and difpoints != "Fiducial_Translation":

                            fidpoint2 = fwData[cameraNames[2]][combs][difpoints]['3Dpoint']

                            fidArray2.append([fidpoint2[0], fidpoint2[1], fidpoint2[2], 1])
                            BoxArray.append(Box[int(combs.split("_")[1]) - 1])
                            print difpoints, Box[int(combs.split("_")[1]) - 1],int(combs.split("_")[1]),fidpoint2[:]

            fidArray2 = numpy.array(fidArray2)
            BoxArray = numpy.array(BoxArray)

            if 1:#aqa
                #This is for a minimization algorithim to find the best rotation*magnification +translation pair for the transform.
                def Rotationthing(SquidIn):
                    theta1 = SquidIn[0]
                    theta2 = SquidIn[1]
                    theta3 = SquidIn[2]
                    magnif = SquidIn[3]
                    # print magnif
                    TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                         [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                    roty = numpy.array(
                        [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                         [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                    rotz = numpy.array(
                        [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                         [0, 0, 1]])
                    # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                    RotMatSQ = numpy.matmul(rotz, numpy.matmul(roty, rotx))
                    PVV = 0
                    for sq in range(len(fidArray2)):
                        # print numpy.isnan(pointssquid[sq][0]),pointssquid[sq][0]
                        Xtotry = numpy.array([fidArray2[sq][0], fidArray2[sq][1], fidArray2[sq][2]])
                        if 1:
                            PV1 = numpy.linalg.norm(
                                BoxArray[sq] - (TranMatSQ + magnif * numpy.matmul(RotMatSQ, Xtotry)))
                            # print PV1,"why not here"
                            PVV += PV1 ** 2
                    PVV = numpy.sqrt(PVV)
                    print PVV,SquidIn
                    return PVV


                SquidIn = [0, 0, 0, 1, 0, 0, 0]

                bds = ((-7, 7), (-7, 7), (-7, 7), (0.0005, 10000), (-1000, 1000), (-1000, 1000), (-1000, 1000))
                # SquidIn = res.x  # [0, 0, 0, 0, 0, 0, 0]
                res = minimize(Rotationthing, SquidIn, bounds=bds)#, constraints=cons)
                SquidIn = res.x
                theta1 = SquidIn[0]
                theta2 = SquidIn[1]
                theta3 = SquidIn[2]
                magnif = SquidIn[3]
                TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                rotx = numpy.array(
                    [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                     [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                roty = numpy.array(
                    [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                     [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                rotz = numpy.array(
                    [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                     [0, 0, 1]])
                # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                RotMatSQ = magnif * numpy.matmul(rotz, numpy.matmul(roty, rotx))















            Result = numpy.linalg.lstsq(fidArray2, BoxArray)[0]
            print numpy.linalg.lstsq(fidArray2, BoxArray)
            ResultRot = numpy.zeros((3, 3))
            ResultTran = numpy.zeros(3)

            ResultRot[0][0] = Result[0][0]
            ResultRot[1][0] = Result[1][0]
            ResultRot[2][0] = Result[2][0]

            ResultRot[0][1] = Result[0][1]
            ResultRot[1][1] = Result[1][1]
            ResultRot[2][1] = Result[2][1]

            ResultRot[0][2] = Result[0][2]
            ResultRot[1][2] = Result[1][2]
            ResultRot[2][2] = Result[2][2]

            ResultTran[0] = Result[3][0]
            ResultTran[1] = Result[3][1]
            ResultTran[2] = Result[3][2]

            print BoxArray
            for iji in range(len(BoxArray)):
                Xtotry = numpy.array([fidArray2[iji][0], fidArray2[iji][1], fidArray2[iji][2]])
                print numpy.matmul(ResultRot.T, Xtotry) + ResultTran,BoxArray[iji]
                print numpy.matmul(RotMatSQ, Xtotry) + TranMatSQ, BoxArray[iji], "SQ"
            try:
                fwData[cameraNames[2]].create_group("fiducial0_T")
            except:
                del fwData[cameraNames[2]]["fiducial0_T"]
                fwData[cameraNames[2]].create_group("fiducial0_T")

            fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                'Fiducial_Rotation', data=ResultRot)

            fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                'Fiducial_Translation', data=ResultTran)

            fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                'Fiducial_fromMinClasicRotMat', data=RotMatSQ)

            fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                'Fiducial_fromMinClasicRotMatTrans', data=TranMatSQ)

            fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                'Fiducial_fromMinClasicRotMatParams', data=SquidIn)


#######################
    def myconvolve3(self,signal, masknum): # convolving algorithim that tapers averages toward edges and gaps.
        aveSignal = []
        #########
        checkingVariable = 0
        ########
        nextTarget = 1
        ##########
        FirstendTarget = 0
        ##########
        BeginingTarget = 1

        ########
        gapWidth=15
        #################
        ### full loop through the signal.
        for i in range(len(signal)):#starts with i=0

            #beginning
            if i == 0:### start with two... what if they are both Nans?
                minilist = numpy.array([signal[i], signal[i + 1]])
                if str(signal[i]) == "nan":
                    aveSignal.append(numpy.nan)
                else:
                    aveSignal.append(numpy.nanmean(minilist))
            elif i == len(signal) - 1:#### look at the end also with just two
                minilist = numpy.array([signal[i], signal[i - 1]])

                if str(signal[i]) == "nan":
                    aveSignal.append(numpy.nan)
                else:
                    aveSignal.append(numpy.nanmean(minilist))

            else:
                if nextTarget == i:
                    k = 0
                    nanflag = 0
                    while k < len(signal) - 1 - i:# what is the i doing here?
                        # len(signal) - 1 - i is what is left of the signal

                        if k == 0:
                            #marking the start with BeginingTarget
                            BeginingTarget = i

                        if str(signal[k + i]) == "nan":
                            #we are looking for a gap
                            nanflag = 1

                            # we have chosen a FirstendTarget, but are we at a gap or just a single nan
                            FirstendTarget = k + i - 1

                            for g in range(len(signal) - (k + i)):
                                #looking at the rest of the signal
                                if str(signal[k + i + g]) != "nan":
                                    checkingVariable = 1

                                    if g > masknum:
                                        #if we have found a real number then we mark were that number is to start the process over again.
                                        nextTarget = k + i + g
                                        k = len(signal) + 1000000 + i
                                        break
                                    else:
                                        # if it is not bigger than the mask number then we keep going
                                        k = k + g - 1
                                        break
                        if k + i == len(signal) - 2:  # and nanflag==0:
                            #have we reached the end?
                            nextTarget = len(signal) + 1
                            FirstendTarget = len(signal) - 1

                        #incramenting k
                        k += 1

                if FirstendTarget - BeginingTarget > 2 * masknum + 1:
                    #looking to see if the path is big enough to make an average
                    masknum2 = masknum
                else:
                    masknum2 = int(numpy.floor(float(FirstendTarget - BeginingTarget) / 2.0))

                if FirstendTarget < len(signal):
                    lensignal = FirstendTarget + 1
                else:
                    lensignal = len(signal)

                if i == BeginingTarget:
                    minilist = numpy.array([signal[i], signal[i + 1]])
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        aveSignal.append(numpy.nanmean(minilist))
                elif i == lensignal - 1:
                    minilist = numpy.array([signal[i], signal[i - 1]])
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        aveSignal.append(numpy.nanmean(minilist))


                else:
                    averagemaker = []
                    if i < masknum2 - 1:
                        for j in range(i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(i):
                            averagemaker.append(signal[-j + i - 1])

                    elif i > lensignal - masknum2:
                        for j in range(lensignal - 1 - i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(lensignal - 1 - i):
                            averagemaker.append(signal[-j + i - 1])

                    else:
                        for j in range(masknum2):
                            averagemaker.append(signal[j + i])
                        for j in range(masknum2 - 1):
                            averagemaker.append(signal[-j + i - 1])

                    averagemaker = numpy.array(averagemaker)
                    meannnn = numpy.nanmean(averagemaker)
                    if str(signal[i]) != "nan" and str(meannnn) == "nan":
                        print "we have a problem at ", i, masknum2, lensignal, BeginingTarget, FirstendTarget, \
                            signal[i], meannnn
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        aveSignal.append(numpy.nanmean(averagemaker))
        return aveSignal

    ######################
    def myconvolve4(self, signal, masknum, TimeSignal):  # convolving algorithim that tapers averages toward edges and gaps of the time thing
        aveSignal = []
        #########
        checkingVariable = 0
        ########
        nextTarget = 1
        ##########
        FirstendTarget = 0
        ##########
        BeginingTarget = 1

        ########
        gapWidth = 15

        ########
        TimeThresh=float(masknum)/240.0

        #################
        ### full loop through the signal.
        for i in range(len(signal)):  # starts with i=0

            # beginning
            if i == 0:  ### start with two... what if they are both Nans?
                minilist = numpy.array([signal[i], signal[i + 1]])
                if str(signal[i]) == "nan":
                    aveSignal.append(numpy.nan)
                else:
                    if TimeSignal[i+1]-TimeSignal[i]<TimeThresh:
                        aveSignal.append(numpy.nanmean(minilist))
                    else:
                        aveSignal.append(signal[i])
            elif i == len(signal) - 1:  #### look at the end also with just two
                minilist = numpy.array([signal[i], signal[i - 1]])

                if str(signal[i]) == "nan":
                    aveSignal.append(numpy.nan)
                else:
                    if TimeSignal[i] - TimeSignal[i-1] < TimeThresh:
                        aveSignal.append(numpy.nanmean(minilist))
                    else:
                        aveSignal.append(signal[i])

            else:
                if nextTarget == i:
                    k = 0
                    nanflag = 0
                    while k < len(signal) - 1 - i:  # what is the i doing here?
                        # len(signal) - 1 - i is what is left of the signal
                        print "next target",nextTarget
                        if k == 0:
                            # marking the start with BeginingTarget
                            BeginingTarget = i

                        if str(signal[k + i]) == "nan" or (signal[k + i]-signal[k + i-1])>1.1/240.0:
                            # we are looking for a gap in nans or in time
                            nanflag = 1

                            # we have chosen a FirstendTarget, but are we at a gap or just a single nan
                            FirstendTarget = k + i - 1

                            if (signal[k + i]-signal[k + i-1])>TimeThresh:
                                nextTarget = k + i
                                k = len(signal) + 1000000 + i
                            else:
                                for g in range(len(signal) - (k + i)):
                                    # looking at the rest of the signal
                                    if str(signal[k + i + g]) != "nan":
                                        checkingVariable = 1

                                        if g > masknum:
                                            # if we have found a real number then we mark were that number is to start the process over again.
                                            nextTarget = k + i + g
                                            k = len(signal) + 1000000 + i
                                            break
                                        else:
                                            # if it is not bigger than the mask number then we keep going
                                            k = k + g - 1
                                            break
                        if k + i == len(signal) - 2:  # and nanflag==0:
                            # have we reached the end?
                            nextTarget = len(signal) + 1
                            FirstendTarget = len(signal) - 1

                        # incramenting k
                        k += 1

                if FirstendTarget - BeginingTarget > 2 * masknum + 1:
                    # looking to see if the path is big enough to make an average
                    masknum2 = masknum
                else:
                    masknum2 = int(numpy.floor(float(FirstendTarget - BeginingTarget) / 2.0))

                if FirstendTarget < len(signal):
                    lensignal = FirstendTarget + 1
                else:
                    lensignal = len(signal)

                if i == BeginingTarget:
                    minilist = numpy.array([signal[i], signal[i + 1]])
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        if TimeSignal[i+1] - TimeSignal[i] < TimeThresh:
                            aveSignal.append(numpy.nanmean(minilist))
                        else:
                            aveSignal.append(signal[i])

                elif i == lensignal - 1:
                    minilist = numpy.array([signal[i], signal[i - 1]])
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        if TimeSignal[i] - TimeSignal[i - 1] < TimeThresh:
                            aveSignal.append(numpy.nanmean(minilist))
                        else:
                            aveSignal.append(signal[i])



                else:
                    averagemaker = []
                    if i < masknum2 - 1:
                        for j in range(i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(i):
                            averagemaker.append(signal[-j + i - 1])

                    elif i > lensignal - masknum2:
                        for j in range(lensignal - 1 - i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(lensignal - 1 - i):
                            averagemaker.append(signal[-j + i - 1])

                    else:
                        for j in range(masknum2):
                            averagemaker.append(signal[j + i])
                        for j in range(masknum2 - 1):
                            averagemaker.append(signal[-j + i - 1])

                    averagemaker = numpy.array(averagemaker)
                    meannnn = numpy.nanmean(averagemaker)
                    if str(signal[i]) != "nan" and str(meannnn) == "nan":
                        print "we have a problem at ", i, masknum2, lensignal, BeginingTarget, FirstendTarget, \
                            signal[i], meannnn
                    if str(signal[i]) == "nan":
                        aveSignal.append(numpy.nan)
                    else:
                        aveSignal.append(numpy.nanmean(averagemaker))
        return aveSignal

    #######################
    def myconvolve3_justPointSpread(self, signal, masknum,i,nextTarget,FirstendTarget,BeginingTarget):
        #aveSignal = []
        checkingVariable = 0

        #for i in range(len(signal)):
        if 1:
            if i == 0:
                averagemaker = numpy.array([signal[i], signal[i + 1]])

            elif i == len(signal) - 1:
                averagemaker = numpy.array([signal[i], signal[i - 1]])



            else:
                if nextTarget == i:
                    k = 0
                    nanflag = 0
                    while k < len(signal) - 1 - i:
                        if k == 0:
                            BeginingTarget = i
                        if str(signal[k + i]) == "nan":
                            nanflag = 1
                            FirstendTarget = k + i - 1
                            for g in range(len(signal) - (k + i)):
                                if str(signal[k + i + g]) != "nan":
                                    checkingVariable = 1
                                    if g > masknum:
                                        nextTarget = k + i + g
                                        k = len(signal) + 1000000 + i
                                        break
                                    else:
                                        k = k + g - 1
                                        break
                        if k + i == len(signal) - 2:  # and nanflag==0:
                            nextTarget = len(signal) + 1
                            FirstendTarget = len(signal) - 1
                        k += 1

                if FirstendTarget - BeginingTarget > 2 * masknum + 1:
                    masknum2 = masknum
                else:
                    masknum2 = int(numpy.floor(float(FirstendTarget - BeginingTarget) / 2.0))

                if FirstendTarget < len(signal):
                    lensignal = FirstendTarget + 1
                else:
                    lensignal = len(signal)

                if i == BeginingTarget:
                    averagemaker = numpy.array([signal[i], signal[i + 1]])

                elif i == lensignal - 1:
                    averagemaker = numpy.array([signal[i], signal[i - 1]])


                else:
                    averagemaker = []
                    if i < masknum2 - 1:
                        for j in range(i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(i):
                            averagemaker.append(signal[-j + i - 1])

                    elif i > lensignal - masknum2:
                        for j in range(lensignal - 1 - i + 1):
                            averagemaker.append(signal[j + i])
                        for j in range(lensignal - 1 - i):
                            averagemaker.append(signal[-j + i - 1])

                    else:
                        for j in range(masknum2):
                            averagemaker.append(signal[j + i])
                        for j in range(masknum2 - 1):
                            averagemaker.append(signal[-j + i - 1])

                    averagemaker = numpy.array(averagemaker)
                    meannnn = numpy.nanmean(averagemaker)
                    if str(signal[i]) != "nan" and str(meannnn) == "nan":
                        print "we have a problem at ", i, masknum2, lensignal, BeginingTarget, FirstendTarget, \
                            signal[i], meannnn

        return averagemaker,nextTarget,FirstendTarget,BeginingTarget
####

    ######################################################################################################
    ######################################################################################################
    # perfoming a moving average arounss the path.
    def movingAverage(self, fwData, window):
        cameraNames = ["camera1", "camera2", "combined"]

        def myconvolve3(signal, masknum):
            aveSignal = []
            checkingVariable = 0
            nextTarget = 1
            FirstendTarget = 0
            BeginingTarget = 1
            for i in range(len(signal)):
                if i == 0:
                    minilist = numpy.array([signal[i], signal[i + 1]])
                    aveSignal.append(numpy.nanmean(minilist))
                elif i == len(signal) - 1:
                    minilist = numpy.array([signal[i], signal[i - 1]])

                    aveSignal.append(numpy.nanmean(minilist))

                else:
                    if nextTarget == i:
                        k = 0
                        nanflag = 0
                        while k < len(signal) - 1 - i:
                            if k == 0:
                                BeginingTarget = i
                            if str(signal[k + i]) == "nan":
                                nanflag = 1
                                FirstendTarget = k + i - 1
                                for g in range(len(signal) - (k + i)):
                                    if str(signal[k + i + g]) != "nan":
                                        checkingVariable = 1
                                        if g > masknum:
                                            nextTarget = k + i + g
                                            k = len(signal) + 1000000 + i
                                            break
                                        else:
                                            k = k + g - 1
                                            break
                            if k + i == len(signal) - 2:  # and nanflag==0:
                                nextTarget = len(signal) + 1
                                FirstendTarget = len(signal) - 1
                            k += 1

                    if FirstendTarget - BeginingTarget > 2 * masknum + 1:
                        masknum2 = masknum
                    else:
                        masknum2 = int(numpy.floor(float(FirstendTarget - BeginingTarget) / 2.0))

                    if FirstendTarget < len(signal):
                        lensignal = FirstendTarget + 1
                    else:
                        lensignal = len(signal)

                    if i == BeginingTarget:
                        minilist = numpy.array([signal[i], signal[i + 1]])
                        aveSignal.append(numpy.nanmean(minilist))
                    elif i == lensignal - 1:
                        minilist = numpy.array([signal[i], signal[i - 1]])
                        aveSignal.append(numpy.nanmean(minilist))

                    else:
                        averagemaker = []
                        if i < masknum2 - 1:
                            for j in range(i + 1):
                                averagemaker.append(signal[j + i])
                            for j in range(i):
                                averagemaker.append(signal[-j + i - 1])

                        elif i > lensignal - masknum2:
                            for j in range(lensignal - 1 - i + 1):
                                averagemaker.append(signal[j + i])
                            for j in range(lensignal - 1 - i):
                                averagemaker.append(signal[-j + i - 1])

                        else:
                            for j in range(masknum2):
                                averagemaker.append(signal[j + i])
                            for j in range(masknum2 - 1):
                                averagemaker.append(signal[-j + i - 1])

                        averagemaker = numpy.array(averagemaker)
                        meannnn = numpy.nanmean(averagemaker)
                        if str(signal[i]) != "nan" and str(meannnn) == "nan":
                            print "we have a problem at ", i, masknum2, lensignal, BeginingTarget, FirstendTarget, \
                            signal[i], meannnn
                        aveSignal.append(numpy.nanmean(averagemaker))
            return aveSignal

        for path in fwData[cameraNames[2]].keys():
            print path
            if path.split("_")[0] != "fiducial0":
                fwDataKeys = fwData[cameraNames[2]][path].keys()
                fwDataKeysInt = []
                print len(fwDataKeys), "fwDataKeys"
                if len(fwDataKeys) < 2:
                    continue
                for s in fwDataKeys:
                    if s != "FrameDelay":
                        fwDataKeysInt.append(int(s))
                fwDataKeysmin = min(fwDataKeysInt)
                fwDataKeysmax = max(fwDataKeysInt)

                inX = []
                inY = []
                inZ = []
                for jj in range(fwDataKeysmin, fwDataKeysmax + 1):
                    if 1:
                        try:
                            InterPoint = numpy.array(fwData[cameraNames[2]][path][str(jj)]['3Dpoint'])
                            inX.append(InterPoint[0])
                            inY.append(InterPoint[1])
                            inZ.append(InterPoint[2])
                        except:
                            inX.append(numpy.NaN)
                            inY.append(numpy.NaN)
                            inZ.append(numpy.NaN)
                if 1:
                    inXX = list(inX)
                    inYY = list(inY)
                    inZZ = list(inZ)

                    for i in range(3):
                        inX_avg = myconvolve3(inXX, window)
                        inY_avg = myconvolve3(inYY, window)
                        inZ_avg = myconvolve3(inZZ, window)

                        inXX = inX_avg
                        inYY = inY_avg
                        inZZ = inZ_avg

                for jj in range(fwDataKeysmin, fwDataKeysmax + 1):
                    try:
                        fwData[cameraNames[2]][path][str(jj)]['3Dpoint']
                    except:
                        continue
                    xinit = numpy.array(
                        [inX_avg[jj - fwDataKeysmin], inY_avg[jj - fwDataKeysmin], inZ_avg[jj - fwDataKeysmin]])
                    if str(inX_avg[jj - fwDataKeysmin]) == "nan":
                        print fwData[cameraNames[2]][path][str(jj)]['3Dpoint'][:], xinit, jj, jj - fwDataKeysmin
                    try:
                        fwData[cameraNames[2]][path][str(int(jj))].create_dataset(
                            'MovingAveragePoint', data=xinit)
                    except:
                        del fwData[cameraNames[2]][path][str(int(jj))]['MovingAveragePoint']
                        fwData[cameraNames[2]][path][str(int(jj))].create_dataset(
                            'MovingAveragePoint', data=xinit)








    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################


    # path analysis

    def createPathDic(self, Insectnumbers1, Insectnumbers2, ResultRot, ResultTran, fwData, UseTwo):
        # gathering info
        cameraNames = ["camera1", "camera2", "combined"]
        DictA = {}
        stumpKnob = numpy.array([-86.41750386, -151.3175956, 14.40064474])
        for i in range(len(Insectnumbers1)):
            j = Insectnumbers1[i]
            if UseTwo == 2:
                jjj = Insectnumbers2[i]
            DictA["inXk" + str(j) + "_" + "1"] = []
            DictA["inYk" + str(j) + "_" + "1"] = []
            DictA["inZk" + str(j) + "_" + "1"] = []
            DictA["dt" + str(j) + "_" + "1"] = []
            DictA["dtFirst" + str(j) + "_" + "1"] = []
            if UseTwo == 2:
                DictA["inXk" + str(jjj) + "_" + "2"] = []
                DictA["inYk" + str(jjj) + "_" + "2"] = []
                DictA["inZk" + str(jjj) + "_" + "2"] = []
                DictA["x2tp_" + str(jjj)] = []
            DictA["x12_" + str(j)] = []
            DictA["x1tp_" + str(j)] = []

            DictA["stumpKnob"] = stumpKnob
            inkeyName1 = "insect" + str(Insectnumbers1[i]) + "_" + "insect" + str(Insectnumbers1[i])
            if UseTwo == 2:
                inkeyName2 = "insect" + str(Insectnumbers2[i]) + "_" + "insect" + str(Insectnumbers2[i])
            didwestart = 0
            for jj in fwData[cameraNames[2]][inkeyName1]:

                if jj != "FrameDelay":

                    # if int(jj)>18000 and int(jj)<18400:
                    if 1:
                        try:

                            InterPointRaw = numpy.array(fwData[cameraNames[2]][inkeyName1][jj]['MovingAveragePoint'])

                            xKal = numpy.array([InterPointRaw[0], InterPointRaw[1], InterPointRaw[2]])

                            xKal = numpy.matmul(ResultRot.T, xKal) + ResultTran

                            if UseTwo == 2:
                                InterPointRaw = numpy.array(
                                    fwData[cameraNames[2]][inkeyName2][jj]['MovingAveragePoint'])

                                xKal2 = numpy.array([InterPointRaw[0], InterPointRaw[1], InterPointRaw[2]])

                                xKal2 = numpy.matmul(ResultRot.T, xKal2) + ResultTran

                            if didwestart == 0:
                                firststart = jj
                                DictA["dtFirst" + str(j) + "_" + "1"] = firststart
                                didwestart = 1
                                #                        dt.append((float(jj)-float(firststart)) / 240.0)
                            DictA["dt" + str(j) + "_" + "1"].append((float(jj)) / 240.0)

                            DictA["inXk" + str(j) + "_" + "1"].append(xKal[0])
                            DictA["inYk" + str(j) + "_" + "1"].append(xKal[1])
                            DictA["inZk" + str(j) + "_" + "1"].append(xKal[2])

                            DictA["x1tp_" + str(j)].append(
                                numpy.sqrt((xKal[0] - stumpKnob[0]) ** 2 +
                                           (xKal[1] - stumpKnob[1]) ** 2 +
                                           (xKal[2] - stumpKnob[2]) ** 2))

                            if UseTwo == 2:
                                DictA["inXk" + str(jjj) + "_" + "2"].append(xKal2[0])
                                DictA["inYk" + str(jjj) + "_" + "2"].append(xKal2[1])
                                DictA["inZk" + str(jjj) + "_" + "2"].append(xKal2[2])

                                DictA["x2tp_" + str(jjj)].append(
                                    numpy.sqrt((xKal2[0] - stumpKnob[0]) ** 2 +
                                               (xKal2[1] - stumpKnob[1]) ** 2 +
                                               (xKal2[2] - stumpKnob[2]) ** 2))
                            if UseTwo == 2:

                                if 1:
                                    DictA["x12_" + str(j)].append(
                                        numpy.sqrt((xKal[0] - xKal2[0]) ** 2 +
                                                   (xKal[1] - xKal2[1]) ** 2 +
                                                   (xKal[2] - xKal2[2]) ** 2))

                                if 0:
                                    DictA["x12_" + str(j)].append(
                                        numpy.sqrt((DictA["inXk" + str(j) + "1"][i] - stumpKnob[0]) ** 2 +
                                                   (DictA["inYk" + str(j) + "1"][i] - stumpKnob[1]) ** 2 +
                                                   (DictA["inZk" + str(j) + "1"][i] - stumpKnob[2]) ** 2))


                        except:
                            None

        return DictA



        ######################################################################################################
        ######################################################################################################
    def createMoreSTLPathDic2(self,PlotDesciptiveDic,DictA,obbTree, polydata):
        print "Doing STL and water"

        # this is just one point above surface
        def STLsurfacePycastVTK_dist2STL(obbTree, polydata, xKal):

            HitSTLfromabove=False
            if 1:
                pSource = [xKal[0], xKal[1], xKal[2]]
                pTarget = [xKal[0], xKal[1], xKal[2] - 10000]

                #  I think this is the initialization of the cell IDs and the points they will be assigned when code is run.
                pointsVTKintersection = vtk.vtkPoints()
                idsVTKintersection = vtk.vtkIdList()

                # this is the ray cast with obbtree.
                code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection,
                                                 idsVTKintersection)

                # this is the collection of intersection points from that ray in the mesh.  we are only going to choose the first one.
                pointsVTKIntersectionData = pointsVTKintersection.GetData()
                # this is the number of points interested with the line.
                noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

                if noPointsVTKIntersection != 0:
                    inter1 = [pointsVTKIntersectionData.GetTuple3(0)[0],
                              pointsVTKIntersectionData.GetTuple3(0)[1],
                              pointsVTKIntersectionData.GetTuple3(0)[2]]
                    inter1 = numpy.array(inter1)
                    xKal = numpy.array(xKal)
                    distz=numpy.linalg.norm(inter1-xKal)
                    #if didwestart == 0:
                    #    firststart = jj
                    #    didwestart = 1
                    # dt.append((float(jj) - float(firststart)) / 240.0)
                    HitSTLfromabove = True


                #except:
                #    None

                ########################################################################################
                ######################################################################################
                ###

                ##                              trying to find the minimum distance from the point to the mesh

                #### only do this if the point is above the mesh in the world frame
                if HitSTLfromabove == True:
                    # xKal

                    ### putting all the points in arrays.
                    Alldistancepoint = []
                    AllthePoints = []
                    for ig in range(polydata.GetNumberOfCells()):
                        # looking at all the verticies in a triangle
                        vertThing = []
                        distancepoint = []
                        for vv in range(3):
                            vert = numpy.array(polydata.GetCell(ig).GetPoints().GetPoint(vv))
                            vertThing.append(vert)

                            distancepoint.append(numpy.linalg.norm(vert - xKal))
                        Alldistancepoint.append(distancepoint)
                        AllthePoints.append(vertThing)

                    Alldistancepoint = numpy.array(Alldistancepoint)
                    AllthePoints = numpy.array(AllthePoints)

                    # finding the minimum and the minimum places.
                    minElement = numpy.amin(Alldistancepoint)
                    # print minElement

                    # this gives the coordinates of the minima
                    result = numpy.where(Alldistancepoint == numpy.amin(Alldistancepoint))
                    # print "result",result[0],result[1]

                    # setting the angles between the legs of the triangle and the path point
                    angleOnTriangle = []
                    triangleLeggs = []
                    angleQualityCount = []

                    # getting the minimum point and the other two points in the trianle
                    for iu in range(len(result[0])):

                        minpoint = AllthePoints[result[0][iu]][result[1][iu]]
                        for iy in range(3):
                            if iy != result[1][iu]:
                                leg1 = AllthePoints[result[0][iu]][iy]
                                iyy = iy
                                break
                        for iy in range(3):
                            if iy != result[1][iu] and iy != iyy:
                                leg2 = AllthePoints[result[0][iu]][iy]
                                break

                        # print AllthePoints[result[0][iu]]
                        # print minpoint,leg1,leg2

                        v11 = leg1 - minpoint
                        v12 = xKal - minpoint
                        v21 = leg2 - minpoint

                        ang1 = numpy.arccos(
                            v11.dot(v12) / (numpy.linalg.norm(v12) * numpy.linalg.norm(v11))) * (
                                           180 / 3.14159)
                        ang2 = numpy.arccos(
                            v21.dot(v12) / (numpy.linalg.norm(v12) * numpy.linalg.norm(v21))) * (
                                       180 / 3.14159)

                        # print ang1,ang2
                        angleOnTriangle.append([ang1, ang2])
                        triangleLeggs.append([v11, v21, v12])

                        if ang1 < 90 and ang2 < 90:
                            angleQualityCount.append(2)
                        elif (ang1 < 90 and ang2 >= 90) or (ang2 < 90 and ang1 >= 90):
                            angleQualityCount.append(1)
                        elif ang1 >= 90 and ang2 >= 90:
                            angleQualityCount.append(0)

                    angleQualityCount = numpy.array(angleQualityCount)
                    Zeroscount = numpy.where(angleQualityCount == 0)[0]
                    OnesCount = numpy.where(angleQualityCount == 1)[0]
                    TwosCount = numpy.where(angleQualityCount == 2)[0]

                    # print angleQualityCount

                    if len(Zeroscount) == len(angleQualityCount):
                        # print "all zeros"
                        None
                        dist=numpy.linalg.norm(v12)

                    elif len(OnesCount) > 0 and len(TwosCount) == 0:
                        # print "no twos but ones"
                        None
                        result2 = numpy.where(angleOnTriangle == numpy.amin(angleOnTriangle))
                        # print result2
                        v12 = triangleLeggs[result2[0][0]][2]
                        v11 = triangleLeggs[result2[0][0]][result2[1][0]]
                        minElement = numpy.amin(angleOnTriangle)
                        # print minElement
                        dist1 = numpy.sin(minElement * 3.1415926 / 180.0) * numpy.linalg.norm(v12)
                        # print minElement,dist1
                        dist=dist1
                        if len(OnesCount) > 1:
                            # print "just one one"
                            None
                    # elif len(TwosCount)==1:
                    # print "just one two"
                    #    None
                    elif len(TwosCount) >= 1:
                        # print "more than one two"
                        None
                        result2 = numpy.where(angleOnTriangle == numpy.amin(angleOnTriangle))
                        minsss = set(result2[0])
                        minsss2 = set(TwosCount)
                        minsint = (minsss.intersection(minsss2))
                        if len(minsint) == 0:
                            print "no"  # if the minimum angle is not on a traingle with two acute angles then it we just take is as a line
                            result2 = numpy.where(angleOnTriangle == numpy.amin(angleOnTriangle))
                            # print result2
                            v12 = triangleLeggs[result2[0][0]][2]
                            v11 = triangleLeggs[result2[0][0]][result2[1][0]]
                            minElement = numpy.amin(angleOnTriangle)
                            # print minElement
                            dist1 = numpy.sin(minElement * 3.1415926 / 180.0) * numpy.linalg.norm(v12)
                            # print minElement,dist1
                            dist=dist1


                        # elif len(minsint)>2:
                        else:

                            ################    we want to find the triangle with the lowest angle with also the other lowest angle.
                            anglesum = []
                            anglesumindex = []
                            for i in minsint:  # finding the lowest angle sum on these triangles.
                                anglesum.append(angleOnTriangle[i][0] + angleOnTriangle[i][1])
                                anglesumindex.append(i)
                            # print len(anglesum)
                            # print numpy.amin(anglesum)
                            result3 = numpy.where(anglesum == numpy.amin(anglesum))
                            minElement = numpy.amin(angleOnTriangle)
                            # print minElement
                            # print angleOnTriangle[anglesumindex[result3[0][0]]]
                            triangleLeggs.append([v11, v21, v12])
                            v11 = triangleLeggs[anglesumindex[result3[0][0]]][0]
                            v21 = triangleLeggs[anglesumindex[result3[0][0]]][1]
                            v12 = triangleLeggs[anglesumindex[result3[0][0]]][2]
                            normv1121 = numpy.cross(v11, v21)
                            normv1121 = normv1121 / numpy.linalg.norm(normv1121)

                            # we are looking for the distance on the vertex point with the vecto and the projection of that
                            # vertex point with the normal of the plane
                            # print abs(normv1121.dot(v12)),"abs(normv1121.dot(v12))"

                            # is this right....?
                            # we are taking the abs value of it because just looking at the distance.
                            dist=abs(normv1121.dot(v12))


                    else:
                        print "what else is there"

            if HitSTLfromabove == True:

                return distz,dist
            else:
                return numpy.nan,numpy.nan



##################################################################################################
        for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):
            ###################################
            # Initializing the dictionary data
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):

                disSTL=[]
                disSTLz=[]

                for i in range(len(DictA["dt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)])):

                    xto=DictA["xt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]
                    yto=DictA["yt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]
                    zto=DictA["zt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]

                    xKal=[xto,yto,zto]
                    distz,dist= STLsurfacePycastVTK_dist2STL(obbTree, polydata, xKal)
                    print distz,dist
                    disSTL.append(dist)
                    disSTLz.append(distz)

                DictA[
                    "disSTL" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = disSTL
                DictA[
                    "disSTLz" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = disSTLz
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        return DictA
        # path analysis--- creating the data dictionary

    def createMorePathDic2(self,PlotDesciptiveDic,DictA,KinPlotDesciptiveDic ):
        print "Doing velocity and accelleration"
        window = int(PlotDesciptiveDic["Smoothing Window"])#17
        dtTresh=2.1/240.0
        #dtTresh = 2.0 / 240.0
        ConvolveRepeat=int(PlotDesciptiveDic["Smoothing Iteration"])#3
        DifferentiationTresh=True
        # aqwq

        avg_mask = numpy.ones(window) / window
        RefVectStr=PlotDesciptiveDic["Reference vector"]
        ReferenceVector=numpy.array([float(RefVectStr.split(",")[0]),float(RefVectStr.split(",")[1]),float(RefVectStr.split(",")[2])])
        Water_Surface_Plane=PlotDesciptiveDic["Water_Surface_Plane"]
        DictErr={}
        stumpKnob = PlotDesciptiveDic["PointofReference"]
        if PlotDesciptiveDic["Use spline instead of average"]=="n":
            ConvolveThing = 3  # 0 regular convolve, 1 sf convolve, 2 spline, 3 myconvolve
            splorder = 3
            bfactor = 2

        elif PlotDesciptiveDic["Use spline instead of average"]=="y":

            ConvolveThing = 2  # 0 regular convolve, 1 sf convolve, 2 spline, 3 myconvolve
            splorder = int(PlotDesciptiveDic["Spline order"])
            bfactor = float(PlotDesciptiveDic["Spline smoothing factor"])

        ########################################aqwq
        #The spline stuff.

        ztsplk=splorder
        ztspls=200
        xtsplk=splorder
        xtspls=20000
        ytsplk=splorder
        ytspls=20000
        xpsplk=splorder
        xpspls=20000
        wdsplk=splorder
        wdspls=20000
        disSTLsplk=splorder
        disSTLspls=20000
        disSTLzsplk=splorder
        disSTLzspls=20000
        xbsplk=splorder
        xbspls=20000




        if Water_Surface_Plane!="":
            abcd = []
            for ic in range(4):
                abcd.append(float(Water_Surface_Plane.split(";")[1].split(",")[ic]))

        #xd = x1 - x2
        # This is the up vector
        zup = numpy.array([0, 0, 1])
        def dragonflyFieldOfVision(v, xd, zup):# in degrees.
            vz = numpy.cross(numpy.cross(v, zup), v)
            vxd = numpy.cross(numpy.cross(v, xd), v)

            vz = vz / numpy.linalg.norm(vz)
            vxd = vxd / numpy.linalg.norm(vxd)

            cosang = numpy.dot(vz, vxd)

            sinangSine = numpy.dot(v, (numpy.cross(vz, vxd))) / abs(numpy.dot(v, (numpy.cross(vz, vxd))))
            sinang = sinangSine * numpy.linalg.norm(numpy.cross(vz, vxd))

            phi = numpy.arctan2(sinang, cosang) *180/3.1415926

            theta = numpy.arccos(numpy.dot(v, xd) / (numpy.linalg.norm(v) * numpy.linalg.norm(xd))) *180/3.1415926
            return theta, phi

        for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):
            ###################################
            # Initializing the dictionary data
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                dt = []
                xt = []
                yt = []
                zt = []
                xtRaw = []
                ytRaw = []
                ztRaw = []

                xp = []
                wd=[]


                for i in range(len(DictA["dt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)])):
                    dt.append(DictA["dt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i])

                    xto=DictA["xt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]
                    yto=DictA["yt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]
                    zto=DictA["zt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i]

                    xtRaw.append(DictA["xt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i])
                    ytRaw.append(DictA["yt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i])
                    ztRaw.append(DictA["zt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)][i])
                    #xp.append(DictA["xp" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j])+ "_" + str(ii+1)][i])
                    xp.append(
                        numpy.sqrt((xto - stumpKnob[0]) ** 2 +
                                   (yto - stumpKnob[1]) ** 2 +
                                   (zto - stumpKnob[2]) ** 2))

                    if Water_Surface_Plane != "":
                        distance = abs(
                            xto * abcd[0] + yto* abcd[1] + zto* abcd[2] + abcd[3]) / numpy.sqrt(
                            abcd[0] ** 2 + abcd[1] ** 2 + abcd[2] ** 2)
                    else:
                        distance=numpy.nan
                    wd.append(distance)


                dt = numpy.array(dt)



                xtRaw = numpy.array(xtRaw)
                ytRaw = numpy.array(ytRaw)
                ztRaw = numpy.array(ztRaw)

                wd=numpy.array(wd)
                xp = numpy.array(xp)

                if KinPlotDesciptiveDic["Calculate distance to STL"] == "y":
                    disSTL = DictA[
                        "disSTL" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)]
                    disSTLz = DictA[
                        "disSTLz" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)]
                else:
                    disSTL=numpy.copy(ztRaw)
                    disSTL[:]=numpy.nan
                    disSTLz=numpy.copy(disSTL)
                    disSTLz_avg = numpy.copy(disSTL)
                    disSTL_avg = numpy.copy(disSTL)
                    DictA[
                        "disSTL_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTL_avg
                    DictA[
                        "disSTLz_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTLz_avg
                    DictA[
                        "disSTL" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTL
                    DictA[
                        "disSTLz" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTLz


                delxt = []
                delyt = []
                delzt = []
                delxp = []
                delwd = []
                deldisSTL = []
                deldisSTLz = []
                PCAsigma = []
                PCAmu = []
                PCAwt = []

                if 1:
                    nextTarget = 1
                    FirstendTarget = 0
                    BeginingTarget = 1
                    for i in range(len(xtRaw)):
                        xtSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(xtRaw, (
                                    window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)
                        ytSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(ytRaw, (
                                    window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)
                        ztSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(ztRaw, (
                                    window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)
                        xpSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(xp, (
                                window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)
                        wdSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(wd, (
                                window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)

                        if KinPlotDesciptiveDic["Calculate distance to STL"] == "y":
                            disSTLSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(
                                disSTL, (window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)
                            disSTLzSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(
                                disSTLz, (window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)

                        # print len(xtSrd),"xtSrd"
                        nextTarget, FirstendTarget, BeginingTarget = nextTarget1, FirstendTarget1, BeginingTarget1

                        delxt.append(numpy.std(xtSrd))
                        delyt.append(numpy.std(ytSrd))
                        delzt.append(numpy.std(ztSrd))
                        delxp.append(numpy.std(xpSrd))
                        delwd.append(numpy.std(wdSrd))
                        if KinPlotDesciptiveDic["Calculate distance to STL"] == "y":
                            deldisSTL.append(numpy.std(disSTLSrd))
                            deldisSTLz.append(numpy.std(disSTLzSrd))
                        errorSrd = []

                        for jr in range(len(xtSrd)):
                            errorSrd.append([xtSrd[jr], ytSrd[jr], ztSrd[jr]])
                        # if (len(xtSrd))>3:
                        try:
                            errorSrd = numpy.array(errorSrd)
                            pcapoint = mlabPCA(errorSrd)
                            # print "a",pcapoint.a
                            # print "sigma", pcapoint.sigma
                            # print "WT", pcapoint.Wt
                            # print "MU", pcapoint.mu
                            PCAsigma.append(pcapoint.sigma)
                            PCAmu.append(pcapoint.mu)
                            PCAwt.append(pcapoint.Wt)
                        # else:
                        except:
                            PCAsigma.append(numpy.array([numpy.nan, numpy.nan, numpy.nan]))
                            PCAmu.append(numpy.array([numpy.nan, numpy.nan, numpy.nan]))
                            PCAwt.append(numpy.array(
                                [[numpy.nan, numpy.nan, numpy.nan], [numpy.nan, numpy.nan, numpy.nan],
                                 [numpy.nan, numpy.nan, numpy.nan]]))

                DictA[
                    "delxt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delxt
                delxtavg=numpy.nanmean(delxt)
                DictA[
                    "delyt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delyt

                delytavg=numpy.nanmean(delyt)

                DictA[
                    "delzt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delzt

                delztavg=numpy.nanmean(delzt)


                DictA[
                    "delxp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delxp
                delxpavg=numpy.nanmean(delxp)

                DictA[
                    "delwd" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delwd
                delwdavg=numpy.nanmean(delwd)

                if KinPlotDesciptiveDic["Calculate distance to STL"] == "y":
                    DictA[
                        "deldisSTL" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = deldisSTL
                    deldisSTLavg = numpy.nanmean(deldisSTL)

                    DictA[
                        "deldisSTLz" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = deldisSTLz
                    deldisSTLzavg = numpy.nanmean(deldisSTLz)

                DictA[
                    "PCAsigma" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = PCAsigma
                DictA[
                    "PCAmu" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = PCAmu
                DictA[
                    "PCAwt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = PCAwt

                DictErr["xt"] = "delxt"
                DictErr["yt"] = "delyt"
                DictErr["zt"] = "delzt"

                DictErr["xt_avg"] = "delxt"
                DictErr["yt_avg"] = "delyt"
                DictErr["zt_avg"] = "delzt"

                DictErr["xp_avg"] = "delxp"
                DictErr["wd_avg"] = "delwd"
                DictErr["xp"] = "delxp"
                DictErr["wd"] = "delwd"
                DictErr["disSTL"] = "deldisSTL"
                DictErr["disSTLz"] = "deldisSTLz"

                ################################################################################################################################################################

                xtRaw_sf = numpy.zeros(len(xtRaw))
                ytRaw_sf = numpy.zeros(len(ytRaw))
                ztRaw_sf = numpy.zeros(len(ztRaw))
                xp_avg=numpy.zeros(len(xp))
                wd_avg = numpy.zeros(len(wd))

                if ConvolveThing==0:
                    xtRaw_sf[:] = numpy.convolve(xtRaw, avg_mask, 'same')  # running average
                    ytRaw_sf[:] = numpy.convolve(ytRaw, avg_mask, 'same')  # running average
                    ztRaw_sf[:] = numpy.convolve(ztRaw, avg_mask, 'same')  # running average
                    xp_avg[:] = numpy.convolve(xp, avg_mask, 'same')  # running average
                    wd_avg[:] = numpy.convolve(wd, avg_mask, 'same')  # running average



                elif ConvolveThing==1:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            xtRaw_sf[:] = savgol_filter(xtRaw, window, 1)
                            ytRaw_sf[:] = savgol_filter(ytRaw, window, 1)
                            ztRaw_sf[:] = savgol_filter(ztRaw, window, 1)
                            xp_avg[:] = savgol_filter(xp, window, 1)
                            wd_avg[:] = savgol_filter(wd, window, 1)

                        else:
                            xtRaw_sf[:] = savgol_filter(xtRaw_sf, window, 1)
                            ytRaw_sf[:] = savgol_filter(ytRaw_sf, window, 1)
                            ztRaw_sf[:] = savgol_filter(ztRaw_sf, window, 1)
                            xp_avg[:] = savgol_filter(xp_avg, window, 1)
                            wd_avg[:] = savgol_filter(wd_avg, window, 1)

                elif ConvolveThing==3:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            xtRaw_sf[:] = self.myconvolve3(xtRaw, (window - 1) / 2)
                            ytRaw_sf[:] = self.myconvolve3(ytRaw, (window - 1) / 2)
                            ztRaw_sf[:] = self.myconvolve3(ztRaw, (window - 1) / 2)
                            xp_avg[:] = self.myconvolve3(xp, (window - 1) / 2)
                            wd_avg[:] = self.myconvolve3(wd, (window - 1) / 2)

                        else:
                            xtRaw_sf[:] = self.myconvolve3(xtRaw_sf, (window - 1) / 2)
                            ytRaw_sf[:] = self.myconvolve3(ytRaw_sf, (window - 1) / 2)
                            ztRaw_sf[:] = self.myconvolve3(ztRaw_sf, (window - 1) / 2)
                            xp_avg[:] = self.myconvolve3(xp_avg, (window - 1) / 2)
                            wd_avg[:] = self.myconvolve3(wd_avg, (window - 1) / 2)

                elif ConvolveThing==2:

                    if 1:
                        xtRawCP=[]
                        dtCPx=[]
                        ytRawCP = []
                        dtCPy = []
                        ztRawCP = []
                        dtCPz = []
                        xpCP=[]
                        dtCPxp=[]
                        wdCP=[]
                        dtCPwd=[]
                        for sp in range(len(xtRaw)):
                            if numpy.isnan(xtRaw[sp])==False:
                                dtCPx.append(dt[sp])
                                xtRawCP.append(xtRaw[sp])
                            if numpy.isnan(ytRaw[sp])==False:
                                dtCPy.append(dt[sp])
                                ytRawCP.append(ytRaw[sp])
                            if numpy.isnan(ztRaw[sp])==False:
                                dtCPz.append(dt[sp])
                                ztRawCP.append(ztRaw[sp])
                            if numpy.isnan(xp[sp])==False:
                                dtCPxp.append(dt[sp])
                                xpCP.append(xp[sp])
                            if numpy.isnan(wd[sp])==False:
                                dtCPwd.append(dt[sp])
                                wdCP.append(wd[sp])

                        dtCPx=numpy.array(dtCPx)
                        xtRawCP=numpy.array(xtRawCP)
                        dtCPy = numpy.array(dtCPy)
                        ytRawCP = numpy.array(ytRawCP)
                        dtCPz = numpy.array(dtCPz)
                        ztRawCP = numpy.array(ztRawCP)

                        dtCPxp = numpy.array(dtCPxp)
                        xpCP = numpy.array(xpCP)
                        dtCPwd = numpy.array(dtCPwd)
                        wdCP = numpy.array(wdCP)
                        #aqwq

                        ztspl = interpolate.UnivariateSpline(dtCPz, ztRawCP, k=ztsplk,s=bfactor*(len(ztRawCP)-1)*delztavg**2)
                        xtspl = interpolate.UnivariateSpline(dtCPx, xtRawCP,k=xtsplk,s=bfactor*(len(xtRawCP)-1)*delxtavg**2)
                        ytspl = interpolate.UnivariateSpline(dtCPy, ytRawCP,k=ytsplk,s=bfactor*(len(ytRawCP)-1)*delytavg**2)

                        #print (len(ztRawCP)-1)*delztavg**2,"zs spline"
                        #print (len(xtRawCP)-1)*delxtavg**2, 'xs spline'
                        #print (len(ytRawCP)-1)*delytavg**2,"ys spline"
                        xpspl = interpolate.UnivariateSpline(dtCPxp, xpCP, k=xpsplk, s=bfactor*(len(xpCP)-1)*delxpavg**2)
                        if len(wdCP)!=0:
                            wdspl = interpolate.UnivariateSpline(dtCPwd, wdCP, k=wdsplk, s=bfactor*(len(wdCP)-1)*delwdavg**2)

                        if 1:
                            xtRaw_sf = xtspl(dt)
                            ytRaw_sf = ytspl(dt)
                            ztRaw_sf = ztspl(dt)

                            xp_avg=xpspl(dt)
                            if len(wdCP) != 0:
                                wd_avg=wdspl(dt)
                            else:
                                wd_avg=numpy.copy(wd)
                            wwz = numpy.isnan(ztRaw)
                            wwy = numpy.isnan(ytRaw)
                            wwx = numpy.isnan(xtRaw)
                            wwxp = numpy.isnan(xp)
                            wwwd = numpy.isnan(wd)
                            ztRaw_sf[wwz] = numpy.nan
                            ytRaw_sf[wwy] = numpy.nan
                            xtRaw_sf[wwx] = numpy.nan
                            xp_avg[wwxp] = numpy.nan
                            wd_avg[wwwd] = numpy.nan




                        #print "xtRaw_sf", xtRaw_sf
                        #print "ytRaw_sf", ytRaw_sf
                        #print "ztRaw_sf", ztRaw_sf

                    #print "xtRaw_sf", xtRaw_sf

                xt_avg=xtRaw_sf
                yt_avg=ytRaw_sf
                zt_avg=ztRaw_sf
                #print "xt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                #        ii + 1),xt_avg

                DictA["xt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = xt_avg
                DictA["yt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = yt_avg
                DictA["zt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = zt_avg

                DictA[
                    "wd" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = wd
                DictA[
                    "wd_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = wd_avg
                DictA[
                    "xp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = xp

                DictA[
                    "xp_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = xp_avg

                ################################################################################################################################################################
                ################################################################################################################################################################

                if KinPlotDesciptiveDic["Calculate distance to STL"] == "y":

                    disSTL_avg = numpy.zeros(len(disSTL))
                    disSTLz_avg = numpy.zeros(len(disSTLz))

                    # print len(vxt)
                    if ConvolveThing == 0:
                        disSTL_avg[:] = numpy.convolve(disSTL, avg_mask, 'same')  # running average
                        disSTLz_avg[:] = numpy.convolve(disSTLz, avg_mask, 'same')  # running average

                    elif ConvolveThing == 1:
                        for yi in range(ConvolveRepeat):
                            if yi == 0:
                                disSTL_avg[:] = savgol_filter(disSTL, window, 1)
                                disSTLz_avg[:] = savgol_filter(disSTLz, window, 1)

                            else:
                                disSTL_avg[:] = savgol_filter(disSTL_avg, window, 1)
                                disSTLz_avg[:] = savgol_filter(disSTLz_avg, window, 1)



                    elif ConvolveThing == 3:
                        for yi in range(ConvolveRepeat):
                            if yi == 0:
                                disSTL_avg[:] = self.myconvolve3(disSTL, (window - 1) / 2)
                                disSTLz_avg[:] = self.myconvolve3(disSTLz, (window - 1) / 2)

                            else:
                                disSTL_avg[:] = self.myconvolve3(disSTL_avg, (window - 1) / 2)
                                disSTLz_avg[:] = self.myconvolve3(disSTLz_avg, (window - 1) / 2)
                    elif ConvolveThing == 2:
                        disSTLCP=[]
                        dtCPx=[]
                        disSTLzCP = []
                        dtCPy = []
                        for sp in range(len(disSTL)):
                            if numpy.isnan(disSTL[sp])==False:
                                dtCPx.append(dt[sp])
                                disSTLCP.append(disSTL[sp])
                            if numpy.isnan(disSTLz[sp])==False:
                                dtCPy.append(dt[sp])
                                disSTLzCP.append(disSTLz[sp])
                        dtCPx=numpy.array(dtCPx)
                        disSTLCP=numpy.array(disSTLCP)
                        dtCPy = numpy.array(dtCPy)
                        disSTLzCP = numpy.array(disSTLzCP)

                        disSTLspl = interpolate.UnivariateSpline(dtCPx, disSTLCP, k=disSTLsplk, s=bfactor*(len(disSTLCP)-1)*deldisSTLavg**2)
                        disSTLzspl = interpolate.UnivariateSpline(dtCPy, disSTLzCP, k=disSTLzsplk, s=bfactor*(len(disSTLzCP)-1)*deldisSTLzavg**2)

                        disSTL_avg = disSTLspl(dt)
                        disSTLz_avg = disSTLzspl(dt)

                        wwL = numpy.isnan(disSTL)
                        wwLz = numpy.isnan(disSTLz)

                        disSTL_avg[wwL] = numpy.nan
                        disSTLz_avg[wwLz] = numpy.nan



                    DictA[
                        "disSTL_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTL_avg
                    DictA[
                        "disSTLz_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                            ii + 1)] = disSTLz_avg





                           ################################################################################################################################################################

                # manually differentiating xt, yt, zt
                # another velocity
                ##############################
                vxt = []
                vyt = []
                vzt = []
                vxp = []
                dtv = []
                delvxt=[]
                delvyt=[]
                delvzt=[]
                delvxp = []



                if ConvolveThing == 2:  # better differentiation
                    ################################ spline derivative
                    vztspl = ztspl.derivative(1)
                    vzt = vztspl(dt)
                    vzt = numpy.delete(vzt, -1, 0)
                    vzt = numpy.delete(vzt, 0, 0)
                    vytspl = ytspl.derivative(1)
                    vyt = vytspl(dt)
                    vyt = numpy.delete(vyt, -1, 0)
                    vyt = numpy.delete(vyt, 0, 0)
                    vxtspl = xtspl.derivative(1)
                    vxt = vxtspl(dt)
                    vxt = numpy.delete(vxt, -1, 0)
                    vxt = numpy.delete(vxt, 0, 0)
                    vxpspl = xpspl.derivative(1)
                    vxp = vxpspl(dt)
                    vxp = numpy.delete(vxp, -1, 0)
                    vxp = numpy.delete(vxp, 0, 0)
                    delvxt=numpy.copy(vxt)
                    delvxt[:]=numpy.nan
                    delvyt = numpy.copy(delvxt)
                    delvzt = numpy.copy(delvxt)
                    delvxp = numpy.copy(delvxt)
                    dtv = numpy.copy(dt)
                    dtv = numpy.delete(dtv, -1, 0)
                    dtv = numpy.delete(dtv, 0, 0)
                else:
                    for i in range(len(xt_avg) - 2):
                        if DifferentiationTresh==False:
                            vxt.append((xt_avg[i + 2] - xt_avg[i]) / ((dt[i + 2] - dt[i])))
                            vyt.append((yt_avg[i + 2] - yt_avg[i]) / ((dt[i + 2] - dt[i])))
                            vzt.append((zt_avg[i + 2] - zt_avg[i]) / ((dt[i + 2] - dt[i])))
                            vxp.append((xp_avg[i + 2] - xp_avg[i]) / ((dt[i + 2] - dt[i])))

                            delvxt.append(numpy.sqrt(2)*delxt[i+1]/ ((dt[i + 2] - dt[i])))
                            delvyt.append(numpy.sqrt(2)*delyt[i+1] / ((dt[i + 2] - dt[i])))
                            delvzt.append(numpy.sqrt(2)*delzt[i+1] / ((dt[i + 2] - dt[i])))
                            delvxp.append(numpy.sqrt(2)*delxp[i+1] / ((dt[i + 2] - dt[i])))

                            dtv.append(dt[i + 1])

                        else:
                            #print i, dt[i],dt[i + 2] - dt[i],dtTresh
                            if (dt[i + 2] - dt[i])<dtTresh:
                                vxt.append((xt_avg[i + 2] - xt_avg[i]) / ((dt[i + 2] - dt[i])))
                                vyt.append((yt_avg[i + 2] - yt_avg[i]) / ((dt[i + 2] - dt[i])))
                                vzt.append((zt_avg[i + 2] - zt_avg[i]) / ((dt[i + 2] - dt[i])))
                                vxp.append((xp_avg[i + 2] - xp_avg[i]) / ((dt[i + 2] - dt[i])))

                                delvxt.append(numpy.sqrt(2) * delxt[i + 1] / ((dt[i + 2] - dt[i])))
                                delvyt.append(numpy.sqrt(2) * delyt[i + 1] / ((dt[i + 2] - dt[i])))
                                delvzt.append(numpy.sqrt(2) * delzt[i + 1] / ((dt[i + 2] - dt[i])))
                                delvxp.append(numpy.sqrt(2) * delxp[i + 1] / ((dt[i + 2] - dt[i])))

                                dtv.append(dt[i + 1])
                            else:
                                vxt.append(numpy.nan)
                                vyt.append(numpy.nan)
                                vzt.append(numpy.nan)
                                vxp.append(numpy.nan)

                                delvxt.append(numpy.nan)
                                delvyt.append(numpy.nan)
                                delvzt.append(numpy.nan)
                                delvxp.append(numpy.nan)

                                dtv.append(dt[i + 1])


                vxt = numpy.array(vxt)
                vyt = numpy.array(vyt)
                vzt = numpy.array(vzt)
                vxp = numpy.array(vxp)

                delvxt = numpy.array(delvxt)
                delvyt = numpy.array(delvyt)
                delvzt = numpy.array(delvzt)
                delvxp = numpy.array(delvxp)

                dtv = numpy.array(dtv)
               # print len(dtv),"len(dtv)",len(dt),"len(dt)",len(xt_avg),"len(xt_avg)"
                DictA[
                    "vxt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vxt
                DictA[
                    "vyt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vyt
                DictA[
                    "vzt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vzt

                DictA[
                    "vxp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vxp


                DictA[
                    "delvxt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delvxt
                DictA[
                    "delvyt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delvyt
                DictA[
                    "delvzt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delvzt

                DictA[
                    "delvxp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = delvxp


                DictA[
                    "dtv" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = dtv



                DictErr["vxt"]="delvxt"
                DictErr["vyt"]="delvyt"
                DictErr["vzt"]="delvzt"

                DictErr["vxt_avg"]="delvxt"
                DictErr["vyt_avg"]="delvyt"
                DictErr["vzt_avg"]="delvzt"
                DictErr["vxp"]="delvxp"
                DictErr["vxp_avg"]="delvxp"


                ################################################################################################################################################################
                ################################################################################################################################################################

                ################################################################################################################################
                #convolve velocity
                ################################################################################################################################
                vxt_avg = numpy.zeros(len(vxt))
                vyt_avg = numpy.zeros(len(vyt))
                vzt_avg = numpy.zeros(len(vzt))
                vxp_avg = numpy.zeros(len(vxp))


                # print len(vxt)
                if ConvolveThing==0:
                    vxt_avg[:] = numpy.convolve(vxt, avg_mask, 'same')  # running average
                    vyt_avg[:] = numpy.convolve(vyt, avg_mask, 'same')  # running average
                    vzt_avg[:] = numpy.convolve(vzt, avg_mask, 'same')  # running average
                    vxp_avg[:] = numpy.convolve(vxp, avg_mask, 'same')  # running average
                elif ConvolveThing==1:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vxt_avg[:] = savgol_filter(vxt, window, 1)
                            vyt_avg[:] = savgol_filter(vyt, window, 1)
                            vzt_avg[:] = savgol_filter(vzt, window, 1)
                            vxp_avg[:] = savgol_filter(vxp, window, 1)
                        else:
                            vxt_avg[:] = savgol_filter(vxt_avg, window, 1)
                            vyt_avg[:] = savgol_filter(vyt_avg, window, 1)
                            vzt_avg[:] = savgol_filter(vzt_avg, window, 1)
                            vxp_avg[:] = savgol_filter(vxp_avg, window, 1)


                elif ConvolveThing==2 or ConvolveThing == 3:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vxt_avg[:] = self.myconvolve3(vxt, (window - 1) / 2)
                            vyt_avg[:] = self.myconvolve3(vyt, (window - 1) / 2)
                            vzt_avg[:] = self.myconvolve3(vzt, (window - 1) / 2)
                            vxp_avg[:] = self.myconvolve3(vxp, (window - 1) / 2)
                        else:
                            vxt_avg[:] = self.myconvolve3(vxt_avg, (window - 1) / 2)
                            vyt_avg[:] = self.myconvolve3(vyt_avg, (window - 1) / 2)
                            vzt_avg[:] = self.myconvolve3(vzt_avg, (window - 1) / 2)
                            vxp_avg[:] = self.myconvolve3(vxp_avg, (window - 1) / 2)

                DictA[
                    "vxt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vxt_avg
                DictA[
                    "vyt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vyt_avg
                DictA[
                    "vzt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vzt_avg

                DictA[
                    "vxp_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vxp_avg

                ################################################################################################################################################################
                ################################################################################################################################################################
                ################################################################################################################################################################
                ################################################################################################################################################################

                xd=ReferenceVector
                #xd=numpy.array([1,0,0])
                zup=numpy.array([0,0,1])
                theta1A=[]
                psi1A=[]
                for i in range(len(dtv)):
                    viii=numpy.array([vxt_avg[i],vyt_avg[i],vzt_avg[i]])
                    theta1, psi1 = dragonflyFieldOfVision(viii, xd, zup)# angle configuration of the paper
                    theta1A.append(theta1)
                    psi1A.append(psi1)


                DictA[
                    "thetaToR" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = theta1A
                DictA[
                    "psiToR" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = psi1A

                dtht=[]

                dtheta=[]
                if 1:  # better differentiation
                    for i in range(len(dtv) - 2):
                        if DifferentiationTresh==False:
                            dtheta.append((theta1A[i + 2] - theta1A[i]) / ((dtv[i + 2] - dtv[i])))

                            dtht.append(dt[i + 1])

                        else:
                            #print i, dt[i],dt[i + 2] - dt[i], dtv[i],dtv[i + 2] - dtv[i],dtTresh
                            if (dtv[i + 2] - dtv[i])<dtTresh:
                                dtheta.append((theta1A[i + 2] - theta1A[i]) / ((dtv[i + 2] - dtv[i])))

                                dtht.append(dt[i + 1])
                            else:
                                dtheta.append(numpy.nan)

                                dtht.append(dt[i + 1])

                DictA[
                    "dthetaToR" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = dtheta
                DictA[
                    "dtht" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = dtht
                ################################################################################################################################################################
                ################################################################################################################################################################
                ################################################################################################################################################################


                # magnitudes
                ###############################

                vavg = numpy.sqrt(vxt_avg ** 2 + vyt_avg ** 2 + vzt_avg ** 2)

                DictA[
                    "vavg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vavg

                vv = numpy.sqrt(vxt ** 2 + vyt ** 2 + vzt ** 2)
                DictA[
                    "vv" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vv


                vv_avg = numpy.zeros(len(vv))
                if ConvolveThing == 0:
                    vv_avg[:] = numpy.convolve(vv, avg_mask, 'same')  # running average

                elif ConvolveThing == 1:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vv_avg[:] = savgol_filter(vv, window, 1)

                        else:
                            vv_avg[:] = savgol_filter(vv_avg, window, 1)


                elif ConvolveThing == 2 or ConvolveThing == 3:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vv_avg[:] = self.myconvolve3(vv, (window - 1) / 2)

                        else:
                            vv_avg[:] = self.myconvolve3(vv_avg, (window - 1) / 2)

                DictA[
                    "vv_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vv_avg
                #############################


                ################################################################################################################################################################
                ################################################################################################################################################################

                # distance from start
                #############################
                xxt = []
                for i in range(len(xt_avg)):
                    xxt.append(numpy.sqrt((xt_avg[i] - xt_avg[0]) ** 2 + (yt_avg[i] - yt_avg[0]) ** 2 + (zt_avg[i] - zt_avg[0]) ** 2))
                xxt = numpy.array(xxt)
                DictA[
                    "xxt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = xxt

                ################################################################################################################################################################
                ################################################################################################################################################################

                # manually differentiating the xxt
                #############################
                vvt = []
                for i in range(len(xt_avg) - 2):
                    if DifferentiationTresh==False:
                        vvt.append((xxt[i + 2] - xxt[i]) / (dt[i + 2] - dt[i]))
                    else:
                        if (dt[i + 2] - dt[i]) < dtTresh:
                            vvt.append((xxt[i + 2] - xxt[i]) / (dt[i + 2] - dt[i]))
                        else:
                            vvt.append(numpy.nan)
                vvt = numpy.array(vvt)
                DictA[
                    "vvt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = vvt

                ################################################################################################################################################################
                ################################################################################################################################################################

                # path length
                #############################
                xpt = []
                xptint = 0
                for i in range(len(xt_avg)):
                    if i==0:
                        xpt.append(0)
                    else:
                        xptint += numpy.sqrt((xt_avg[i] - xt_avg[i-1]) ** 2 + (yt_avg[i] - yt_avg[i-1]) ** 2 + (zt_avg[i] - zt_avg[i-1]) ** 2)
                        xpt.append(xptint)
                xpt = numpy.array(xpt)
                DictA[
                    "xpt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        ii + 1)] = xpt


                ################################################################################################################################################################
                ################################################################################################################################################################
                #Path length velocity
                ###############################
                vpvt = []
                for i in range(len(xpt) - 2):
                    if 1:
                        if DifferentiationTresh == False:
                            vpvt.append((xpt[i + 2] - xpt[i]) / (dt[i + 2] - dt[i]))
                        else:
                            if (dt[i + 2] - dt[i]) < dtTresh:
                                vpvt.append((xpt[i + 2] - xpt[i]) / (dt[i + 2] - dt[i]))
                            else:
                                vpvt.append(numpy.nan)
                vpvt = numpy.array(vpvt)

                vpvt_avg = numpy.zeros(len(vpvt))

                if ConvolveThing == 0:
                    vpvt_avg[:] = numpy.convolve(vpvt, avg_mask, 'same')  # running average

                elif ConvolveThing == 1:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vpvt_avg[:] = savgol_filter(vpvt, window, 1)

                        else:
                            vpvt_avg[:] = savgol_filter(vpvt_avg, window, 1)


                elif ConvolveThing == 2 or ConvolveThing == 3:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            vpvt_avg[:] = self.myconvolve3(vpvt, (window - 1) / 2)

                        else:
                            vpvt_avg[:] = self.myconvolve3(vpvt_avg, (window - 1) / 2)


                DictA[
                    "vpvt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = vpvt_avg






                ################################################################################################################################################################
                ################################################################################################################################################################
                #acceleration
                #################################
                axt = []
                ayt = []
                azt = []
                axp = []

                delaxt = []
                delayt = []
                delazt = []
                delaxp = []

                dta = []

                if ConvolveThing == 2: # better differentiation
                    ################################qwq
                    aztspl = ztspl.derivative(2)
                    azt = aztspl(dt)
                    azt = numpy.delete(azt, -1, 0)
                    azt = numpy.delete(azt, 0, 0)
                    ################################ spline derivative

                    aytspl = ytspl.derivative(2)
                    ayt = aytspl(dt)
                    ayt = numpy.delete(ayt, -1, 0)
                    ayt = numpy.delete(ayt, 0, 0)
                    axtspl = xtspl.derivative(2)
                    axt = axtspl(dt)
                    axt = numpy.delete(axt, -1, 0)
                    axt = numpy.delete(axt, 0, 0)
                    axpspl = xpspl.derivative(2)
                    axp = axpspl(dt)
                    axp = numpy.delete(axp, -1, 0)
                    axp = numpy.delete(axp, 0, 0)
                    delaxt=numpy.copy(vxt)
                    delaxt[:]=numpy.nan
                    delayt = numpy.copy(delvxt)
                    delazt = numpy.copy(delvxt)
                    delaxp = numpy.copy(delvxt)
                    dta = numpy.copy(dt)
                    dta = numpy.delete(dta, -1, 0)
                    dta = numpy.delete(dta, 0, 0)


                else:
                    for i in range(len(xt_avg) - 2):
                        if DifferentiationTresh==False:
                            axt.append((xt_avg[i + 2] - 2 * xt_avg[i + 1] + xt_avg[i]) / ((dt[i + 1] - dt[i]) ** 2))
                            ayt.append((yt_avg[i + 2] - 2 * yt_avg[i + 1] + yt_avg[i]) / ((dt[i + 1] - dt[i]) ** 2))
                            azt.append((zt_avg[i + 2] - 2 * zt_avg[i + 1] + zt_avg[i]) / ((dt[i + 1] - dt[i]) ** 2))
                            axp.append((xp_avg[i + 2] - 2 * xp_avg[i + 1] + xp_avg[i]) / ((dt[i + 1] - dt[i]) ** 2))

                            delaxt.append((2 * delxt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                            delayt.append((2 * delyt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                            delazt.append((2 * delzt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                            delaxp.append((2 * delxp[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))

                            dta.append(dt[i + 1])  # ????
                        else:
                            dt1 = (dt[i + 2] - dt[i + 1])
                            dt2 = (dt[i + 1] - dt[i + 0])
                            if (dt1+dt2)<dtTresh:
                                axt.append((dt2*xt_avg[i + 2] - (dt1+dt2) * xt_avg[i + 1] + dt1*xt_avg[i]) / (dt1*dt2** 2))
                                ayt.append((dt2*yt_avg[i + 2] - (dt1+dt2) * yt_avg[i + 1] + dt1*yt_avg[i]) / (dt1*dt2 ** 2))
                                azt.append((dt2*zt_avg[i + 2] - (dt1+dt2) * zt_avg[i + 1] + dt1*zt_avg[i]) / (dt1*dt2 ** 2))
                                axp.append((dt2*xp_avg[i + 2] - (dt1+dt2) * xp_avg[i + 1] + dt1*xp_avg[i]) / (dt1*dt2 ** 2))

                                delaxt.append((2 * delxt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                                delayt.append((2 * delyt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                                delazt.append((2 * delzt[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))
                                delaxp.append((2 * delxp[i + 1]) / ((dt[i + 1] - dt[i]) ** 2))


                                dta.append(dt[i + 1])#????
                            else:
                                axt.append(numpy.nan)
                                ayt.append(numpy.nan)
                                azt.append(numpy.nan)
                                axp.append(numpy.nan)

                                delaxt.append(numpy.nan)
                                delayt.append(numpy.nan)
                                delazt.append(numpy.nan)
                                delaxp.append(numpy.nan)

                                dta.append(dt[i + 1])  # ????

                axt = numpy.array(axt)
                ayt = numpy.array(ayt)
                azt = numpy.array(azt)
                axp = numpy.array(axp)




                delaxt = numpy.array(delaxt)
                delayt = numpy.array(delayt)
                delazt = numpy.array(delazt)
                delaxp = numpy.array(delaxp)

                DictA[
                    "axt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = axt
                DictA[
                    "ayt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = ayt
                DictA[
                    "azt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = azt
                DictA[
                    "axp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = axp


                DictA[
                    "delaxt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = delaxt
                DictA[
                    "delayt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = delayt
                DictA[
                    "delazt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = delazt
                DictA[
                    "delaxp" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = delaxp



                ################################################################################################################################################################
                ################################################################################################################################################################

                #magnitudes
                ###############################

                aa = numpy.sqrt(axt ** 2 + ayt ** 2 + azt ** 2)
                DictA[
                    "aa" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = aa





                ################################################################################################################################################################


                ################################################################################################################################################################
                ################################################################################################################################################################

                azt_avg = numpy.zeros(len(azt))
                ayt_avg = numpy.zeros(len(azt))
                axt_avg = numpy.zeros(len(azt))
                axp_avg = numpy.zeros(len(axp))



                if ConvolveThing == 0:
                    azt_avg[:] = numpy.convolve(azt, avg_mask, 'same')
                    ayt_avg[:] = numpy.convolve(ayt, avg_mask, 'same')
                    axt_avg[:] = numpy.convolve(axt, avg_mask, 'same')
                    axp_avg[:] = numpy.convolve(axp, avg_mask, 'same')
                elif ConvolveThing == 1:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            axt_avg[:] = savgol_filter(axt, window, 1)
                            ayt_avg[:] = savgol_filter(ayt, window, 1)
                            azt_avg[:] = savgol_filter(azt, window, 1)
                            axp_avg[:] = savgol_filter(axp, window, 1)
                        else:
                            axt_avg[:] = savgol_filter(axt_avg, window, 1)
                            ayt_avg[:] = savgol_filter(ayt_avg, window, 1)
                            azt_avg[:] = savgol_filter(azt_avg, window, 1)
                            axp_avg[:] = savgol_filter(axp_avg, window, 1)


                elif ConvolveThing == 2  or ConvolveThing == 3:
                    for yi in range(ConvolveRepeat):
                        if yi == 0:
                            axt_avg[:] = self.myconvolve3(axt, (window - 1) / 2)
                            ayt_avg[:] = self.myconvolve3(ayt, (window - 1) / 2)
                            azt_avg[:] = self.myconvolve3(azt, (window - 1) / 2)
                            axp_avg[:] = self.myconvolve3(axp, (window - 1) / 2)
                        else:
                            axt_avg[:] = self.myconvolve3(axt_avg, (window - 1) / 2)
                            ayt_avg[:] = self.myconvolve3(ayt_avg, (window - 1) / 2)
                            azt_avg[:] = self.myconvolve3(azt_avg, (window - 1) / 2)
                            axp_avg[:] = self.myconvolve3(axp_avg, (window - 1) / 2)




                DictA["azt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = azt_avg
                DictA[
                    "ayt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = ayt_avg
                DictA[
                    "axt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = axt_avg
                DictA[
                    "axp_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = axp_avg

                DictErr["axt"] = "delaxt"
                DictErr["ayt"] = "delayt"
                DictErr["azt"] = "delazt"

                DictErr["axt_avg"] = "delaxt"
                DictErr["ayt_avg"] = "delayt"
                DictErr["azt_avg"] = "delazt"
                DictErr["axp"] = "delaxp"
                DictErr["axp_avg"] = "delaxp"

                ################################################################################################################################################################

                # averages across path
                ######################
                aavgfull = []
                aztfull = []
                aavedt = []
                for i in range(len(azt)):
                    # print azt[i]
                    # print i, aavg[i]
                    if i > len(azt) * .25 and i < len(azt) - len(azt) * .25:
                        aavgfull.append(aa[i])
                        aztfull.append(azt_avg[i])
                        aavedt.append(dta[i])
                aavgfull = numpy.array(aavgfull)
                aztfull = numpy.array(aztfull)
                aavgfullMean = numpy.mean(aavgfull)
                # aqwq
                #print "j,", j, ",average accell, aa,+-,azt,+-,", numpy.mean(aavgfull), ",", numpy.std(
                 #   aavgfull), ",", numpy.mean(aztfull), ",", numpy.std(aztfull)

                ################################################################################################################################################################
                ################################################################################################################################################################

                aavg = numpy.sqrt(axt_avg ** 2 + ayt_avg ** 2 + azt_avg ** 2)
                DictA[
                    "aavg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)] = aavg


                ################################################################################################################################################################
                ################################################################################################################################################################

                #curvature
                #################################
                curvature = []
                curveRadius = []
                for i in range(len(dta)):


                    if 1:
                        avect = numpy.array([axt_avg[i], ayt_avg[i], azt_avg[i]])
                        vvect = numpy.array([vxt_avg[i], vyt_avg[i], vzt_avg[i]])



                    curve = numpy.linalg.norm(numpy.cross(vvect, avect)) / ((numpy.linalg.norm(vvect)) ** 3)
                    curvature.append(curve)
                    curveRadius.append(1 / curve)

                DictA["curvature" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)] =curvature
                DictA["curveRadius" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j])+ "_" + str(ii+1)] = curveRadius






            if range(PlotDesciptiveDic["insectNumGroupNumber"]) > 1:
                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                    for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                        if ii != iii and ii < iii:
                            xb=DictA["xb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]
                            dt=DictA["dt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][j]) + "_" + str(ii+1)]
                            dt = numpy.array(dt)
                            xb = numpy.array(xb)




                            ####################################################################################
                            ####################################################################################
                            ####################################################################################
                            ####################################################################################

                            delxb = []


                            if 1:
                                nextTarget = 1
                                FirstendTarget = 0
                                BeginingTarget = 1
                                for i in range(len(xtRaw)):

                                    xbSrd, nextTarget1, FirstendTarget1, BeginingTarget1 = self.myconvolve3_justPointSpread(
                                        xb, (
                                                window - 1) / 2, i, nextTarget, FirstendTarget, BeginingTarget)


                                    nextTarget, FirstendTarget, BeginingTarget = nextTarget1, FirstendTarget1, BeginingTarget1


                                    delxb.append(numpy.std(xbSrd))


                            DictA["delxb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= delxb
                            delxbavg=numpy.nanmean(delxb)


                            ################################################################################################################################################################




                            xb_avg = numpy.zeros(len(xb))

                            if ConvolveThing == 0:
                                xb_avg[:] = numpy.convolve(xb, avg_mask, 'same')  # running average

                            elif ConvolveThing == 1:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        xb_avg[:] = savgol_filter(xb, window, 1)

                                    else:
                                        xb_avg[:] = savgol_filter(xb_avg, window, 1)


                            elif ConvolveThing == 3:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        xb_avg[:] = self.myconvolve3(xb, (window - 1) / 2)

                                    else:
                                        xb_avg[:] = self.myconvolve3(xb_avg, (window - 1) / 2)

                            elif ConvolveThing == 2:
                                xbCP = []
                                dtCPx = []

                                for sp in range(len(xb)):
                                    if numpy.isnan(xb[sp]) == False:
                                        dtCPx.append(dt[sp])
                                        xbCP.append(xb[sp])

                                dtCPx = numpy.array(dtCPx)
                                xbCP = numpy.array(xbCP)

                                xbspl = interpolate.UnivariateSpline(dtCPx, xbCP, k=xbsplk, s=bfactor*(len(xbCP)-1)*delxbavg**2)


                                xb_avg = xbspl(dt)


                                wwL = numpy.isnan(xb)


                                xb_avg[wwL] = numpy.nan





                            DictA["xb_avg" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= xb_avg


                            ################################################################################################################################################################

                            vxb = []
                            dtv = []

                            if ConvolveThing == 2:
                                vxbspl = xbspl.derivative(1)
                                vxb = vxbspl(dt)
                                vxb = numpy.delete(vxb, -1, 0)
                                vxb = numpy.delete(vxb, 0, 0)



                            else:# better differentiation
                                for i in range(len(xb) - 2):
                                    if DifferentiationTresh==False:
                                        vxb.append((xb_avg[i + 2] - xb_avg[i]) / ((dt[i + 2] - dt[i])))
                                    else:
                                        if (dt[i + 2] - dt[i]) < dtTresh:
                                            vxb.append((xb_avg[i + 2] - xb_avg[i]) / ((dt[i + 2] - dt[i])))
                                        else:
                                            vxb.append(numpy.nan)

                            vxb = numpy.array(vxb)

                            dtv = numpy.array(dtv)
                            DictA["vxb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= vxb




                            ################################################################################################################################################################
                            ################################################################################################################################################################

                            vxb_avg = numpy.zeros(len(vxb))



                            if ConvolveThing == 0:
                                vxb_avg[:] = numpy.convolve(vxb, avg_mask, 'same')  # running average

                            elif ConvolveThing == 1:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        vxb_avg[:] = savgol_filter(vxb, window, 1)

                                    else:
                                        vxb_avg[:] = savgol_filter(vxb_avg, window, 1)


                            elif ConvolveThing == 2 or ConvolveThing == 3:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        vxb_avg[:] = self.myconvolve3(vxb, (window - 1) / 2)

                                    else:
                                        vxb_avg[:] = self.myconvolve3(vxb_avg, (window - 1) / 2)

                            DictA["vxb_avg" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= vxb_avg



                            ################################################################################################################################################################
                            ################################################################################################################################################################

                            axb = []

                            if ConvolveThing == 2:
                                axbspl = xbspl.derivative(2)
                                axb = axbspl(dt)
                                axb = numpy.delete(axb, -1, 0)
                                axb = numpy.delete(axb, 0, 0)

                            else:  # better differentiation
                                for i in range(len(xb) - 2):
                                    if DifferentiationTresh==False:
                                        axb.append((xb_avg[i + 2] - 2 * xb_avg[i + 1] + xb_avg[i]) / ((dt[i + 1] - dt[i]) ** 2))
                                    else:
                                        dt1 = (dt[i + 2] - dt[i + 1])
                                        dt2 = (dt[i + 1] - dt[i + 0])
                                        if (dt1 + dt2) < dtTresh:
                                            axb.append((dt2 * xb_avg[i + 2] - (dt1 + dt2) * xb_avg[i + 1] + dt1 * xb_avg[i]) / (
                                                        dt1 * dt2 ** 2))
                                        else:
                                            axb.append(numpy.nan)

                            axb = numpy.array(axb)

                            DictA["axb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= axb





                            ################################################################################################################################################################
                            ################################################################################################################################################################

                            axb_avg = numpy.zeros(len(axb))

                            if ConvolveThing == 0:
                                axb_avg[:] = numpy.convolve(axb, avg_mask, 'same')  # running average

                            elif ConvolveThing == 1:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        axb_avg[:] = savgol_filter(axb, window, 1)
                                    else:
                                        axb_avg[:] = savgol_filter(axb_avg, window, 1)

                            elif ConvolveThing == 2 or ConvolveThing == 3:
                                for yi in range(ConvolveRepeat):
                                    if yi == 0:
                                        axb_avg[:] = self.myconvolve3(axb, (window - 1) / 2)
                                    else:
                                        axb_avg[:] = self.myconvolve3(axb_avg, (window - 1) / 2)

                            DictA["axb_avg" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]= axb_avg



                            ################################################################################################################################################################
                            ################################################################################################################################################################

                            if 1:#Angle of eachother.
                                xtii=DictA["xt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)]
                                ytii=DictA["yt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)]
                                ztii=DictA["zt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(ii + 1)]

                                xtiii=DictA["xt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(iii + 1)]
                                ytiii=DictA["yt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(iii + 1)]
                                ztiii=DictA["zt_avg" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(iii + 1)]


                                vxt_avgii=DictA[
                                    "vxt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                        ii + 1)]
                                vyt_avgii=DictA[
                                    "vyt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                        ii + 1)]
                                vzt_avgii=DictA[
                                    "vzt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                        ii + 1)]

                                vxt_avgiii=DictA[
                                    "vxt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(
                                        iii + 1)]
                                vyt_avgiii=DictA[
                                    "vyt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(
                                        iii + 1)]
                                vzt_avgiii=DictA[
                                    "vzt_avg" + str(PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(
                                        iii + 1)]
                                zup = numpy.array([0, 0, 1])

                                thetaToRii=DictA[
                                    "thetaToR" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                        ii + 1)]
                                thetaToRiii=DictA[
                                    "thetaToR" + str(PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + "_" + str(
                                        iii + 1)]

                                dtv=DictA[
                                    "dtv" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                        ii + 1)]

                                phi1A=[]
                                phi2A=[]
                                psi1A=[]
                                psi2A=[]
                                phi1OA=[]
                                phi2OA=[]
                                phi1EA=[]
                                phi2EA=[]
                                alpha1A=[]
                                alpha2A=[]

                                for ti in range(len(vxt_avgii)):
                                    xd=numpy.array([xtii[ti+1]-xtiii[ti+1],ytii[ti+1]-ytiii[ti+1],ztii[ti+1]-ztiii[ti+1]])
                                    vii=numpy.array([vxt_avgii[ti],vyt_avgii[ti],vzt_avgii[ti]])
                                    viii=numpy.array([vxt_avgiii[ti],vyt_avgiii[ti],vzt_avgiii[ti]])

                                    #need to know the direction of the xd vector
                                    phi1, psi1 = dragonflyFieldOfVision(vii, -xd, zup)
                                    phi2, psi2 = dragonflyFieldOfVision(viii, xd, zup)

                                    beta1=numpy.arccos(numpy.dot(-xd,viii)/(numpy.linalg.norm(-xd)*numpy.linalg.norm(viii)))
                                    phi1O=numpy.arcsin(numpy.linalg.norm(viii)*numpy.sin(beta1)/numpy.linalg.norm(vii))*180/3.1415926

                                    beta2=numpy.arccos(numpy.dot(xd,vii)/(numpy.linalg.norm(xd)*numpy.linalg.norm(vii)))
                                    phi2O=numpy.arcsin(numpy.linalg.norm(vii)*numpy.sin(beta2)/numpy.linalg.norm(viii))*180/3.1415926

                                    phi1E=phi1-phi1O
                                    phi2E=phi2-phi2O

                                    alpha1=phi1+thetaToRii[ti]
                                    alpha2 = phi2 + thetaToRiii[ti]

                                    phi1A.append(phi1)
                                    phi2A.append(phi2)
                                    psi1A.append(psi1)
                                    psi2A.append(psi2)
                                    phi1OA.append(phi1O)
                                    phi2OA.append(phi2O)
                                    phi1EA.append(phi1E)
                                    phi2EA.append(phi2E)
                                    alpha1A.append(alpha1)
                                    alpha2A.append(alpha2)


                                DictA["phi1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi1A

                                DictA["phi2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi2A

                                DictA["psi1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=psi1A

                                DictA["psi2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=psi2A

                                DictA["phiO1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi1OA

                                DictA["phiO2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi2OA



                                DictA["phiE1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi1EA

                                DictA["phiE2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=phi2EA


                                DictA["alpha1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=alpha1A

                                DictA["alpha2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=alpha2A



                                dphi1A=[]
                                dphi2A=[]
                                dpsi1A=[]
                                dpsi2A=[]
                                dphi1OA=[]
                                dphi2OA=[]
                                dphi1EA=[]
                                dphi2EA=[]
                                dalpha1A=[]
                                dalpha2A=[]

                                if 1:  # better differentiation
                                    for i in range(len(dtv) - 2):
                                        if DifferentiationTresh == False:
                                            dphi1A.append((phi1A[i + 2] - phi1A[i]) / ((dtv[i + 2] - dtv[i])))
                                            dphi2A.append((phi2A[i + 2] - phi2A[i]) / ((dtv[i + 2] - dtv[i])))
                                            dpsi1A.append((psi1A[i + 2] - psi1A[i]) / ((dtv[i + 2] - dtv[i])))
                                            dpsi2A.append((psi2A[i + 2] - psi2A[i]) / ((dtv[i + 2] - dtv[i])))
                                            dphi1OA.append((phi1OA[i + 2] - phi1OA[i]) / ((dtv[i + 2] - dtv[i])))
                                            dphi2OA.append((phi2OA[i + 2] - phi2OA[i]) / ((dtv[i + 2] - dtv[i])))
                                            dphi1EA.append((phi1EA[i + 2] - phi1EA[i]) / ((dtv[i + 2] - dtv[i])))
                                            dphi2EA.append((phi2EA[i + 2] - phi2EA[i]) / ((dtv[i + 2] - dtv[i])))
                                            dalpha1A.append((alpha1A[i + 2] - alpha1A[i]) / ((dtv[i + 2] - dtv[i])))
                                            dalpha2A.append((alpha2A[i + 2] - alpha2A[i]) / ((dtv[i + 2] - dtv[i])))



                                        else:
                                            # print i, dt[i],dt[i + 2] - dt[i],dtTresh
                                            if (dtv[i + 2] - dtv[i]) < dtTresh:
                                                dphi1A.append((phi1A[i + 2] - phi1A[i]) / ((dtv[i + 2] - dtv[i])))
                                                dphi2A.append((phi2A[i + 2] - phi2A[i]) / ((dtv[i + 2] - dtv[i])))
                                                dpsi1A.append((psi1A[i + 2] - psi1A[i]) / ((dtv[i + 2] - dtv[i])))
                                                dpsi2A.append((psi2A[i + 2] - psi2A[i]) / ((dtv[i + 2] - dtv[i])))
                                                dphi1OA.append((phi1OA[i + 2] - phi1OA[i]) / ((dtv[i + 2] - dtv[i])))
                                                dphi2OA.append((phi2OA[i + 2] - phi2OA[i]) / ((dtv[i + 2] - dtv[i])))
                                                dphi1EA.append((phi1EA[i + 2] - phi1EA[i]) / ((dtv[i + 2] - dtv[i])))
                                                dphi2EA.append((phi2EA[i + 2] - phi2EA[i]) / ((dtv[i + 2] - dtv[i])))
                                                dalpha1A.append((alpha1A[i + 2] - alpha1A[i]) / ((dtv[i + 2] - dtv[i])))
                                                dalpha2A.append((alpha2A[i + 2] - alpha2A[i]) / ((dtv[i + 2] - dtv[i])))
                                            else:
                                                dphi1A.append(numpy.nan)
                                                dphi2A.append(numpy.nan)
                                                dpsi1A.append(numpy.nan)
                                                dpsi2A.append(numpy.nan)
                                                dphi1OA.append(numpy.nan)
                                                dphi2OA.append(numpy.nan)
                                                dphi1EA.append(numpy.nan)
                                                dphi2EA.append(numpy.nan)
                                                dalpha1A.append(numpy.nan)
                                                dalpha2A.append(numpy.nan)




                                DictA["dphi1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi1A

                                DictA["dphi2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi2A

                                DictA["dpsi1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dpsi1A

                                DictA["dpsi2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dpsi2A

                                DictA["dphiO1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi1OA

                                DictA["dphiO2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi2OA



                                DictA["dphiE1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi1EA

                                DictA["dphiE2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dphi2EA


                                DictA["dalpha1to2" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dalpha1A

                                DictA["dalpha2to1" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]=dalpha2A


        return DictA, DictErr

        ######################################################################################################
        ######################################################################################################


        # path analysis--- creating the data dictionary

    def createPathDic2(self,PlotDesciptiveDic , ResultRot, ResultTran, fwData):
        # gathering info
        #PlotDesciptiveDic is the info inputed into the excel file
        cameraNames = ["camera1", "camera2", "combined"]
        DictA = {}
        stumpKnob = PlotDesciptiveDic["PointofReference"]
        #stumpKnob = numpy.array([-86.41750386, -151.3175956, 14.40064474])

        #making a dictionary of all h5's used.---- including other projects with same map
        DictH5={}

        for i in range(len(PlotDesciptiveDic["OtherProjects"])):
            if i==0:
                #DictH5[PlotDesciptiveDic["OtherProjects"][i]]=PlotDesciptiveDic["OtherProjectsPaths"][i]
                if PlotDesciptiveDic["OtherProjectsPaths"][i]=="":
                    DictH5[PlotDesciptiveDic["OtherProjects"][i]]=fwData
                else:

                    DictH5[PlotDesciptiveDic["OtherProjects"][i]] = h5py.File(PlotDesciptiveDic["OtherProjectsPaths"][i],"r").get('data')
                #fw = h5py.File(h5filenamewrite, 'a')

                #fwData = fw.get('data')

            else:
                if PlotDesciptiveDic["OtherProjects"][i] not in DictH5.keys():
                    if PlotDesciptiveDic["OtherProjectsPaths"][i] == "":
                        DictH5[PlotDesciptiveDic["OtherProjects"][i]] = fwData
                    else:

                        DictH5[PlotDesciptiveDic["OtherProjects"][i]] = h5py.File(
                            PlotDesciptiveDic["OtherProjectsPaths"][i], "r").get('data')

                    #DictH5[PlotDesciptiveDic["OtherProjects"][i]] = PlotDesciptiveDic["OtherProjectsPaths"][i]


        for i in range(len(PlotDesciptiveDic["Insectnumbers1"])):# number of insects to include on the graph
            ###################################
            # Initializing the dictionary data
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):# number of simultanious insects.
                DictA["xt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]) + "_" + str(ii+1)] = []
                DictA["yt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i])+ "_" + str(ii+1)] = []
                DictA["zt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]) + "_" + str(ii+1)] = []
                DictA["dt" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]) + "_" + str(ii+1)] = []
                DictA["dtFirst" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]) + "_" + str(ii+1)] = []
                DictA["xp" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i])+ "_" + str(ii+1)] = []

            # Initializing shared data
            if range(PlotDesciptiveDic["insectNumGroupNumber"]) > 1:
                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                    for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                        if ii!=iii and ii<iii:
                            DictA["xb"+str(ii+1)+"-"+str(iii+1)+"__" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i])+"-"+str(PlotDesciptiveDic["Insectnumbers"+str(iii+1)][i])] = []

            #this is the fixed point.
            DictA["stumpKnob"] = stumpKnob

            #loading times.
            if PlotDesciptiveDic["StartFrame"][i]=="":
                Startframe=0
                Endframe=100000000
            else:
                print PlotDesciptiveDic["StartFrame"][i]
                Startframe=int(PlotDesciptiveDic["StartFrame"][i])
                Endframe=int(PlotDesciptiveDic["EndFrame"][i])
            inKeyName=[]
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                #inKeyName.append("insect" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]) + "_" + "insect" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i]))
                inKeyName.append("insect" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i].split("-")[0]) + "_" + "insect" + str(
                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i].split("-")[0]))

            #print inKeyName
            didwestart = 0

            fwDataKeysInt=[]
            for s in DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[0]].keys():
                if s != "FrameDelay":
                    fwDataKeysInt.append(int(s))
            fwDataKeysmin = min(fwDataKeysInt)
            fwDataKeysmax = max(fwDataKeysInt)
            #print fwDataKeysmin,fwDataKeysmax,"fwDataKeysmin,fwDataKeysmax"
            #for jj in fwData[cameraNames[2]][inKeyName[0]]:
            #print DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[0]].keys()
            #for jj in DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[0]]:
            for jj in range(fwDataKeysmin,fwDataKeysmax):
                #if jj != "FrameDelay":
                if 1:

                    if int(jj)>Startframe and int(jj)<Endframe:
                    #if 1:
                        usePoint=0
                        xKalA = [[]] * ((PlotDesciptiveDic["insectNumGroupNumber"]))
                        xKalARR = [[]] * ((PlotDesciptiveDic["insectNumGroupNumber"]))
                        try:

                            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):


                                InterPointRawRR = numpy.array(
                                    DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[ii]][str(jj)][
                                        '3Dpoint'])

                                xKalRR = numpy.array([InterPointRawRR[0], InterPointRawRR[1], InterPointRawRR[2]])

                                xKalARR[ii] = numpy.matmul(ResultRot.T, xKalRR) + ResultTran
                                if didwestart==0:
                                    C1dat = numpy.array(
                                        DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[ii]][
                                            str(jj)][
                                            'cameraData1'])
                                    C1x=numpy.array([C1dat[0],C1dat[1],C1dat[2]])
                                    C1y=numpy.array([C1dat[3],C1dat[4],C1dat[5]])
                                    C1z=numpy.array([C1dat[6],C1dat[7],C1dat[8]])
                                    C1O=numpy.array([C1dat[9],C1dat[10],C1dat[11]])
                                    C1x=numpy.matmul(ResultRot.T, C1x)
                                    C1y =numpy.matmul(ResultRot.T, C1y)
                                    C1z =numpy.matmul(ResultRot.T, C1z)
                                    C1x = C1x / numpy.linalg.norm(C1x)
                                    C1y = C1y / numpy.linalg.norm(C1y)
                                    C1z = C1z / numpy.linalg.norm(C1z)
                                    C1O =numpy.matmul(ResultRot.T, C1O) + ResultTran
                                    C2dat = numpy.array(
                                            DictH5[PlotDesciptiveDic["OtherProjects"][i]][cameraNames[2]][inKeyName[ii]][
                                                str(jj)][
                                                'cameraData2'])
                                    C2x=numpy.array([C2dat[0],C2dat[1],C2dat[2]])
                                    C2y=numpy.array([C2dat[3],C2dat[4],C2dat[5]])
                                    C2z=numpy.array([C2dat[6],C2dat[7],C2dat[8]])
                                    C2O=numpy.array([C2dat[9],C2dat[10],C2dat[11]])
                                    C2x=numpy.matmul(ResultRot.T, C2x)
                                    C2y =numpy.matmul(ResultRot.T, C2y)
                                    C2z =numpy.matmul(ResultRot.T, C2z)
                                    C2x=C2x/numpy.linalg.norm(C2x)
                                    C2y = C2y / numpy.linalg.norm(C2y)
                                    C2z = C2z / numpy.linalg.norm(C2z)
                                    C2O =numpy.matmul(ResultRot.T, C2O) + ResultTran


                            usePoint=1
                            #print jj,xKalRR[0],"here in the creation"
                        except:
                            None

                        if usePoint==1:
                            #print xKalA
                            if didwestart == 0:
                                firststart = jj
                                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                    DictA["dtFirst" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)]=firststart
                                    DictA["C1x" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C1x
                                    DictA["C1y" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C1y
                                    DictA["C1z" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C1z
                                    DictA["C1O" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C1O
                                    DictA["C2x" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C2x
                                    DictA["C2y" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C2y
                                    DictA["C2z" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C2z
                                    DictA["C2O" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                        ii + 1)] = C2O

                                didwestart = 1
                                #                        dt.append((float(jj)-float(firststart)) / 240.0)

                            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                DictA["dt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append((float(int(jj))) / 240.0)



                                DictA["xt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(xKalARR[ii][0])
                                DictA["yt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(xKalARR[ii][1])
                                DictA["zt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(xKalARR[ii][2])


                                #DictA["xp" + str(PlotDesciptiveDic["Insectnumbers"+str(ii+1)][i])+ "_" + str(ii+1)].append(
                                #    numpy.sqrt((xKalA[ii][0] - stumpKnob[0]) ** 2 +
                                #               (xKalA[ii][1] - stumpKnob[1]) ** 2 +
                                #               (xKalA[ii][2] - stumpKnob[2]) ** 2))



                                ###################################################


                            if range(PlotDesciptiveDic["insectNumGroupNumber"])>1:
                                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                    for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                        if ii != iii and ii < iii:
                                            DictA["xb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "-" + str(
                                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][i])].append(
                                                numpy.sqrt((xKalARR[ii][0] - xKalARR[iii][0]) ** 2 +
                                                           (xKalARR[ii][1] - xKalARR[iii][1]) ** 2 +
                                                           (xKalARR[ii][2] - xKalARR[iii][2]) ** 2))
                        else:

                            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                #DictA["dt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                #    ii + 1)].append((float(int(jj))) / 240.0)
                                DictA["dt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(numpy.nan)

                                DictA["xt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(numpy.nan)
                                DictA["yt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(numpy.nan)
                                DictA["zt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                    ii + 1)].append(numpy.nan)



                            if range(PlotDesciptiveDic["insectNumGroupNumber"]) > 1:
                                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                    for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                                        if ii != iii and ii < iii:
                                            DictA["xb" + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "-" + str(
                                                PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][i])].append(numpy.nan)





        del DictH5
        return DictA


    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################








    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################


    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################






    def outputPathsinCSVMajor(self, PlotDesciptiveDic,KinPlotDesciptiveDic,DictA,book):

        sheet4 = book.add_sheet('CVS output')


        PathWrite = open(PlotDesciptiveDic["Path2SaveImages"]+"/"+ PlotDesciptiveDic["SaveFileName"]+"_CSV.txt", "w")

        ParamNamesXsSing=["xt",	"yt",	"zt","xt_avg",	"yt_avg",	"zt_avg","xxt","xpt", "xp","wd","xp_avg","wd_avg" ,"disSTL","disSTL_avg","disSTLz","disSTLz_avg","PCAsigma","PCAmu","PCAwt"]
        ParamNamesVsSing=["vxt",	"vyt",	"vzt",	"vxt_avg",	"vyt_avg",	"vzt_avg",	"vavg",	"vv",	"vv_avg","vvt","vpvt_avg",
                      "axt",	"ayt",	"azt",	"aavg",	"axt_avg",	"ayt_avg",	"azt_avg","curvature",	"curveRadius",
                          "vxp","vxp_avg","axp","axp_avg","thetaToR","psiToR","aa"]
        ParamNamesXsDouble=["xb","xb_avg"]
        ParamNamesVsDouble=["vxb","vxb_avg","axb","axb_avg","phi1to2","psi1to2","phi2to1","psi2to1","phiO1to2",
                            "phiO2to1",
                            "phi1to2",
                            "phi2to1",
                            "phiE1to2",
                            "phiE2to1",
                            "alpha1to2",
                            "alpha2to1"
                            ]
        ParamNamesDthetasDouble=["dphi1to2",
                                "dphi2to1",
                                "dpsi1to2",
                                "dpsi2to1",
                                "dphiO1to2",
                                "dphiO2to1",
                                "dphiE1to2",
                                "dphiE2to1",
                                "dalpha1to2",
                                "dalpha2to1"]
        #ParamNamesVsDouble=[]
        if 0:
            if 0:
                PathWrite = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/trespass" + str(i) + ".txt"
                PathWrite = open(PathWrite, "w")
            print i
            j = Insectnumbers1[i]
            # print DictA["inXk" + str(j) + "1"]
            for ii in range(len(DictA["inXk" + str(j) + "_" + "1"])):
                PathWrite.write(
                    str(DictA["inXk" + str(j) + "_" + "1"][ii]) + " " + str(
                        DictA["inYk" + str(j) + "_" + "1"][ii]) + " " + str(
                        DictA["inZk" + str(j) + "_" + "1"][ii]) + " " + str(
                        colorarray[i][2]) + " " + str(colorarray[i][1]) + " " + str(
                        colorarray[i][0]) + "\n")


        lineTrack=0
        for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):
            groupnums=""
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                if ii==0:
                    groupnums+=PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]
                else:
                    groupnums += "-"+PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]
            stringHeader0 ="Insect numbers for this group are "+groupnums+";"

            PathWrite.write("\n\n"+stringHeader0 + "\n")
            lineTrack+=1
            for collit in range(len(stringHeader0.split(";"))):
                sheet4.write(lineTrack, collit,
                            stringHeader0.split(";")[collit])
            lineTrack += 2
            stringHeader="dt;"
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                for ju in range(len(ParamNamesXsSing)):
                    stringHeader+=ParamNamesXsSing[ju]+"_"+str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j])+";"
                for ju in range(len(ParamNamesVsSing)):
                    stringHeader += ParamNamesVsSing[ju]+"_" + str(
                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + ";"
            if range(PlotDesciptiveDic["insectNumGroupNumber"]) > 1:
                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                    for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                        if ii != iii and ii < iii:
                            for ju in range(len(ParamNamesXsDouble)):
                                stringHeader += ParamNamesXsDouble[ju] +  "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])+";"
                            for ju in range(len(ParamNamesVsDouble)):
                                stringHeader += ParamNamesVsDouble[ju]  + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + ";"
                            for ju in range(len(ParamNamesDthetasDouble)):
                                stringHeader += ParamNamesDthetasDouble[ju] + "__" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j]) + ";"

            PathWrite.write(stringHeader+"\n")

            for collit in range(len(stringHeader.split(";"))):
                sheet4.write(lineTrack, collit,
                            stringHeader.split(";")[collit])
            lineTrack+=1


            ###############################################################################################3


            dt = DictA[
                "dt" + str(PlotDesciptiveDic["Insectnumbers" + str(0 + 1)][j]) + "_" + str(
                    0 + 1)]
            print len(dt),"len(dt)"
            for ku in range(len(dt)):
                stringstuff=str(dt[ku])+";"
                for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                    for ju in range(len(ParamNamesXsSing)):
                        ###############################################################################################3

                        #print len(DictA[
                        #    ParamNamesXsSing[ju] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                        #        ii + 1)]), ParamNamesXsSing[ju],len(dt),ku
                        stringstuff +=str(DictA[
                            ParamNamesXsSing[ju] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][ku])+";"
                    ###############################################################################################3
                    if ku!=0 and ku<len(dt)-2:
                        for ju in range(len(ParamNamesVsSing)):

                            stringstuff +=str(DictA[
                            ParamNamesVsSing[ju] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][ku])+";"
                    else:
                        for ju in range(len(ParamNamesVsSing)):
                            stringstuff +=""+";"
                if range(PlotDesciptiveDic["insectNumGroupNumber"]) > 1:
                    for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                        for iii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                            if ii != iii and ii < iii:
                                ###############################################################################################3
                                for ju in range(len(ParamNamesXsDouble)):
                                    stringstuff += str(DictA[ParamNamesXsDouble[ju] + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                        PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])][ku]) + ";"
                                ###############################################################################################3


                                if ku != 0 and ku < len(dt)-2:
                                    for ju in range(len(ParamNamesVsDouble)):
                                        #print ParamNamesVsDouble[ju],"DictA[ParamNamesVsDouble[ju] +",ku
                                        #print len((DictA[ParamNamesVsDouble[ju] + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                        #    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                        #    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])]))

                                        stringstuff += str(DictA[ParamNamesVsDouble[ju] + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                            PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                            PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])][ku]) + ";"
                                ###############################################################################################3
                                else:
                                    for ju in range(len(ParamNamesVsDouble)):
                                        stringstuff += "" + ";"
                                ###############################################################################################3
                                if ku >1 and ku < len(dt)-4:#### cant be 5
                                    for ju in range(len(ParamNamesDthetasDouble)):
                                        #print ParamNamesDthetasDouble[ju],"DictA[ParamNamesVsDouble[ju] +", ku
                                        #print len(DictA[ParamNamesDthetasDouble[ju] + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                        #    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                        #    PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])])
                                        stringstuff += str(DictA[ParamNamesDthetasDouble[ju] + str(ii + 1) + "-" + str(iii + 1) + "__" + str(
                                            PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "-" + str(
                                            PlotDesciptiveDic["Insectnumbers" + str(iii + 1)][j])][ku]) + ";"
                                ###############################################################################################3
                                else:
                                    for ju in range(len(ParamNamesDthetasDouble)):
                                        stringstuff += "" + ";"

                PathWrite.write(stringstuff + "\n")
                for collit in range(len(stringstuff.split(";"))):
                    if stringstuff.split(";")[collit]=="nan" or len(stringstuff.split(";")[collit].split("nan"))>1:
                        sheet4.write(lineTrack, collit,
                                     "")
                    else:
                        sheet4.write(lineTrack, collit,
                                     (stringstuff.split(";")[collit]))
                lineTrack+=1



    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################



    def MakeVideoFromGoPro6Video(self,capture1):
        GoProNum = "010322"

        GoProName = "Sam"
        vidfile = "F:/GoPros/GoPro" + GoProName + "/GX" + GoProNum + ".mp4"
        saveVidFile = "F:/GoPros/GoPro" + GoProName + "RD/GP" + GoProNum + "R.mp4"

        Make720 = True

        #capture1 = cv2.VideoCapture(vidfile)


        capture1.set(cv2.CAP_PROP_POS_FRAMES, 300)

        if 1:
            ret, frame1 = capture1.read()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Mike = cv2.imread(beginDirect +"/" + file_list[0])
            h, w, th = numpy.shape(frame1)
            print w, h
            #            video_writer = cv2.VideoWriter(saveVidFile + "/GP1.avi", fourcc, 30, (w, h))
            if Make720 == True:
                print ""
                h = 720
                w = int(w * 720.0 / 1080.0)
            video_writer = cv2.VideoWriter(saveVidFile, fourcc, 60, (w, h))

            # cv2.VideoWriter.open(str(self.SaveFile) + "\Mike_"+str(fileInc),CV_FOURCC=PIM1,fps=5)

            # for i in range(60*240):
            iui = 0
            while frame1 is not None:
                # while iui < 200:
                ret, frame1 = capture1.read()
                if frame1 is not None:
                    if Make720 == True:
                        print ""
                        # as_array = numpy.asarray(frame[:, :])

                        frame1 = cv2.resize(frame1, (w, h))

                    video_writer.write(frame1)
                else:
                    None
                iui += 1
                print iui
            video_writer.release()






    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def FillMapDictionaries(self,fwData, cameraNames, f1, f2, frame1num, h5SquidDelay, frame2num, dat2,
                             cameraMatrix2, distCoeffs2, dat1, cameraMatrix1, distCoeffs1, MapErrorDicMeauredPoint,
                             MapErrorDicMapPoint,CameraOriginArray):
        try:
            try:
                ResultRot = numpy.array(
                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])
                ResultRot = ResultRot.T

                ResultTran = numpy.array(
                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

            except:
                ResultRot = numpy.array(
                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])  # l;l

                ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])
        except:
            None

        try:
            Mapdat1 = numpy.array(f1['F' + str(int(frame1num - h5SquidDelay))]['MapData'][:])

            Mapdat2 = numpy.array(f2['F' + str(int(frame2num - h5SquidDelay))]['MapData'][:])
            Mapdat1set = set()
            Mapdat2set = set()
            Mapdat1Dic = {}
            Mapdat2Dic = {}
            for fd in range(len(Mapdat1)):
                Mapdat1set.add(int(Mapdat1[fd][3]))
                Mapdat1Dic[int(Mapdat1[fd][3])] = [Mapdat1[fd][5], Mapdat1[fd][6], Mapdat1[fd][7], Mapdat1[fd][8],
                                                   Mapdat1[fd][9]]
            for fd in range(len(Mapdat2)):
                Mapdat2set.add(int(Mapdat2[fd][3]))
                Mapdat2Dic[int(Mapdat2[fd][3])] = [Mapdat2[fd][5], Mapdat2[fd][6], Mapdat2[fd][7],
                                                   Mapdat2[fd][8], Mapdat2[fd][9]]
            # print Mapdat2[:][0]
            twoCameraIntersection = Mapdat1set.intersection(Mapdat2set)
            print len(twoCameraIntersection), len(Mapdat1set), len(Mapdat2set)
            for fd in twoCameraIntersection:
                point1 = numpy.array([Mapdat1Dic[fd][3], Mapdat1Dic[fd][4]])
                point2 = numpy.array([Mapdat2Dic[fd][3], Mapdat2Dic[fd][4]])
                CamOrgn1, pointvect1 = self.theVectors(dat1, point1, cameraMatrix1, distCoeffs1)
                CamOrgn2, pointvect2 = self.theVectors(dat2, point2, cameraMatrix2, distCoeffs2)

                cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1,
                                                                          CamOrgn2,
                                                                          pointvect2)

                cloPoAve = numpy.array([(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                        (cloPo1[2] + cloPo2[2]) / 2])
                cloPoAveTran = numpy.matmul(ResultRot.T, cloPoAve) + ResultTran
                CameraOriginArray.append([numpy.matmul(ResultRot.T, CamOrgn1) + ResultTran,numpy.matmul(ResultRot.T, CamOrgn2) + ResultTran])
                try:
                    MapErrorDicMeauredPoint[fd]
                    MapErrorDicMapPoint[fd]
                except:
                    MapErrorDicMeauredPoint[fd] = []
                    MapPointfmd = numpy.array([Mapdat1Dic[fd][0], Mapdat1Dic[fd][1], Mapdat1Dic[fd][2]])
                    MapErrorDicMapPoint[fd] = numpy.matmul(ResultRot.T, MapPointfmd) + ResultTran
                MapErrorDicMeauredPoint[fd].append(cloPoAveTran)
                MapPointfmd1 = numpy.array([Mapdat1Dic[fd][0], Mapdat1Dic[fd][1], Mapdat1Dic[fd][2]])
                MapPointfmd2 = numpy.array([Mapdat2Dic[fd][0], Mapdat2Dic[fd][1], Mapdat2Dic[fd][2]])
                #print numpy.matmul(ResultRot.T, MapPointfmd1) + ResultTran,numpy.matmul(ResultRot.T, MapPointfmd2) + ResultTran,cloPoAveTran




        except:
            print "could not get map data"
        return MapErrorDicMeauredPoint,MapErrorDicMapPoint,CameraOriginArray

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################



    def analyizeMapErrorDics(self,CameraOriginArray,MapErrorDicMeauredPoint,MapErrorDicMapPoint):
        CameraAverage = []
        CameraDistance = []
        for fd in range(len(CameraOriginArray)):
            CameraAverage.append((CameraOriginArray[fd][0] + CameraOriginArray[fd][1]) / 2)
            CameraDistance.append(numpy.linalg.norm(CameraOriginArray[fd][0] - CameraOriginArray[fd][1]))
        CameraAverage = numpy.array(CameraAverage)
        CameraDistance = numpy.array(CameraDistance)
        print "CameraAverage", numpy.mean(CameraAverage, axis=0)
        print "CameraDistance", numpy.mean(CameraDistance, axis=0)
        otherbook=Workbook()
        sheeto = otherbook.add_sheet('CVS output')
        zhat=numpy.array([0,0,1])
        distance2point=[]
        vertError=[]
        longError=[]
        Dxmin=80
        Dxmax=1000#streem
        #Dxmax=300#coral
        rowlit=0
        StringDescrip=""
        sheeto.write(0, 0,
                     "Distance")
        sheeto.write(0, 1,
                     "Vertical error vs. Distance"+StringDescrip)
        sheeto.write(0, 2,
                     "Axial Error vs. Distance"+StringDescrip)
        sheeto.write(0, 3,
                     "Camera inter distance")
        sheeto.write(1, 3,
                      numpy.mean(CameraDistance, axis=0))
        for fd in range(len(MapErrorDicMeauredPoint.keys())):
            if len(MapErrorDicMeauredPoint[MapErrorDicMeauredPoint.keys()[fd]]) > 50:
                MapAvg = numpy.array(MapErrorDicMeauredPoint[MapErrorDicMeauredPoint.keys()[fd]])
                Vect2Point=(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0))/numpy.linalg.norm(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0))
                Vect2Pointz=numpy.cross(numpy.cross(Vect2Point,zhat),Vect2Point)
                Vect2Pointz=Vect2Pointz/numpy.linalg.norm(Vect2Pointz)
                if numpy.linalg.norm(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0))>Dxmin and numpy.linalg.norm(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0))<Dxmax:
                    distance2point.append(numpy.linalg.norm(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0)))
                    vertError.append(numpy.std(numpy.abs(numpy.matmul(MapAvg,Vect2Pointz)), axis=0))
                    longError.append(numpy.std(numpy.abs(numpy.matmul(MapAvg, Vect2Point)), axis=0))
                    sheeto.write(rowlit + 3, 0,
                                 numpy.linalg.norm(numpy.mean(MapAvg, axis=0)-numpy.mean(CameraAverage, axis=0)))
                    sheeto.write(rowlit + 3, 1,
                                 numpy.std(numpy.abs(numpy.matmul(MapAvg, Vect2Pointz)), axis=0))
                    sheeto.write(rowlit + 3, 2,
                                 numpy.std(numpy.abs(numpy.matmul(MapAvg, Vect2Point)), axis=0))
                    rowlit+=1



                print MapErrorDicMapPoint[MapErrorDicMeauredPoint.keys()[fd]], \
                    len(MapErrorDicMeauredPoint[MapErrorDicMeauredPoint.keys()[fd]]), \
                    numpy.mean(MapAvg, axis=0), numpy.std(MapAvg, axis=0), numpy.linalg.norm(
                    numpy.mean(CameraAverage, axis=0) - numpy.mean(MapAvg, axis=0)),numpy.std(numpy.abs(numpy.matmul(MapAvg,zhat)), axis=0), \
                    numpy.std(numpy.abs(numpy.matmul(MapAvg, Vect2Point)), axis=0)




        fig, ax = plt.subplots(2,figsize=(6,7))

        ax[0].scatter(distance2point,vertError)
        ax[1].scatter(distance2point, longError)



        z = numpy.polyfit(distance2point, vertError, 1, full=False, cov=True)
        print z,"vertError"
        p = numpy.poly1d(z[0])
        ax[0].plot(distance2point, p(distance2point), "r--")

        sheeto.write(0, 4,
                      "Vert_A")
        sheeto.write(0, 5,
                      "delA")
        sheeto.write(0, 6,
                      "B")
        sheeto.write(0, 7,
                     "delB")

        sheeto.write(1, 4,
                     z[0][0])
        sheeto.write(1, 5,
                     numpy.sqrt(z[1][0][0]))
        sheeto.write(1, 6,
                     z[0][1])
        sheeto.write(1, 7,
                     numpy.sqrt(z[1][1][1]))

        z = numpy.polyfit(distance2point, longError, 1, full=False, cov=True)
        print z,"longError"
        p = numpy.poly1d(z[0])
        ax[1].plot(distance2point, p(distance2point), "r--")

        sheeto.write(0, 8,
                      "Axial_A")
        sheeto.write(0, 9,
                      "delA")
        sheeto.write(0, 10,
                      "B")
        sheeto.write(0, 11,
                     "delB")

        sheeto.write(1, 8,
                     z[0][0])
        sheeto.write(1, 9,
                     numpy.sqrt(z[1][0][0]))
        sheeto.write(1, 10,
                     z[0][1])
        sheeto.write(1, 11,
                     numpy.sqrt(z[1][1][1]))


        ax[1].set_title("Axial Error vs. Distance"+StringDescrip, fontsize=20)
        ax[0].set_ylabel("Error in cm", fontsize=20)
        ax[0].set_xlabel("Distance in cm", fontsize=20)
        #ax[0].set(xlim=(0,220))
        # axs[nop].set(ylim=(
        box = ax[0].get_position()
        # axs[nop].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax[0].set_position([box.x0, box.y0 + box.height * 0.0, box.width, box.height * 1.0])

        ax[0].set_title("Vertical error vs. Distance"+StringDescrip, fontsize=20)
        ax[1].set_ylabel("Error in cm", fontsize=20)
        ax[1].set_xlabel("Distance in cm", fontsize=20)
        #ax[1].set(xlim=(0,220))
        # axs[nop].set(ylim=(
        box = ax[1].get_position()
        # axs[nop].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax[1].set_position([box.x0, box.y0 + box.height * 0.0, box.width, box.height * 1.0])


        if 1:
            for axx in ax.flat:
               axx.label_outer()

        if 0:
            paththing = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/NewPMIMageFolder"
            otherbook.save(paththing + "/" + "Error_vs_distance_Coral" + "_INFO.xls")
            fig.savefig(paththing + '/' + "Error_vs_distance_Coral" + '.png', bbox_inches='tight',
                        dpi=600)
        try:
            if PlotDesciptiveDic["SaveImages"]=="y":
                otherbook.save(PlotDesciptiveDic["Path2SaveImages"] + '/' + PlotDesciptiveDic["SaveFileName"] +'_Dist_vs_Err' + "_INFO.xls")
                fig.savefig(PlotDesciptiveDic["Path2SaveImages"] + '/' + PlotDesciptiveDic["SaveFileName"] +'_Dist_vs_Err' + '.png', bbox_inches='tight',
                            dpi=600)
        except:
            print "q has not been pressed previously.   No data were saved."

        plt.show()

    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################

    def CalibrateCameraOpenCVCheckerBoard(self,Path):

        # from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objp = np.zeros((6*7,3), np.float32)
        objp = numpy.zeros((6 * 9, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        #images = glob.glob('C:/Users/Parrish/Documents/aOTIC/put on a hard drive/images/callibR/*.png')
        images = glob.glob(Path+'/*.png')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print fname
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print "found one"
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                if 0:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        #cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print "ret is",ret
        #print mtx
        #print dist
        print "Copy the following camera calibration parameters into the YAML section of ProjectMain.xlsx:"
        print ""
        print "Camera.fx,	    Camera.fy,	    Camera.cx,	    Camera.cy,	    Camera.k1,  	Camera.k2,  	Camera.p1,  	Camera.p2,   	Camera.k3"
        print mtx[0][0],",",mtx[1][1],",",mtx[0][2],",",mtx[1][2],",",dist[0][0],",",dist[0][1],",",dist[0][2],",",dist[0][3],",",dist[0][4]


    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################


    def outputPathsinCSVGraph(self,PlotDesciptiveDic, KinPlotDesciptiveDic, DictA,book):
        sheet4 = book.add_sheet('CVS output')

        ParamNames=["xt",	"yt",	"zt",	"vxt",	"vyt",	"vzt",	"vxt_avg",	"vyt_avg",	"vzt_avg",	"vavg",	"vv",	"vv_avg",	"xxt",
                    "vvt",	"xpt",	"vpvt_avg",	"axt",	"ayt",	"azt","aa",	"aavg",	"axt_avg",	"ayt_avg",	"azt_avg","disSTL","disSTLz",
                    "curvature",	"curveRadius"]
        ParamLabelDic={}
        ParamLabelDic[""]=""
        PlotNotScatter=False
        if KinPlotDesciptiveDic["Show Error Bars"]=="y":
            ErrorInGraph = True
        else:
            ErrorInGraph = False



        NumOfPanels=int(KinPlotDesciptiveDic["Panel number"])


        lineTrack=0

        for nop in range(NumOfPanels):
            BoxPlotArray=[]
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                ###################################
                # Initializing the dictionary data

                for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):

                    if 1:
                        dtv=DictA[
                            "dtv" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        dt=DictA[
                            "dt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        dtht=DictA[
                            "dtht" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        jj = 1

                        if 1:
                            if 1:
                                PlotoneT = KinPlotDesciptiveDic["Parameters to graph (comma separated)" + str(jj)][nop]
                                plotoneindx=0
                                for Plotone in PlotoneT.split(","):
                                    if len(Plotone.split("-")) == 1:

                                        DictToUse = DictA[
                                            Plotone + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                ii + 1)]
                                        try:
                                        #if 1:
                                            Err2Use=DictA[
                                                DictErr[Plotone] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                    ii + 1)]
                                        except:
                                            ErrorInGraph = False





                                    elif len(Plotone.split("-")) > 1 and ii==0: #becaurful not to pot double.
                                        # if Plotone.split("-")[0] in parameter2between:
                                        if 1:
                                            DictToUse = DictA[
                                                Plotone.split("-")[0] + Plotone.split("-")[1] + "-" + Plotone.split("-")[
                                                    2] + "__" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[1]][j]) + "-" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[2]][j])]
                                            try:
                                            #if 1:
                                                Err2Use = DictA[
                                                    DictErr[Plotone.split("-")[0]] + Plotone.split("-")[1] + "-" + Plotone.split("-")[
                                                        2] + "__" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[1]][j]) + "-" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[2]][j])]
                                            except:
                                                ErrorInGraph=False







                                    if KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(jj)][nop]=="":
                                        if len(DictToUse) == len(dt):
                                            dtuse = dt
                                            dtusename="dt"
                                        elif len(DictToUse) == len(dtv):
                                            dtuse = dtv
                                            dtusename = "dtv"
                                        elif len(DictToUse) == len(dtht):
                                            dtuse = dtht
                                            dtusename = "dtht"
                                    else:
                                        Plotonedt=KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(jj)][nop]
                                        if len(Plotonedt.split("-")) == 1:

                                            dtuse = DictA[
                                                Plotonedt + str(
                                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                    ii + 1)]
                                            dtusename = Plotonedt
                                        elif len(Plotonedt.split("-")) > 1:

                                            dtuse = DictA[
                                                Plotonedt.split("-")[0] + Plotonedt.split("-")[1] + "-" +
                                                Plotonedt.split("-")[
                                                    2] + "__" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotonedt.split("-")[1]][
                                                        j]) + "-" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotonedt.split("-")[2]][j])]
                                            dtusename = Plotonedt

                                    if plotoneindx==0:
                                        sheet4.write(0, lineTrack,
                                                     dtusename)
                                        for rowlit in range(len(dtuse)):
                                            if str(DictToUse[rowlit]) == "nan":
                                                sheet4.write(rowlit+1, lineTrack,
                                                             "")
                                            else:
                                                sheet4.write(rowlit+1, lineTrack,
                                                             dtuse[rowlit])
                                        lineTrack += 1

                                    sheet4.write(0, lineTrack,
                                                 Plotone)

                                    for rowlit in range(len(dtuse)):
                                        if str(DictToUse[rowlit]) == "nan":
                                            sheet4.write(rowlit+1,lineTrack,
                                                         "")
                                        else:
                                            sheet4.write(rowlit+1,lineTrack,
                                                         DictToUse[rowlit])
                                    lineTrack += 1
                                    plotoneindx +=1










    ################################################################################################################################    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################
    ################################################################################################################################


    def outputGraphs2(self, PlotDesciptiveDic,KinPlotDesciptiveDic,DictA,DictErr,book):
        if PlotDesciptiveDic["OutputCVSType"] == "y":
            sheet4 = book.add_sheet('CVS output')
        ParamLabelDic={}
        ParamLabelDic[""]=""
        PlotNotScatter=False
        lineTrack=0
        if KinPlotDesciptiveDic["Show Error Bars"]=="y":
            ErrorInGraph = True
        else:
            ErrorInGraph = False



        parameter2point=["xp","vp","ap"]
        parameter2between=["xb","vb","ab"]
        NumOfPanels=int(KinPlotDesciptiveDic["Panel number"])
        # fig = plt.figure(figsize=(float(KinPlotDesciptiveDic["Panel Size x"]), float(KinPlotDesciptiveDic["Panel Size y"])))
        if KinPlotDesciptiveDic["Polar Plot"]=="n":
            figK, ax = plt.subplots(NumOfPanels, figsize=(
                float(KinPlotDesciptiveDic["Panel Size x"]), float(KinPlotDesciptiveDic["Panel Size y"])))
        if KinPlotDesciptiveDic["Polar Plot"]=="y":
            figK, ax = plt.subplots(NumOfPanels, figsize=(
                float(KinPlotDesciptiveDic["Panel Size x"]), float(KinPlotDesciptiveDic["Panel Size y"])),subplot_kw = dict(projection='polar'))
         #figK.canvas.manager.window.move(0, 0)
        if NumOfPanels==1:
            axs=[ax]
        else:
            axs=ax

        colorVarString = PlotDesciptiveDic["PathColorGradientString"]

        ##############################################################################
        if colorVarString != "":
            inColorTotal = []
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                print str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)]), "PlotDesciptiveDic"
                for i in range(len(PlotDesciptiveDic["Insectnumbers1"])):
                    inColor = DictA[
                        colorVarString + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(ii + 1)]
                    inColorTotal += list(inColor)

            inColorTotal = numpy.array(inColorTotal)
            inColorMax = numpy.nanmax(inColorTotal)
            inColorMin = numpy.nanmin(inColorTotal)
            print inColorMax, inColorMin, "max min polar"
            aCol = 1 / (inColorMax - inColorMin)
            bCol = -inColorMin / (inColorMax - inColorMin)



        dtsallaxis = []
        for nop in range(NumOfPanels):
            BoxPlotArray=[]
            BoxPlotLabels=[]
            for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):
                ###################################
                # Initializing the dictionary data

                for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):

                    if 1:
                        dtv=DictA[
                            "dtv" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        dt=DictA[
                            "dt" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        dtht=DictA[
                            "dtht" + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)]
                        jj = 1

                        #else:
                        if 1:
                            if 1:
                                PlotoneT = KinPlotDesciptiveDic["Parameters to graph (comma separated)" + str(jj)][nop].strip()
                                PlotoneLabels = KinPlotDesciptiveDic["Path Labels" + str(jj)][nop].strip()
                                if 0:#KinPlotDesciptiveDic["Polar Plot"] == "y" and PlotoneT.split("a")[0] != "thet":
                                    continue

                                if KinPlotDesciptiveDic["Color gradient across plots"]!="":
                                    #print "colors length thing",PlotDesciptiveDic["insectNumGroupNumber"]*len(PlotDesciptiveDic["Insectnumbers1"])*len(PlotoneT.split(","))
                                    Multcolorarray = getattr(plt.cm, KinPlotDesciptiveDic["Color gradient across plots"])\
                                        (numpy.linspace(0, 1, PlotDesciptiveDic["insectNumGroupNumber"]*len(PlotDesciptiveDic["Insectnumbers1"])*len(PlotoneT.split(","))))
                                Plotoneindx=0
                                for Plotone1 in PlotoneT.split(","):
                                    Plotone=Plotone1.strip()
                                    if len(Plotone.split("-")) == 1:
                                        if len(Plotone.split(".")) == 1:

                                            DictToUse = DictA[
                                                Plotone + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                    ii + 1)]
                                            try:
                                            #if 1:
                                                Err2Use=DictA[
                                                    DictErr[Plotone] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                        ii + 1)]
                                            except:
                                                ErrorInGraph = False
                                        else:
                                            if ii==int(Plotone.split(".")[1])-1:
                                                DictToUse = DictA[
                                                    Plotone.split(".")[0] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                        ii + 1)]
                                                try:
                                                #if 1:
                                                    Err2Use=DictA[
                                                        DictErr[Plotone.split(".")[0]] + str(PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                            ii + 1)]
                                                except:
                                                    ErrorInGraph = False




                                        #############################################################################
                                        if 0:#KinPlotDesciptiveDic["Polar Plot"] == "y":
                                            phiDictToUse = DictA[
                                                "phiToR" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                    ii + 1)]
                                    elif len(Plotone.split("-")) > 1 and ii==0: #becaurful not to pot double.
                                        # if Plotone.split("-")[0] in parameter2between:
                                        if 1:
                                            DictToUse = DictA[
                                                Plotone.split("-")[0] + Plotone.split("-")[1] + "-" + Plotone.split("-")[
                                                    2] + "__" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[1]][j]) + "-" + str(
                                                    PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[2]][j])]
                                            try:
                                            #if 1:
                                                Err2Use = DictA[
                                                    DictErr[Plotone.split("-")[0]] + Plotone.split("-")[1] + "-" + Plotone.split("-")[
                                                        2] + "__" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[1]][j]) + "-" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[2]][j])]
                                            except:
                                                ErrorInGraph=False





                                            #######################################################################3
                                            if 0:#KinPlotDesciptiveDic["Polar Plot"] == "y":
                                                #print "phi" + Plotone.split("-")[0].split("a")[1]
                                                phiDictToUse = DictA[
                                                    "phi" + Plotone.split("-")[0].split("a")[1] + Plotone.split("-")[
                                                        1] + "-" + Plotone.split("-")[
                                                        2] + "__" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[1]][
                                                            j]) + "-" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotone.split("-")[2]][j])]


                                    if KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(jj)][nop]=="":
                                        #print "di", nop, len(DictToUse),len(dt),len(dtv),len(dtht)
                                        if len(DictToUse) == len(dt):
                                            dtuse = dt
                                            dtusename = "dt"
                                        elif len(DictToUse) == len(dtv):
                                            dtuse = dtv
                                            dtusename = "dtv"
                                        elif len(DictToUse) == len(dtht):
                                            dtuse = dtht
                                            dtusename = "dtht"

                                    else:
                                        Plotonedt=KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(jj)][nop].strip()
                                        if len(Plotonedt.split("-")) == 1:

                                            dtuse = DictA[
                                                Plotonedt + str(
                                                    PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                                    ii + 1)]
                                            dtusename = Plotonedt



                                        elif len(Plotonedt.split("-")) > 1:
                                            # if Plotone.split("-")[0] in parameter2between:
                                            if 1:
                                                dtuse = DictA[
                                                    Plotonedt.split("-")[0] + Plotonedt.split("-")[1] + "-" +
                                                    Plotonedt.split("-")[
                                                        2] + "__" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotonedt.split("-")[1]][
                                                            j]) + "-" + str(
                                                        PlotDesciptiveDic["Insectnumbers" + Plotonedt.split("-")[2]][j])]
                                                dtusename = Plotonedt


                                    if KinPlotDesciptiveDic["From relative start"] == "y":
                                        dtuse = numpy.array(dtuse)
                                        for inan in range(len(dtuse)):
                                            if numpy.isnan(dtuse[inan])==False:
                                                inanno=inan
                                                break

                                        dtuse[:] = dtuse[:] - dtuse[inanno]
                                    # plt.plot(dtuse,DictToUse , label=Plotone)  # derivatev of kalman
                                    #axs[nop].plot(dtuse, DictToUse, label=Plotone)
                                    #axs[nop].add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

                                    #########################################################
                                    #Path labels

                                    if PlotoneLabels == "":
                                        Pathlabel = PlotDesciptiveDic["InsectLabels" + str(ii + 1)][j]
                                    else:
                                        Pathlabel = PlotDesciptiveDic["InsectLabels" + str(ii + 1)][j] + "_" + \
                                                    PlotoneLabels.split(",")[Plotoneindx]
                                    if PlotDesciptiveDic["OutputCVSType"]=="y":
                                        if Plotoneindx == 0:
                                            sheet4.write(1, lineTrack,
                                                         dtusename)
                                            sheet4.write(0, lineTrack,
                                                         Pathlabel)
                                            for rowlit in range(len(dtuse)):
                                                if str(DictToUse[rowlit]) == "nan":
                                                    sheet4.write(rowlit + 2, lineTrack,
                                                                 "")
                                                else:
                                                    sheet4.write(rowlit + 2, lineTrack,
                                                                 dtuse[rowlit])
                                            lineTrack += 1

                                        sheet4.write(1, lineTrack,
                                                     Plotone)
                                        sheet4.write(0, lineTrack,
                                                     Pathlabel)
                                        for rowlit in range(len(dtuse)):
                                            if str(DictToUse[rowlit]) == "nan":
                                                sheet4.write(rowlit + 2, lineTrack,
                                                             "")
                                            else:
                                                sheet4.write(rowlit + 2, lineTrack,
                                                             DictToUse[rowlit])
                                        lineTrack += 1
                                        #plotoneindx += 1


                                    if KinPlotDesciptiveDic["Polar Plot"] == "y":
                                        if (len(Plotone.split("-")) > 1 and ii == 0) or len(Plotone.split("-")) <= 1:
                                            phiDictToUse = numpy.array(
                                                dtuse) * 3.1415926 / 180  # Because there is some inherent error in the polar plot.   It displays in degrees but plots in radians.
                                            #phiDictToUse) *3.1415926 / 180  # Because there is some inherent error in the polar plot.   It displays in degrees but plots in radians.

                                            if colorVarString=="":
                                                axs[nop].plot(phiDictToUse, DictToUse,
                                                              color=PlotDesciptiveDic["InsectColors" + str(ii + 1)][j],
                                                              label=Plotone)

                                            else:
                                                inColor = DictA[
                                                    colorVarString + str(
                                                        PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][i]) + "_" + str(
                                                        ii + 1)]
                                                inColor = numpy.array(inColor)

                                                if len(inColor) < len(phiDictToUse):
                                                    inColor = numpy.insert(inColor, 1, inColor[0])
                                                if len(inColor) < len(phiDictToUse):
                                                    inColor = numpy.insert(inColor, len(inColor), inColor[len(inColor) - 1])

                                                #for io in range(DictToUse):
                                                axs[nop].scatter(phiDictToUse, DictToUse, color=plt.cm.rainbow(aCol * inColor + bCol),label=Plotone)
                                                #print len(phiDictToUse),len(inColor),"polPlot"
                                                #print inColor
                                            axs[nop].set_theta_zero_location("N")
                                            axs[nop].set_theta_direction(-1)

                                            #axs[nop].tick_params(axis='x', labelsize=20)
                                            #axs[nop].tick_params(axis='y', labelsize=20)
                                    else:
                                        #axs[nop].plot(dtuse, DictToUse, label=Plotone)
                                        if KinPlotDesciptiveDic["Box Plot"] == "n":
                                            if (len(Plotone.split("-")) > 1 and ii==0) or len(Plotone.split("-")) <= 1:
                                                if nop==0:
                                                    None
                                                    #aqwq
                                                    #print j,";",2*numpy.polyfit(dtuse,DictToUse,2,cov=True)[0][0],";",numpy.sqrt(numpy.polyfit(dtuse,DictToUse,2,cov=True)[1][0][0])*2
                                                if PlotNotScatter==True:
                                                    axs[nop].plot(dtuse, DictToUse,color=PlotDesciptiveDic["InsectColors" + str(ii + 1)][j], label=PlotDesciptiveDic["InsectLabels" + str(ii + 1)][j])
                                                else:
                                                    if len(Plotone.split(".")) != 1:
                                                        iirc=int(Plotone.split(".")[1]) - 1
                                                    else:
                                                        iirc=ii
                                                    if ii == iirc:
                                                        if ErrorInGraph == False:
                                                            if 1:
                                                                #print "index",ii + PlotDesciptiveDic["insectNumGroupNumber"] * (j)+PlotDesciptiveDic["insectNumGroupNumber"] *len(PlotDesciptiveDic["Insectnumbers1"]) * (Plotoneindx)



                                                                if KinPlotDesciptiveDic["Color gradient across plots"]!="":
                                                                    #print len(Multcolorarray),"Multcolorarray"
                                                                    axs[nop].scatter(dtuse, DictToUse,
                                                                                     color=Multcolorarray[ii+PlotDesciptiveDic["insectNumGroupNumber"]*(j)
                                                                                                          +PlotDesciptiveDic["insectNumGroupNumber"]*
                                                                                                          len(PlotDesciptiveDic["Insectnumbers1"])*(Plotoneindx)],
                                                                                     label=Pathlabel,marker=".")
                                                                else:
                                                                    axs[nop].scatter(dtuse, DictToUse,
                                                                                  color=PlotDesciptiveDic["InsectColors" + str(ii + 1)][j],
                                                                                  label=Pathlabel,marker=".")
                                                            if 0:
                                                                print len(dtuse),len(DictToUse[2:len(DictToUse)-4])
                                                                axs[nop].scatter(dtuse, DictToUse[2:len(DictToUse)-4],
                                                                              color=PlotDesciptiveDic["InsectColors" + str(ii + 1)][j],
                                                                              label=Plotone)

                                                        if ErrorInGraph == True:

                                                            axs[nop].errorbar(dtuse, DictToUse,yerr= Err2Use,xerr=None,
                                                                            color=
                                                                             PlotDesciptiveDic["InsectColors" + str(ii + 1)][j],
                                                                             label=Plotone,fmt='o')

                                                dtsallaxis=dtsallaxis+list(dtuse)
                                            else:
                                                None
                                        else:

                                            DictToUse=numpy.array(DictToUse)
                                            filtered_data = DictToUse[~numpy.isnan(DictToUse)]
                                            BoxPlotArray.append(filtered_data)
                                            BoxPlotLabels.append(PlotDesciptiveDic["InsectLabels" + str(ii + 1)][j])
                                            #print "box plots"

                                    Plotoneindx+=1

            if KinPlotDesciptiveDic["Box Plot"] == "y":
                positionsn=range(len(BoxPlotArray))
                #positionsn[1]=0
                #positionsn[2] = 0
                #print BoxPlotArray
                bp=axs[nop].boxplot(BoxPlotArray, positions=positionsn,labels=BoxPlotLabels, showfliers=False,patch_artist=True)
                if 1:
                    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                        plt.setp(bp[element], color="k")
                    ii=0
                    j=0
                    for patch in bp['boxes']:
                        if 1:
                            patch.set(facecolor=Multcolorarray[ii + PlotDesciptiveDic["insectNumGroupNumber"] * (j)
                                                   + PlotDesciptiveDic["insectNumGroupNumber"] *
                                                   len(PlotDesciptiveDic["Insectnumbers1"]) * (0)])

                        if 0:
                            patch.set(facecolor=PlotDesciptiveDic["InsectColors" + str(ii + 1)][j/2])
                        j+=1
                        #patch.set(hatch='/')

            #using jj=0 looks right for right here.
            if KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(1)][nop] != "":
                axs[nop].set(ylim=(
                float(KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(1)][nop].split(",")[0]),
                float(KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(1)][nop].split(",")[1])))

            if KinPlotDesciptiveDic["Polar Plot"] != "y":
                if KinPlotDesciptiveDic["Y-axis label" + str(1)][nop] != "":
                    axs[nop].set_ylabel(KinPlotDesciptiveDic["Y-axis label" + str(1)][nop],fontsize=20)
                if KinPlotDesciptiveDic["X-axis label" + str(1)][nop] != "":
                    axs[nop].set_xlabel(KinPlotDesciptiveDic["X-axis label" + str(1)][nop],fontsize=20)
            #print KinPlotDesciptiveDic["Graph Title" + str(1)][nop], "KinPlotDesciptiveDic[Graph Title + str(1)][nop]"
            if KinPlotDesciptiveDic["Graph Title" + str(1)][nop] != "":
                if KinPlotDesciptiveDic["Polar Plot"] != "y":
                    axs[nop].set_title(KinPlotDesciptiveDic["Graph Title" + str(1)][nop],fontsize=20)
                else:
                    axs[nop].set_title(KinPlotDesciptiveDic["Graph Title" + str(1)][nop], fontsize=20,y=1.1)#va = 'bottom')

            if KinPlotDesciptiveDic["X-axis range (comma separated)" + str(1)][nop] != "":
                axs[nop].set(xlim=(
                float(KinPlotDesciptiveDic["X-axis range (comma separated)" + str(1)][nop].split(",")[0]),
                float(KinPlotDesciptiveDic["X-axis range (comma separated)" + str(1)][nop].split(",")[1])))

            if  KinPlotDesciptiveDic["Polar Plot"] != "y":
                box = axs[nop].get_position()
                #axs[nop].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                if KinPlotDesciptiveDic["Graph Right Offset" + str(1)][nop]=="":
                    boxThinner =.9
                else:
                    boxThinner=float(KinPlotDesciptiveDic["Graph Right Offset" + str(1)][nop])
                axs[nop].set_position([box.x0+box.width * 0.1, box.y0+box.height*.2, box.width * boxThinner, box.height*.8])
                if KinPlotDesciptiveDic["Box Plot"] != "y":
                    axs[nop].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #elif KinPlotDesciptiveDic["Box Plot"] != "y" and KinPlotDesciptiveDic["Polar Plot"] == "y":
            else:
                box = axs[nop].get_position()
                #axs[nop].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axs[nop].set_position([box.x0, box.y0+box.height*.2, box.width, box.height*.8])
                #axs[nop].legend(loc='upper right')
                #axs[nop].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ########## size of the numbers
            #axs[nop].tick_params(axis='x', labelsize=25)
            #axs[nop].tick_params(axis='y', labelsize=25)
        if KinPlotDesciptiveDic["Polar Plot"]=="n" and KinPlotDesciptiveDic["Box Plot"] != "y" and KinPlotDesciptiveDic["Sync X axis"]=="y":
            dtsallaxis=numpy.array(dtsallaxis)
            dtsallaxismin=numpy.nanmin(dtsallaxis)
            dtsallaxismax = numpy.nanmax(dtsallaxis)
            if 1:
                for nop in range(NumOfPanels):
                    axs[nop].set(xlim=(dtsallaxismin,dtsallaxismax))
                    #axs[0].plot(dtuse, DictToUse, label=Plotone)

                    #axs[0].set_title('Creole Wrasse-Sunny')
                    #axs[0].set(ylim=(0,50))#, xlim=(76, 690))


                #KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(j)] = (
                #    workbook2.sheet_by_name(KinPlotSheetName).cell(ik, KinPlotNameInsectHeader[
                #        "Y-axis range (comma separated)"]).value)


        if nop>1:
            for ax in axs.flat:
               ax.label_outer()

        #mngr = plt.get_current_fig_manager()
        #mngr.window.SetPosition(100, 20)
        #mngr.SetPosition((100, 20))
        #mngr.window.setGeometry(100,200,300,400)
        #figK.canvas.manager.window.SetPosition(0, 0)
        #figK.canvas.manager.window.
        #plt.show()

        if PlotDesciptiveDic["SaveImages"]=="y":
            figK.savefig(PlotDesciptiveDic["Path2SaveImages"] + '/' + PlotDesciptiveDic["SaveFileName"] +'_Kin' + '.png', bbox_inches='tight',
                        dpi=600)





    def extractUVdata(self, fUV, datOrdAlt, cameraMatrix1, distCoeffs1, width, height, frame1, polydata, obbTree):
        print "you pressed 8"

        try:
            fUVSTL = fUV.create_group('STL')
            print "made STL"


        except:
            fUVSTL = fUV['STL']

        # squidVert
        print polydata.GetNumberOfPoints(), "number of point"
        print polydata.GetNumberOfCells(), "number of point"

        print polydata.GetCellData(), "cells"
        print polydata.GetCell(0).GetPoints().GetPoint(0)
        print polydata.GetCell(0).GetPoints().GetPoint(1)
        print polydata.GetCell(0).GetPoints().GetPoint(2)

        RotMat0 = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        TranMat0 = numpy.zeros(3)

        pSource = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])

        # going through all the cells
        for ig in range(polydata.GetNumberOfCells()):
            STLcount = 0
            CellMesh = []
            OrigCellMesh = []
            # looking at all the verticies in a triangle
            for vv in range(3):
                pTarget = numpy.array(polydata.GetCell(ig).GetPoints().GetPoint(vv))
                pointsIntersection, cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)
                print pTarget, pointsIntersection, cellIdsInter
                # looking to see if it intersects the mesh only once.  If it does then it is facing the camera
                if len(pointsIntersection) != 1:
                    STLcount += 1
                else:
                    CellMesh.append(pointsIntersection[0])
                    OrigCellMesh.append(polydata.GetCell(ig).GetPoints().GetPoint(vv))
            if STLcount == 0:
                makeInt = False
                CellMesh = numpy.array(CellMesh)
                CellMeshT = numpy.transpose(CellMesh)
                # print CellMesh,"CellMesh"
                MeshProjCell = self.ReturnMeshProjection(CellMeshT, RotMat0, TranMat0, datOrdAlt,
                                                         cameraMatrix1,
                                                         distCoeffs1,
                                                         width, height, makeInt)

                print MeshProjCell
                xbound = numpy.array([MeshProjCell[0][0], MeshProjCell[1][0], MeshProjCell[2][0]])
                ybound = numpy.array([MeshProjCell[0][1], MeshProjCell[1][1], MeshProjCell[2][1]])
                # print xbound,ybound,"bound"


                # here is the list of pixels that are occluded by the triangle
                PixelArray = []

                # and how much they are occluded.
                PixelWeight = []

                # here is fancy geometry using ogr

                # looking at the intersection of a pixil square and the Mesh triangle

                for ix in range(numpy.floor(xbound.min()), numpy.ceil(xbound.max())):
                    for iy in range(numpy.floor(ybound.min()), numpy.ceil(ybound.max())):
                        # this is a square pixel
                        subjectPolygon = [[float(ix), float(iy)], [float(ix + 1), float(iy)],
                                          [float(ix + 1), float(iy + 1)], [float(ix), float(iy + 1)]]

                        # this is the triangle.  There are two here for some reason
                        clipPOlygon = [[MeshProjCell[0][0], MeshProjCell[0][1]],
                                       [MeshProjCell[1][0], MeshProjCell[1][1]],
                                       [MeshProjCell[2][0], MeshProjCell[2][1]]]
                        clipPOlygon = [[MeshProjCell[2][0], MeshProjCell[2][1]],
                                       [MeshProjCell[1][0], MeshProjCell[1][1]],
                                       [MeshProjCell[0][0], MeshProjCell[0][1]]]


                        if 0: #delete when you can
                            # here we are putting the polygons in the right format
                            # pixel
                            wkt1 = "POLYGON (("
                            wkt1it = 0
                            for sP in subjectPolygon:
                                if wkt1it == 0:
                                    wkt1 += str(sP[0]) + " " + str(sP[1])
                                    wkt1it += 1
                                else:
                                    wkt1 += " , " + str(sP[0]) + " " + str(sP[1])
                            wkt1 += " , " + str(subjectPolygon[0][0]) + " " + str(subjectPolygon[0][1])
                            wkt1 += "))"

                            # print wkt1
                            # triangle
                            wkt2 = "POLYGON (("
                            wkt1it = 0
                            for sP in clipPOlygon:
                                if wkt1it == 0:
                                    wkt2 += str(sP[0]) + " " + str(sP[1])
                                    wkt1it += 1
                                else:
                                    wkt2 += " , " + str(sP[0]) + " " + str(sP[1])
                            wkt2 += " , " + str(clipPOlygon[0][0]) + " " + str(clipPOlygon[0][1])
                            wkt2 += "))"

                            # print wkt2

                            poly1 = ogr.CreateGeometryFromWkt(wkt1)
                            poly2 = ogr.CreateGeometryFromWkt(wkt2)

                            # getting the intersection
                            intersection = poly1.Intersection(poly2)


                            #????
                            areaTriangle = poly2.GetArea()

                        poly1 = Polygon(subjectPolygon)

                        poly2 = Polygon(clipPOlygon)
                        areaTriangle=poly2.area

                        try:
                            #area = intersection.GetArea()
                            area=poly1.intersection(poly2).area
                        except:
                            area = 0
                        # here we choose the pixels that were intersected as well as how much of the pixel was interesected
                        if area > 0.0000001:
                            PixelArray.append([ix, iy])
                            PixelColor.append(frame1[ix, iy])
                            PixelWeight.append(area)

                # now we are recording it.
                if 1:
                    try:
                        fUVSTL[filenameSTL].create_group(str(int(ig)))

                        fUVSTL[filenameSTL][str(int(ig))].create_dataset("PixelArray",
                                                                         data=PixelArray)

                        fUVSTL[filenameSTL][str(int(ig))].create_dataset("PixelColor",
                                                                         data=PixelColor)

                        fUVSTL[filenameSTL][str(int(ig))].create_dataset("clipPOlygon",
                                                                         data=clipPOlygon)
                        fUVSTL[filenameSTL][str(int(ig))].create_dataset("CellMesh",
                                                                         data=CellMesh)

                    except:

                        fUVSTL[filenameSTL][str(int(ig))]["PixelArray"] = PixelArray
                        fUVSTL[filenameSTL][str(int(ig))]["PixelColor"] = PixelColor
                        fUVSTL[filenameSTL][str(int(ig))]["clipPOlygon"] = clipPOlygon
                        fUVSTL[filenameSTL][str(int(ig))]["CellMesh"] = CellMesh
                        fUVSTL[filenameSTL][str(int(ig))]["OrigCellMesh"] = OrigCellMesh

        print "Done"
        RoiStringIndex = 0

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################











    #################################################################################################################
            #################################################################################################################
            #################################################################################################################
            #################################################################################################################


            #squid Only


            #################################################################################################################
            #################################################################################################################
            #################################################################################################################
            #################################################################################################################




    ##############################################################################################################
    #########################################################################################################
    ####  asamesh

    #  returns a 2d list of points that are projected from a specific camera projection

    def ReturnMeshProjection(self, pcMat, RotMat, TranMat, dat1, cameraMatrix1, distCoeffs1, width, height, makeInt):

        pcMatm = numpy.copy(pcMat)

        zvect1 = numpy.zeros(3)
        zvect2 = numpy.zeros(3)

        xvect1 = numpy.zeros(3)
        xvect2 = numpy.zeros(3)

        yvect1 = numpy.zeros(3)
        yvect2 = numpy.zeros(3)

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        boundary = .01 * width

        CamOrgn1 = numpy.zeros(3)
        CamOrgn2 = numpy.zeros(3)

        rotVect1 = numpy.zeros(3)
        rotVect2 = numpy.zeros(3)

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

        dat2 = dat1

        xvect2[0] = dat2[0]
        xvect2[1] = dat2[1]
        xvect2[2] = dat2[2]

        yvect2[0] = dat2[3]
        yvect2[1] = dat2[4]
        yvect2[2] = dat2[5]

        zvect2[0] = dat2[6]
        zvect2[1] = dat2[7]
        zvect2[2] = dat2[8]

        CamOrgn1[0] = dat1[9]
        CamOrgn1[1] = dat1[10]
        CamOrgn1[2] = dat1[11]

        CamOrgn2[0] = dat2[9]
        CamOrgn2[1] = dat2[10]
        CamOrgn2[2] = dat2[11]

        CamOrgn2 = numpy.array(CamOrgn2)
        CamOrgn1 = numpy.array(CamOrgn1)

        zvect1 = numpy.array(zvect1)
        zvect2 = numpy.array(zvect2)

        xvect1 = numpy.array(xvect1)
        xvect2 = numpy.array(xvect2)

        yvect1 = numpy.array(yvect1)
        yvect2 = numpy.array(yvect2)


        # getting the shape
        xyz, meshnum = pcMatm.shape

        # Choosing which to modify
        if 1:

            # rotating before the translation
            pcMatm = numpy.matmul(RotMat, pcMatm)

            # Translating
            for ii in range(meshnum):
                pcMatm[0, ii] = pcMatm[0, ii] + TranMat[0]
                pcMatm[1, ii] = pcMatm[1, ii] + TranMat[1]
                pcMatm[2, ii] = pcMatm[2, ii] + TranMat[2]

        else:  # we probablty wont use this again but it did work

            RotMat = numpy.array(RotMat)
            dst, jacobian = cv2.Rodrigues(RotMat, rot)
            tran = TranMat
            CamOrgn1[0] = TranMat[0]
            CamOrgn1[1] = TranMat[1]
            CamOrgn1[2] = TranMat[2]

        # finding the undistorted boundary for the image plane

        test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
        test[0][0][0] = boundary
        test[0][0][1] = boundary

        undistort1 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        test[0][0][0] = width - boundary
        test[0][0][1] = height - boundary

        undistort2 = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

        bound1 = [undistort1[0][0][0], undistort1[0][0][1], undistort2[0][0][0], undistort2[0][0][1]]

        # Mesh Projection initialization
        MeshProj = [[0, 0]]
        k1 = 0
        k2 = 0  # various helpin parameters

        for i in range(meshnum):

            cop2 = CamOrgn1 - pcMatm[:, i]  # work

            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            if 1:  # this is the original camera image plane projection

                d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
                d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
                d3test[0][0][2] = 1

                if d3test[0][0][0] >= bound1[0] and d3test[0][0][0] <= bound1[2] and d3test[0][0][1] >= bound1[
                    1] and \
                                d3test[0][0][1] <= bound1[3]:

                    imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

                    if makeInt == True:
                        if k2 == 0:
                            MeshProj[0][0] = int(imagePoints2[0][0][0])
                            MeshProj[0][1] = int(imagePoints2[0][0][1])
                            k2 += 1
                        else:
                            MeshProj.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])

                    else:
                        # print "there are no more drift solders"
                        if k2 == 0:
                            MeshProj[0][0] = (imagePoints2[0][0][0])
                            MeshProj[0][1] = (imagePoints2[0][0][1])
                            k2 += 1
                        else:
                            MeshProj.append([(imagePoints2[0][0][0]), (imagePoints2[0][0][1])])

            else:
                # here we are testing out the rot and tran thing
                if 1:
                    d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
                    d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
                    d3test[0][0][2] = 1
                else:

                    d3test[0][0][0] = cop2[0]
                    d3test[0][0][1] = cop2[1]
                    d3test[0][0][2] = cop2[2]
                imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

                if makeInt == True:
                    if k2 == 0:
                        MeshProj[0][0] = int(imagePoints2[0][0][0])
                        MeshProj[0][1] = int(imagePoints2[0][0][1])
                        k2 += 1
                    else:
                        MeshProj.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])
                else:
                    # print "there are no more drift solders"
                    if k2 == 0:
                        MeshProj[0][0] = (imagePoints2[0][0][0])
                        MeshProj[0][1] = (imagePoints2[0][0][1])
                        k2 += 1
                    else:
                        MeshProj.append([(imagePoints2[0][0][0]), (imagePoints2[0][0][1])])

        return MeshProj





        ##################################################################################################################

    def ReturnPointsection(self,ThreeDMarkedPoints, RotMat, TranMat, dat1, cameraMatrix1, distCoeffs1, makeInt,Tobject,frame1num,fwData):
        TwoDMarkedPoints2 = []
        ThreeDMarkedPoints2 = []
        insectPointArray=[]
        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:

                TwoDMarkedPoints2.append(
                    [fwData["camera1"][insectNum][str(int(frame1num))][0],
                     fwData["camera1"][insectNum][str(int(frame1num))][1]])

                ThreeDMarkedPoints2.append(ThreeDMarkedPoints[i])
                insectPointArray.append(insectNum)
            except:
                None


        ThreeDMarkedPoints2 = numpy.array(ThreeDMarkedPoints2)
        TwoDMarkedPoints2 = numpy.array(TwoDMarkedPoints2)
        ThreeDMarkedPoints2=numpy.transpose(ThreeDMarkedPoints2)

        if len(ThreeDMarkedPoints2)>0:
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

            xyz, meshnum = ThreeDMarkedPoints2.shape

            if 1:
                # rotating before the translation
                ThreeDMarkedPoints2 = numpy.matmul(RotMat, ThreeDMarkedPoints2)

                # Translating
                for ii in range(meshnum):
                    ThreeDMarkedPoints2[0, ii] = ThreeDMarkedPoints2[0, ii] + TranMat[0]
                    ThreeDMarkedPoints2[1, ii] = ThreeDMarkedPoints2[1, ii] + TranMat[1]
                    ThreeDMarkedPoints2[2, ii] = ThreeDMarkedPoints2[2, ii] + TranMat[2]
            # Mesh Projection initialization
            MeshProj = [[0, 0]]
            k1 = 0
            k2 = 0  # various helpin parameters
            for i in range(meshnum):

                cop2 = CamOrgn1 - ThreeDMarkedPoints2[:, i]  # work
                d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)
                if 1:  # this is the original camera image plane projection
                    d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
                    d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
                    d3test[0][0][2] = 1
                    if 1:
                        imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)
                        if makeInt == True:
                            if k2 == 0:
                                MeshProj[0][0] = int(imagePoints2[0][0][0])
                                MeshProj[0][1] = int(imagePoints2[0][0][1])
                                k2 += 1
                            else:
                                MeshProj.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])

                        else:
                            # print "there are no more drift solders"
                            if k2 == 0:
                                MeshProj[0][0] = (imagePoints2[0][0][0])
                                MeshProj[0][1] = (imagePoints2[0][0][1])
                                k2 += 1
                            else:
                                MeshProj.append([(imagePoints2[0][0][0]), (imagePoints2[0][0][1])])

                else:
                    # here we are testing out the rot and tran thing
                    if 1:
                        d3test[0][0][0] = xvect1.dot(cop2) / zvect1.dot(cop2)
                        d3test[0][0][1] = yvect1.dot(cop2) / zvect1.dot(cop2)
                        d3test[0][0][2] = 1
                    else:
                        d3test[0][0][0] = cop2[0]
                        d3test[0][0][1] = cop2[1]
                        d3test[0][0][2] = cop2[2]
                    imagePoints2, jacobian = cv2.projectPoints(d3test, rot, tran, cameraMatrix1, distCoeffs1)

                    if makeInt == True:
                        if k2 == 0:
                            MeshProj[0][0] = int(imagePoints2[0][0][0])
                            MeshProj[0][1] = int(imagePoints2[0][0][1])
                            k2 += 1
                        else:
                            MeshProj.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])
                    else:
                        # print "there are no more drift solders"
                        if k2 == 0:
                            MeshProj[0][0] = (imagePoints2[0][0][0])
                            MeshProj[0][1] = (imagePoints2[0][0][1])
                            k2 += 1
                        else:
                            MeshProj.append([(imagePoints2[0][0][0]), (imagePoints2[0][0][1])])
        else:
            MeshProj=[]
        return MeshProj,insectPointArray






############################################################################################################

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = numpy.asarray(axis)
        axis = axis / numpy.linalg.norm(axis)

        a = numpy.cos(theta / 2.0)
        b, c, d = -axis * numpy.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    #########################################################################################################






    #########################################################################################################
    #
    #Getting the points from an intersection of a line from a source point and a target point
    #
    def VTKcaster(self, obbTree, pSource, pTarget):
        #  I think this is the initialization of the cell IDs and the points they will be assigned when code is run.
        pointsVTKintersection = vtk.vtkPoints()
        idsVTKintersection = vtk.vtkIdList()

        # this is the ray cast with obbtree.
        code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, idsVTKintersection)

        # this is the collection of intersection points from that ray in the mesh.  we are only going to choose the first one.
        pointsVTKIntersectionData = pointsVTKintersection.GetData()
        # this is the number of points interested with the line.
        noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
        PointsinVTKIntersection = []
        cellIdsInter = []

        if noPointsVTKIntersection != 0:
            for i in range(noPointsVTKIntersection):
                PointsinVTKIntersection.append(pointsVTKIntersectionData.GetTuple3(i))
                cellIdsInter.append(idsVTKintersection.GetId(i))

        return PointsinVTKIntersection,cellIdsInter



        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        ##
        ## XXX
        ##
    def STLsurfacePycastVTK(self, obbTree,stlMax):
        pcMat = []
        maxx = 30.1
        #maxSquare = 200.#for cuttlefish
        #maxSquare = 10.#for squid
        maxSquare=stlMax*1.5

        for ii in range(int(maxx)):
            for jj in range(int(maxx)):
                if 0:#normal mode
                    pSource = [0.0, ii * 2 * (maxSquare / maxx) - maxSquare, jj * 2 * (maxSquare / maxx) - maxSquare]
                    pTarget = [-500, ii * 2 * (maxSquare / maxx) - maxSquare, jj * 2 * (maxSquare / maxx) - maxSquare]
                if 1:
                    pSource = [ ii * 2 * (maxSquare / maxx) - maxSquare, jj * 2 * (maxSquare / maxx) - maxSquare,20]
                    pTarget = [ ii * 2 * (maxSquare / maxx) - maxSquare, jj * 2 * (maxSquare / maxx) - maxSquare,-100]

                #  this is the initialization of the cell IDs and the points they will be assigned when code is run.
                pointsVTKintersection = vtk.vtkPoints()
                idsVTKintersection = vtk.vtkIdList()

                # this is the ray cast with obbtree.
                code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, idsVTKintersection)

                # this is the collection of intersection points from that ray in the mesh.  we are only going to choose the first one.
                pointsVTKIntersectionData = pointsVTKintersection.GetData()
                # this is the number of points interested with the line.
                noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

                if noPointsVTKIntersection != 0:
                    pcMat.append(pointsVTKIntersectionData.GetTuple3(0))

        pcMat = numpy.array(pcMat)
        pcMat = numpy.transpose(pcMat)

        return pcMat




    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    #
    #This returns a list of pixels that define the outline of a mesh
    # asamesh
    #
    def getMeshOutline(self, TwoDMarkedPoints2, datOrdAlt, cameraMatrix1, distCoeffs1, obbTree):

        # this is the attempt to find the outline of the mesh  from the circle.
        ##########################################

        # so we can use a list of all the verticies as the pcMat
        # this will be esier to do.


        ## for bounding the squid
        twoAve = numpy.zeros(2)
        for two in range(len(TwoDMarkedPoints2)):
            twoAve[0] += TwoDMarkedPoints2[two][0]
            twoAve[1] += TwoDMarkedPoints2[two][1]
        twoAve = twoAve / len(TwoDMarkedPoints2)

        twoLen = numpy.zeros((len(TwoDMarkedPoints2)))
        for two in range(len(TwoDMarkedPoints2)):
            twoLen[two] = numpy.linalg.norm(twoAve - TwoDMarkedPoints2[two])

        maxtwodist = 2 * numpy.max(twoLen)

        #CircleMax  is this the boundig circle around the squid?
        CircleMax = 50 # this for squid.  just the number of points

        CircleArray = numpy.zeros((CircleMax, 2))
        maxtwodist = 2 * numpy.max(twoLen)
        print maxtwodist,"maxtwodist"
        print twoAve,"twoAve"
        for two in range(CircleMax):
            CircleArray[two][0] = maxtwodist * numpy.cos(two * 2 * 3.14159 / CircleMax) + twoAve[0]
            CircleArray[two][1] = maxtwodist * numpy.sin(two * 2 * 3.14159 / CircleMax) + twoAve[1]

        OutlineInterations = 100
        Outline = numpy.zeros((CircleMax, 2))

        twolegacy = 0
        Ptarlen=10000
        for two in range(CircleMax):
            if two == 0:
                for three in range(OutlineInterations):
                    interpoint = CircleArray[two] + three * (twoAve - CircleArray[two]) / OutlineInterations

                    CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1, distCoeffs1)

                    pSource = CamOrgn23
                    pTarget = CamOrgn23 + Ptarlen * pointvect23
                    pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                    if not pointsIntersection:
                        None
                    else:
                        print three,"three"
                        Outline[two] = CircleArray[two] + (three - 1) * (twoAve - CircleArray[two]) / OutlineInterations
                        twolegacy = three - 1
                        break

            else:
                interpoint = CircleArray[two] + twolegacy * (twoAve - CircleArray[two]) / OutlineInterations

                CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1, distCoeffs1)

                pSource = CamOrgn23
                pTarget = CamOrgn23 + Ptarlen * pointvect23
                pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                if not pointsIntersection:
                    for three in range(twolegacy, OutlineInterations):
                        interpoint = CircleArray[two] + three * (twoAve - CircleArray[two]) / OutlineInterations

                        CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1,
                                                                 distCoeffs1)

                        pSource = CamOrgn23
                        pTarget = CamOrgn23 + Ptarlen * pointvect23
                        pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                        if not pointsIntersection:
                            None
                        else:
                            Outline[two] = CircleArray[two] + (three - 1) * (
                                twoAve - CircleArray[two]) / OutlineInterations
                            twolegacy = three - 1
                            break

                else:
                    for three in reversed(range(0, twolegacy)):
                        interpoint = CircleArray[two] + three * (twoAve - CircleArray[two]) / OutlineInterations

                        CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1,
                                                                 distCoeffs1)
                        pSource = CamOrgn23
                        pTarget = CamOrgn23 + Ptarlen * pointvect23
                        pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                        if not pointsIntersection:
                            Outline[two] = CircleArray[two] + (three) * (
                                twoAve - CircleArray[two]) / OutlineInterations
                            twolegacy = three
                            break
                        else:
                            None

        return CircleArray, Outline, twoAve




#######??????????????????????????


        ##############################3
        #### trying to adjust the roll of the squid to get zero roll.
        ####
        #asamesh

    def AdjustRolltoMinimizeRoll(self, thetaY):

        thetaY = thetaY * 3.14159 / 180
        datOrdAlt = numpy.copy(self.datOrdAltGlobe)

        zvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        camOrgn = numpy.zeros(3)

        xvect1[0] = datOrdAlt[0]
        xvect1[1] = datOrdAlt[1]
        xvect1[2] = datOrdAlt[2]

        yvect1[0] = datOrdAlt[3]
        yvect1[1] = datOrdAlt[4]
        yvect1[2] = datOrdAlt[5]

        zvect1[0] = datOrdAlt[6]
        zvect1[1] = datOrdAlt[7]
        zvect1[2] = datOrdAlt[8]

        camOrgn[0] = datOrdAlt[9]
        camOrgn[1] = datOrdAlt[10]
        camOrgn[2] = datOrdAlt[11]

        zhat = numpy.array([0, 0, 1.0])
        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926

        xvect1R = numpy.zeros(3)
        yvect1R = numpy.zeros(3)
        zvect1R = numpy.zeros(3)
        camOrgnR = numpy.zeros(3)
        thetaY=thetaY[0]

        RotAroundY = [[1.0, 0.0, 0.0], [0, numpy.cos(thetaY), numpy.sin(thetaY)],
                      [0.0, -1 * numpy.sin(thetaY), numpy.cos(thetaY)]]

        RotAroundY = [[numpy.cos(thetaY), 0.0, numpy.sin(thetaY)], [0.0, 1.0, 0.0],
                      [-1.0 * numpy.sin(thetaY), 0.0, numpy.cos(thetaY)]]

        RotAroundY = numpy.array(RotAroundY)
        xvect1R = numpy.matmul(RotAroundY, xvect1)
        yvect1R = numpy.matmul(RotAroundY, yvect1)
        zvect1R = numpy.matmul(RotAroundY, zvect1)
        camOrgnR = numpy.matmul(RotAroundY, camOrgn)

        camHrizontal = numpy.cross(zvect1R, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1R, camHrizontal) / (
            numpy.linalg.norm(xvect1R) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926
        return CameraRoll

            #####################################################################################################3333





        ####AdjustPitchtoMinimizeRoll is for minimizing the roll,
        # and AdjustPitchtoMinimizeRollout is giving the output of that.

            ##############################3
        #### trying to adjust the pitch of the squid to get zero roll.
        ### asamesh

    def AdjustPitchtoMinimizeRoll(self, thetaY):

        thetaY = thetaY * 3.14159 / 180
        datOrdAlt = numpy.copy(self.datOrdAltGlobe)

        zvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        camOrgn = numpy.zeros(3)

        xvect1[0] = datOrdAlt[0]
        xvect1[1] = datOrdAlt[1]
        xvect1[2] = datOrdAlt[2]

        yvect1[0] = datOrdAlt[3]
        yvect1[1] = datOrdAlt[4]
        yvect1[2] = datOrdAlt[5]

        zvect1[0] = datOrdAlt[6]
        zvect1[1] = datOrdAlt[7]
        zvect1[2] = datOrdAlt[8]

        camOrgn[0] = datOrdAlt[9]
        camOrgn[1] = datOrdAlt[10]
        camOrgn[2] = datOrdAlt[11]
        zhat = numpy.array([0, 0, 1.0])
        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926

        xvect1R = numpy.zeros(3)
        yvect1R = numpy.zeros(3)
        zvect1R = numpy.zeros(3)
        camOrgnR = numpy.zeros(3)

        RotAroundY = [[numpy.cos(thetaY), 0, numpy.sin(thetaY)], [0.0, 1.0, 0.0],
                      [-1 * numpy.sin(thetaY), 0.0, numpy.cos(thetaY)]]
        RotAroundY = [[1.0, 0.0, 0.0], [0, numpy.cos(thetaY), numpy.sin(thetaY)],
                      [0.0, -1 * numpy.sin(thetaY), numpy.cos(thetaY)]]

        RotAroundY = numpy.array(RotAroundY)
        xvect1R = numpy.matmul(RotAroundY, xvect1)
        yvect1R = numpy.matmul(RotAroundY, yvect1)
        zvect1R = numpy.matmul(RotAroundY, zvect1)
        camOrgnR = numpy.matmul(RotAroundY, camOrgn)

        camHrizontal = numpy.cross(zvect1R, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1R, camHrizontal) / (
            numpy.linalg.norm(xvect1R) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926
        return CameraRoll

        #####################################################################################################3333






        ##############################3
        #### trying to adjust the pitch of the squid to get zero roll.
        #asamesh
    def AdjustPitchtoMinimizeRollOut(self, thetaY):

        thetaY = thetaY * 3.14159 / 180
        datOrdAlt = numpy.copy(self.datOrdAltGlobe)

        zvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        camOrgn = numpy.zeros(3)

        xvect1[0] = datOrdAlt[0]
        xvect1[1] = datOrdAlt[1]
        xvect1[2] = datOrdAlt[2]

        yvect1[0] = datOrdAlt[3]
        yvect1[1] = datOrdAlt[4]
        yvect1[2] = datOrdAlt[5]

        zvect1[0] = datOrdAlt[6]
        zvect1[1] = datOrdAlt[7]
        zvect1[2] = datOrdAlt[8]

        camOrgn[0] = datOrdAlt[9]
        camOrgn[1] = datOrdAlt[10]
        camOrgn[2] = datOrdAlt[11]

        zhat = numpy.array([0, 0, 1.0])
        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926


        #Rotation around Y???
        RotAroundY = [[numpy.cos(thetaY), 0, numpy.sin(thetaY)], [0.0, 1.0, 0.0],
                      [-1 * numpy.sin(thetaY), 0.0, numpy.cos(thetaY)]]
        RotAroundY = [[1.0, 0.0, 0.0], [0, numpy.cos(thetaY), numpy.sin(thetaY)],
                      [0.0, -1 * numpy.sin(thetaY), numpy.cos(thetaY)]]

        RotAroundY = numpy.array(RotAroundY)
        xvect1R = numpy.matmul(RotAroundY, xvect1)
        yvect1R = numpy.matmul(RotAroundY, yvect1)
        zvect1R = numpy.matmul(RotAroundY, zvect1)
        camOrgnR = numpy.matmul(RotAroundY, camOrgn)

        #the new camera location
        datOrdAltAlt = numpy.zeros(12)

        datOrdAltAlt[0] = xvect1R[0]
        datOrdAltAlt[1] = xvect1R[1]
        datOrdAltAlt[2] = xvect1R[2]
        datOrdAltAlt[3] = yvect1R[0]
        datOrdAltAlt[4] = yvect1R[1]
        datOrdAltAlt[5] = yvect1R[2]
        datOrdAltAlt[6] = zvect1R[0]
        datOrdAltAlt[7] = zvect1R[1]
        datOrdAltAlt[8] = zvect1R[2]

        # camera origin
        datOrdAltAlt[9] = camOrgnR[0]
        datOrdAltAlt[10] = camOrgnR[1]
        datOrdAltAlt[11] = camOrgnR[2]

        camHrizontal = numpy.cross(zvect1R, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)

        #camera roll??? or pitch?
        CameraRoll = numpy.arccos(numpy.dot(xvect1R, camHrizontal) / (
            numpy.linalg.norm(xvect1R) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926
        return CameraRoll, RotAroundY, datOrdAltAlt






        ##############################3
        #### trying to adjust the roll of the squid to get zero camera roll.
        ### asamesh
    def AdjustRolltoMinimizeRollOut(self, thetaY):

        thetaY = thetaY * 3.14159 / 180
        datOrdAlt = numpy.copy(self.datOrdAltGlobe)

        zvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        camOrgn = numpy.zeros(3)

        xvect1[0] = datOrdAlt[0]
        xvect1[1] = datOrdAlt[1]
        xvect1[2] = datOrdAlt[2]

        yvect1[0] = datOrdAlt[3]
        yvect1[1] = datOrdAlt[4]
        yvect1[2] = datOrdAlt[5]

        zvect1[0] = datOrdAlt[6]
        zvect1[1] = datOrdAlt[7]
        zvect1[2] = datOrdAlt[8]

        camOrgn[0] = datOrdAlt[9]
        camOrgn[1] = datOrdAlt[10]
        camOrgn[2] = datOrdAlt[11]

        zhat = numpy.array([0, 0, 1.0])
        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926

        #Rotation around Y???
        RotAroundY = [[numpy.cos(thetaY), 0, numpy.sin(thetaY)], [0.0, 1.0, 0.0],
                      [-1 * numpy.sin(thetaY), 0.0, numpy.cos(thetaY)]]

        RotAroundY = numpy.array(RotAroundY)
        xvect1R = numpy.matmul(RotAroundY, xvect1)
        yvect1R = numpy.matmul(RotAroundY, yvect1)
        zvect1R = numpy.matmul(RotAroundY, zvect1)
        camOrgnR = numpy.matmul(RotAroundY, camOrgn)

        #the new camera location
        datOrdAltAlt = numpy.zeros(12)

        datOrdAltAlt[0] = xvect1R[0]
        datOrdAltAlt[1] = xvect1R[1]
        datOrdAltAlt[2] = xvect1R[2]
        datOrdAltAlt[3] = yvect1R[0]
        datOrdAltAlt[4] = yvect1R[1]
        datOrdAltAlt[5] = yvect1R[2]
        datOrdAltAlt[6] = zvect1R[0]
        datOrdAltAlt[7] = zvect1R[1]
        datOrdAltAlt[8] = zvect1R[2]

        # camera origin
        datOrdAltAlt[9] = camOrgnR[0]
        datOrdAltAlt[10] = camOrgnR[1]
        datOrdAltAlt[11] = camOrgnR[2]

        camHrizontal = numpy.cross(zvect1R, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)

        #camera roll??? or pitch?
        CameraRoll = numpy.arccos(numpy.dot(xvect1R, camHrizontal) / (
            numpy.linalg.norm(xvect1R) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926

        return CameraRoll, RotAroundY, datOrdAltAlt






    #####################################################################################################3333
    #
    #extracting the angle from color map.
    #
    def ReturnAngleAndIntenOfBkgnd(self,bitmap1):
        saveFile = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/ImagesForSquidAnalysis/AolpIMUGraphs/"
        if 0:#making the color bar... We use ful image for this not cropped image.
            colorBar = numpy.zeros(((653 - 328), 3))
            for gg in range(len(colorBar)):
                colorBar[gg][:] = bitmap1[328 + gg][1196][:]
            numpy.save(saveFile + "colorBar" + ".npy", colorBar)
        if 1:#loading the color bar
            colorBar=numpy.load(saveFile + "colorBar" + ".npy", colorBar)
        aolpboxx1 = 890
        aolpboxy1 = 606
        aolpboxx2 = 958
        aolpboxy2 = 652
        color2ave = numpy.zeros(3)
        color2aveIndex = 0
        for gg in range(aolpboxx1, aolpboxx2):
            for hh in range(aolpboxy1, aolpboxy2):
                if bitmap1[hh][gg][0] > 5 and bitmap1[hh][gg][1] > 5 and bitmap1[hh][gg][2] > 5:
                    color2ave += bitmap1[hh][gg][:]
                    color2aveIndex += 1
        color2ave = color2ave / float(color2aveIndex)

        colorBardiff = numpy.zeros((len(colorBar)))
        for gg in range(len(colorBar)):
            colorBardiff[gg] = numpy.linalg.norm(color2ave - colorBar[gg])
        aolpOfBackground = numpy.argmin(colorBardiff) * 180 / float(326)
        aolpboxx1 = 566
        aolpboxy1 = 258
        aolpboxx2 = 639
        aolpboxy2 = 316
        color2ave = numpy.zeros(3)
        color2aveIndex = 0
        for gg in range(aolpboxx1, aolpboxx2):
            for hh in range(aolpboxy1, aolpboxy2):
                if 1:
                    color2ave += bitmap1[hh][gg][:]
                    color2aveIndex += 1
        color2ave = color2ave / float(color2aveIndex)
        greenAve = color2ave[1]
        return aolpOfBackground,greenAve

    #####################################################################################################3333
    #####################################################################################################3333
    #####################################################################################################3333
    #####################################################################################################3333
    #####################################################################################################3333
    #####################################################################################################3333

    def ReturnAngleArrayOfBkgnd(self,bitmap1,dat1, cameraMatrix1, distCoeffs1,frame1num):
        saveFile = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/ImagesForSquidAnalysis/AolpIMUGraphs/"

        #we are looking at the colorBar on the image
        if 0:#making the color bar... We use ful image for this not cropped image.
            colorBar = numpy.zeros(((653 - 328), 3))
            for gg in range(len(colorBar)):
                colorBar[gg][:] = bitmap1[328 + gg][1196][:]
            numpy.save(saveFile + "colorBar" + ".npy", colorBar)
        if 1:#loading the color bar
            colorBar=numpy.load(saveFile + "colorBar" + ".npy")

        Relative_Angle_Array=[]
        Relative_Angle_Array_Rads = []
        aop_observed_rads_Array = []


        ### different values used for this.
        pixelX=numpy.array(range(4))*200+192
        pixelY=numpy.array(range(2))*200+112

        #pixelX=numpy.array(range(1))*800+192+400
        #pixelY=numpy.array(range(2))*400+112

        #pixelX=numpy.array(range(1))*800+192+400
        #pixelY=numpy.array(range(1))*400+112+200

        #pixelX = numpy.array(range(1)) * 800 + 192+400
        #pixelY = numpy.array(range(3)) * 200 + 112+0
        #pixelY = numpy.array(range(1)) * 200 + 112+400
        #pixelY = numpy.array(range(10)) * 40 + 112+0



        dat1=[ 1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0, 0]
        print "dat1",dat1
        for rr in range(len(pixelX)):
            for ss in range(len(pixelY)):
                CamOrgn1, pointvect1 = self.theVectors(dat1, [pixelX[rr],pixelY[ss]], cameraMatrix1, distCoeffs1)
                pointvect1=numpy.array(pointvect1)
                zvector=numpy.array([0,  0,  1])
                xvector = numpy.array([1, 0, 0])
                yvector = numpy.array([0, 1, 0])

                print "x",pixelX[rr], numpy.arctan(xvector.dot(pointvect1) / zvector.dot(pointvect1))*180/3.14159
                print "y",pixelY[ss], numpy.arctan(-yvector.dot(pointvect1) / zvector.dot(pointvect1))*180/3.14159

                Relative_Angle_Array.append([numpy.arctan(xvector.dot(pointvect1) / zvector.dot(pointvect1))*180/3.14159,
                                             numpy.arctan(
                                                 -yvector.dot(pointvect1) / zvector.dot(pointvect1)) * 180 / 3.14159])

                Relative_Angle_Array_Rads.append([numpy.arctan(xvector.dot(pointvect1) / zvector.dot(pointvect1)),
                                             numpy.arctan(
                                                 -yvector.dot(pointvect1) / zvector.dot(pointvect1))])



                # this is the box to get aolp
                aolpboxx1 = pixelX[rr]-10
                aolpboxy1 = pixelY[ss]-10
                aolpboxx2 = pixelX[rr]+10
                aolpboxy2 = pixelY[ss]+10
                color2ave = numpy.zeros(3)
                color2aveIndex = 0

                #######################   just  getting the color average for the whole box
                for gg in range(aolpboxx1, aolpboxx2):
                    for hh in range(aolpboxy1, aolpboxy2):
                        if bitmap1[hh][gg][0] > 5 and bitmap1[hh][gg][1] > 5 and bitmap1[hh][gg][2] > 5:
                            color2ave += bitmap1[hh][gg][:]
                            color2aveIndex += 1
                color2ave = color2ave / float(color2aveIndex)


                #####  Minimizing it with the color bar.
                colorBardiff = numpy.zeros((len(colorBar)))
                for gg in range(len(colorBar)):
                    colorBardiff[gg] = numpy.linalg.norm(color2ave - colorBar[gg])



                #  what is 326 the length of the color bar?
                aolpOfBackground = numpy.argmin(colorBardiff) * 180 / float(326)

                print "aop",aolpOfBackground

                #aop_observed_rads_Array.append(aolpOfBackground*3.14159/180)

                if 1:   # trying to adjust for the spherical abboration.
                    point2=[0,0]

                    pointHypont=3
                    point2[0]=pointHypont*numpy.cos(aolpOfBackground*3.14159/180)+pixelX[rr]
                    point2[1] = pointHypont * numpy.sin(aolpOfBackground * 3.14159 / 180)+pixelY[ss]

                    # so the new goalvect will probably just be this:
                    test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
                    test[0][0][0] = point2[0]
                    test[0][0][1] = point2[1]

                    undistort = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)
                    # print undistort

                    UdistortVect2 = numpy.zeros(2)

                    UdistortVect2[0] = undistort[0][0][0]
                    UdistortVect2[1] = undistort[0][0][1]



                    # so the new goalvect will probably just be this:
                    test = numpy.zeros((1, 1, 2), dtype=numpy.float32)
                    test[0][0][0] = pixelX[rr]
                    test[0][0][1] = pixelY[ss]

                    undistort = cv2.undistortPoints(test, cameraMatrix1, distCoeffs1)

                    UdistortVectPixel = numpy.zeros(2)

                    UdistortVectPixel[0] = undistort[0][0][0]
                    UdistortVectPixel[1] = undistort[0][0][1]


                    aolpOfBackgroundadj=(180/3.14159)*numpy.arctan((UdistortVectPixel[1]-UdistortVect2[1])/(UdistortVectPixel[0]-UdistortVect2[0]))


                    if aolpOfBackgroundadj<0:
                        aolpOfBackgroundadj=aolpOfBackgroundadj+180
                    print "adjusted angle ",aolpOfBackgroundadj


                    aop_observed_rads_Array.append(aolpOfBackgroundadj * 3.14159 / 180)


        if 1:
            print Simulator_Helper_pb.EstimateHeading_UnknownElevation_with_Array.estimate_heading_and_elevation(
                aop_observed_rads_Array, Relative_Angle_Array_Rads, 0, 0, 0 * 3.14159 / 180, 42 * 3.14159 / 180, .5*3.14159 / 180,
                0.0)


        if 0:  #ploting the value   Using simulator helper
            optimization_variables=(.3,.2)
            xa=100
            ya=50
            AziAnglesForVisualization=numpy.array(range(xa))
            MinimizationVisualization=numpy.zeros((xa,ya))
            pitchAnglesForVisualization=numpy.array(range(ya))

            AziAnglesForVisualization=AziAnglesForVisualization*2*3.14159/xa
            pitchAnglesForVisualization=pitchAnglesForVisualization*3.14159/ya-.5*3.14159

            for ss in range(xa):
                for rr in range(ya):
                    optimization_variables=(AziAnglesForVisualization[ss],pitchAnglesForVisualization[rr])
                    MinimizationVisualization[ss][rr]=Simulator_Helper_pb.EstimateHeading_UnknownElevation_with_Array.cost_function(optimization_variables,
                        0,0,0, aop_observed_rads_Array,Relative_Angle_Array_Rads)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(MinimizationVisualization, cmap='terrain',clim=(0,.1))
            fig.savefig(saveFile + '/' +"minMap_"+str(pixelX[0])+"_"+str(pixelY[0])+"_"+ str(int(frame1num)) + '.png',
                        bbox_inches='tight', dpi=300)


            plt.show()


        #return aolpOfBackground


########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################

    def goingThoughVariousCameraParameters(self,fwData, frame1num, datOrdAlt, datOrdAltAlt, CameraRoll, RotAroundY,resx0,RotMatCopy):

        # Here is a mess of calculating various parameters.
        # cameraOrginW === the noe origin of the minimized DatOrdAlt

        cameraOrginW = numpy.array([datOrdAltAlt[9], datOrdAltAlt[10], datOrdAltAlt[11]])
        Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
        cameraZ = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], datOrdAltAlt[8]])
        cameraZX = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], 0])
        cameraPitch = (180 / 3.1415926) * numpy.arccos(
            numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))
        cameraPitchFV = cameraPitch
        print "squid pitch", resx0 # zxz
        print "camera pitch", cameraPitch
        print "Asimuth", Asimuth
        print "Distance", numpy.linalg.norm(cameraOrginW)

        # Here we revisit this with datOrdAlt insted
        # doing what we did above  Nonadjusted
        cameraOrginW = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])
        Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
        cameraZ = numpy.array([datOrdAlt[6], datOrdAlt[7], datOrdAlt[8]])
        cameraZX = numpy.array([datOrdAlt[6], datOrdAlt[7], 0])
        cameraPitch = (180 / 3.1415926) * numpy.arccos(
            numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))

        ####################
        zvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)

        xvect1[0] = datOrdAlt[0]
        xvect1[1] = datOrdAlt[1]
        xvect1[2] = datOrdAlt[2]

        yvect1[0] = datOrdAlt[3]
        yvect1[1] = datOrdAlt[4]
        yvect1[2] = datOrdAlt[5]

        zvect1[0] = datOrdAlt[6]
        zvect1[1] = datOrdAlt[7]
        zvect1[2] = datOrdAlt[8]
        # these are the non adjusted camera vectors.



        ## this is the up....
        zhat = numpy.array([0, 0, 1.0])

        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)

        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926

        print "squid pitch", 0
        print "camera pitch orig", cameraPitch
        print "Asimuth orig", Asimuth
        print "Distance orig", numpy.linalg.norm(cameraOrginW)
        print "camera Roll", CameraRoll

        ####################
        # we want to look at the adjusted rotation matrix sometimes.
        if 0:
            RotMatCopy[0, 0] = datOrdAltAlt[0]
            RotMatCopy[1, 0] = datOrdAltAlt[1]
            RotMatCopy[2, 0] = datOrdAltAlt[2]

            RotMatCopy[0, 1] = datOrdAltAlt[3]
            RotMatCopy[1, 1] = datOrdAltAlt[4]
            RotMatCopy[2, 1] = datOrdAltAlt[5]

            RotMatCopy[0, 2] = datOrdAltAlt[6]
            RotMatCopy[1, 2] = datOrdAltAlt[7]
            RotMatCopy[2, 2] = datOrdAltAlt[8]

        if 1:  # Getting the camera IMU
            try:
                RealHeading = fwData["direction"][str(frame1num)]["Real IMU Solar Heading"].value
                # the Roll and the pitch are switched
                RealRoll = fwData["direction"][str(frame1num)]["Real IMU camera pitch"].value
                RealPitch = fwData["direction"][str(frame1num)]["Real IMU camera roll"].value
            except:
                print "no IMU"

            # we are adjusting the pitch
            RealPitch = RealPitch - 10
            RealRoll = RealRoll
            print RealHeading, "RealHeading"
            print RealPitch, "RealPitch"
            print RealRoll, "RealRoll"

            # now to make vectors out of the cameras IMU data.

            cz = numpy.array([-numpy.cos(RealPitch * 3.14159 / 180) * numpy.cos(RealHeading * 3.14159 / 180),
                              numpy.cos(RealPitch * 3.14159 / 180) * numpy.sin(RealHeading * 3.14159 / 180),
                              numpy.sin(RealPitch * 3.14159 / 180)])
            cx = numpy.array([-numpy.sin(RealHeading * 3.14159 / 180),
                              -numpy.cos(RealHeading * 3.14159 / 180),
                              0.0])
            cx = -cx
            cy = numpy.array([numpy.sin(RealPitch * 3.14159 / 180) * numpy.cos(RealHeading * 3.14159 / 180),
                              -numpy.sin(RealPitch * 3.14159 / 180) * numpy.sin(RealHeading * 3.14159 / 180),
                              numpy.cos(RealPitch * 3.14159 / 180)])
            cy = -cy
            # we have made the adjustments that make these vectors work in our system.




            ###### Geting the rotation from the Roll
            RollRotMat = self.rotation_matrix(cz, RealRoll * 3.14159 / 180)
            cx = numpy.matmul(RollRotMat, cx)
            cy = numpy.matmul(RollRotMat, cy)

            # Getting the rotation matrix for the IMU vectors
            RotMatIMU = numpy.zeros((3, 3))

            RotMatIMU[0, 0] = cx[0]
            RotMatIMU[1, 0] = cx[1]
            RotMatIMU[2, 0] = cx[2]
            RotMatIMU[0, 1] = cy[0]
            RotMatIMU[1, 1] = cy[1]
            RotMatIMU[2, 1] = cy[2]
            RotMatIMU[0, 2] = cz[0]
            RotMatIMU[1, 2] = cz[1]
            RotMatIMU[2, 2] = cz[2]

            zhatTrans = numpy.zeros(3)
            yhatTrans = numpy.zeros(3)
            xhatTrans = numpy.zeros(3)

            xhat = numpy.array([1.0, 0, 0])
            yhat = numpy.array([0, -1.0, 0])
            print "???????????????????????????????????????????????????????????????????"
            rot3 = numpy.array([RotMatCopy[0, 2], RotMatCopy[1, 2], RotMatCopy[2, 2]])

            # Here we are going from the camera frame to the IMU frame
            if 1:
                zhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), zhat))
                yhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), yhat))
                xhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), xhat))

            ### is any of this recorded any where???

            # this is the heading in the IMU frame....
            fhead = numpy.arctan2(yhatTrans[1], -yhatTrans[0]) * 180 / 3.14159
            if fhead < 0:
                fhead = fhead + 360
            print "fish heading", fhead

            yhatTransnonZ = numpy.array([yhatTrans[0], yhatTrans[1], 0])
            print "fish pitch", numpy.arctan(yhatTrans[2] / numpy.linalg.norm(yhatTransnonZ)) * 180 / 3.14159
            print zhatTrans, "zhatTrans"
            yzcross = numpy.cross(numpy.cross(yhatTrans, zhat), yhatTrans)
            yzcross = yzcross / numpy.linalg.norm(yzcross)
            fishRoll = numpy.arccos(numpy.dot(yzcross, zhatTrans) / (
                numpy.linalg.norm(yzcross) * numpy.linalg.norm(zhatTrans))) * 180 / 3.14159
            ####????

            # the sign...
            fishRoll = fishRoll * numpy.dot(
                numpy.cross(zhatTrans, yzcross) / numpy.linalg.norm(numpy.cross(zhatTrans, yzcross)),
                yhatTrans / numpy.linalg.norm(yhatTrans))
            print "coplanar", numpy.dot(
                numpy.cross(zhatTrans, yzcross) / numpy.linalg.norm(numpy.cross(zhatTrans, yzcross)),
                yhatTrans / numpy.linalg.norm(yhatTrans))

            print "coplanar", numpy.dot(
                numpy.cross(zhatTrans, yzcross),
                yhatTrans)

            print "fish Roll", fishRoll

            print "plus pitches", cameraPitchFV + RealPitch
            print "minus pitches", -cameraPitchFV + RealPitch

            print "plus pitches orign", cameraPitch + RealPitch
            print "minus pitches orign", -cameraPitch + RealPitch

            # finding fishRelAzimuth with the yvector and the camera vector.
            fishRelAzimuth = numpy.arccos(numpy.dot(cz, yhatTrans) / (
                numpy.linalg.norm(cz) * numpy.linalg.norm(yhatTrans))) * 180 / 3.14159
            ####????

            # the sign...
            azisign = numpy.dot(
                numpy.cross(cz, yhatTrans),
                zhatTrans)
            if azisign < 0:
                fishRelAzimuth = fishRelAzimuth * -1.0

            print "Fish relative azimuth", fishRelAzimuth
            print fwData["direction"][str(frame1num)]["TorA"][0]

            fishPitch=numpy.arctan(yhatTrans[2] / numpy.linalg.norm(yhatTransnonZ)) * 180 / 3.14159

        return fishRoll,fishRelAzimuth,fishPitch






    ######################################################3
    #
    # Main PNP routene
    #asamesh
    #
    def PnPtoFacePoints(self,ThreeDMarkedPoints,fwData,cameraNames,moviewatch,frame1num,cameraMatrix1, distCoeffs1,pcMat,Tobject,obbTree,UseRandom,insectnumit):
            # here is the group of 3D points I think

        print ThreeDMarkedPoints,"ThreeDMarkedPoints"
        #We use a randomization of this technique so that we can get an estimate of the error.
        if UseRandom==True:
            PathWrite = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/RandomPnP.txt"
            if os.path.isfile(PathWrite):
                PathWrite = open(PathWrite, "a")
            else:
                PathWrite = open(PathWrite, "w")

        TwoDMarkedPoints = numpy.zeros((len(ThreeDMarkedPoints), 2))
        TwoDMarkedPoints2 = []
        ThreeDMarkedPoints2 = []


        # We are matching the 2D pixel points with the 3D points.   So we are going though the 3D points to find those that are
        # present.

        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if UseRandom==False:
                    TwoDMarkedPoints2.append(
                        [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                         fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                else:
                    #this variable is the random spred for the PnP error
                    pixelerrsize=10
                    if 1:

                        if 1:
                            TwoDMarkedPoints2.append(
                                [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0]+numpy.random.random_sample()*pixelerrsize,
                                 fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]+numpy.random.random_sample()*pixelerrsize])
                        else:
                            TwoDMarkedPoints2.append(
                                [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                                 fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])


                ThreeDMarkedPoints2.append(ThreeDMarkedPoints[i])
            except:
                None

        TwoDMarkedPoints = numpy.array(TwoDMarkedPoints)
        ThreeDMarkedPoints2 = numpy.array(ThreeDMarkedPoints2)
        TwoDMarkedPoints2 = numpy.array(TwoDMarkedPoints2)

        #This is the open cv PnP algorithim
        print ThreeDMarkedPoints2,TwoDMarkedPoints2,"ThreeDMarkedPoints2,TwoDMarkedPoints2,"

        rvec = cv2.solvePnPRansac(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1)[1]
        tvec = cv2.solvePnPRansac(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1)[2]



        if 1:#Here we are just using solvePnP
            rvec = cv2.solvePnP(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1, rvec=rvec,
                                tvec=tvec, useExtrinsicGuess=1)[1]
            tvec = \
                cv2.solvePnP(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1, rvec=rvec,
                             tvec=tvec,
                             useExtrinsicGuess=1)[2]

        ###############################################################################################################
        # changing rvec to a matrix
        rvec2Mat = numpy.zeros((3, 3))
        dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

        RotMat = dst
        # using the inverse: (Rt-RtT)
        RotMat = numpy.matrix.transpose(dst)
        TranMat = (-1) * numpy.matmul(RotMat, tvec)

        datOrdAlt = numpy.zeros(12)
        RotMatCopy = numpy.copy(RotMat)
        # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

        datOrdAlt[0] = RotMat[0, 0]
        datOrdAlt[1] = RotMat[1, 0]
        datOrdAlt[2] = RotMat[2, 0]
        datOrdAlt[3] = RotMat[0, 1]
        datOrdAlt[4] = RotMat[1, 1]
        datOrdAlt[5] = RotMat[2, 1]
        datOrdAlt[6] = RotMat[0, 2]
        datOrdAlt[7] = RotMat[1, 2]
        datOrdAlt[8] = RotMat[2, 2]

        # camera origin
        datOrdAlt[9] = TranMat[0]
        datOrdAlt[10] = TranMat[1]
        datOrdAlt[11] = TranMat[2]

        #With this routine we are getting the outline of the mesh in the frame
        CircleArray, Outline, twoAve = self.getMeshOutline(TwoDMarkedPoints2, datOrdAlt, cameraMatrix1,
                                                           distCoeffs1, obbTree)

        # projecting the new mesh????
        newProjMesh = []
        pcWid, pcLen = pcMat.shape
        usePnpProj = True

        #yes projecting the new mesh
        for ff in range(pcLen):
            d3test = numpy.zeros((1, 1, 3), dtype=numpy.float32)

            d3test[0][0][0] = pcMat[0][ff]
            d3test[0][0][1] = pcMat[1][ff]
            d3test[0][0][2] = pcMat[2][ff]

            (imagePoints2, jacobian) = cv2.projectPoints(d3test, rvec,
                                                         tvec, cameraMatrix1, distCoeffs1)
            newProjMesh.append([int(imagePoints2[0][0][0]), int(imagePoints2[0][0][1])])




# can we put this in its own subroutene



        #inputs
        #total routine:

        #ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num, cameraMatrix1, distCoeffs1, pcMat, Tobject, obbTree, UseRandom, insectnumit

        #for this section
        # datOrdAlt
        #fwData,frame1num


        #outputs:
        #total
        #newProjMesh,datOrdAlt,CameraRoll, RotAroundY, datOrdAltAlt,ThreeDMarkedPoints2,TwoDMarkedPoints2,Outline,fishRoll
        #from this section
        #CameraRoll, RotAroundY, datOrdAltAlt,fishRoll

        #for random error calculation:
            #fishRelAzimuth,fishRoll
            #"fish pitch ???? numpy.arctan(yhatTrans[2] / numpy.linalg.norm(yhatTransnonZ)) * 180 / 3.14159) + "\n")

            ######

        ########################################################################3############
        ##############################3

        # we use constraints to aid in the fitting of the mesh with the image.

        #### trying to adjust the pitch of the squid to get zero roll.
        self.datOrdAltGlobe = numpy.copy(datOrdAlt)
        res = minimize(self.AdjustPitchtoMinimizeRoll, 0, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})


        ##### Here we are getting the specific values. For fish we don't really use the pitch minimization
        # we use pitch minimization for squid, because of the geomtry of the problem.

        CameraRoll, RotAroundY, datOrdAltAlt = self.AdjustPitchtoMinimizeRollOut(res.x[0])
        resx0=res.x[0]
        fishRoll, fishRelAzimuth, fishPitch=self.goingThoughVariousCameraParameters(fwData,frame1num,datOrdAlt,datOrdAltAlt,CameraRoll, RotAroundY,resx0,RotMatCopy)


        #printing out the random results.
        if UseRandom:
            PathWrite.write("Fish relative azimuth"+";"+str(fishRelAzimuth)+";"+"fish roll"+";"+str(fishRoll)+";"+"fish pitch"+";"+str(numpy.arctan(yhatTrans[2] / numpy.linalg.norm(yhatTransnonZ)) * 180 / 3.14159)+"\n")

#######################################################################################


        return newProjMesh,datOrdAlt,CameraRoll, RotAroundY, datOrdAltAlt,ThreeDMarkedPoints2,TwoDMarkedPoints2,Outline,fishRoll






            ######################################################3
    #
    # PNP routene to get camera info if FO is in frame
    #
    #
    def PnPtoFacePointsForFOinGoPros(self, ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num, cameraMatrix1,
                        distCoeffs1, f1,f2, frameDelay):
        # here is the group of 3D points I think



        TwoDMarkedPoints = numpy.zeros((len(ThreeDMarkedPoints), 2))
        TwoDMarkedPoints2 = []
        ThreeDMarkedPoints2 = []

        # We are matching the 2D pixel points with the 3D points.   So we are going though the 3D points to find those that are
        # present.
        #print str(int(frame1num)) - frameDelay,str(int(frame1num)),"str(int(frame1num)) - frameDelay"
        Tobject="fiducialM_"
        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if 1:
                    if moviewatch==0:
                        TwoDMarkedPoints2.append(
                            [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                             fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                    if moviewatch==1:
                        TwoDMarkedPoints2.append(
                            [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][0],
                             fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][1]])

                ThreeDMarkedPoints2.append(ThreeDMarkedPoints[i])
            except:
                None

        TwoDMarkedPoints = numpy.array(TwoDMarkedPoints)
        ThreeDMarkedPoints2 = numpy.array(ThreeDMarkedPoints2)
        TwoDMarkedPoints2 = numpy.array(TwoDMarkedPoints2)
        print ThreeDMarkedPoints2,TwoDMarkedPoints2
        print len(ThreeDMarkedPoints2),len(TwoDMarkedPoints2)
        # This is the open cv PnP algorithim
        rvec = cv2.solvePnPRansac(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1)[1]
        tvec = cv2.solvePnPRansac(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1)[2]

        if 1:  # Here we are just using solvePnP
            rvec = cv2.solvePnP(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1, rvec=rvec,
                                tvec=tvec, useExtrinsicGuess=1)[1]
            tvec = \
                cv2.solvePnP(ThreeDMarkedPoints2, TwoDMarkedPoints2, cameraMatrix1, distCoeffs1, rvec=rvec,
                             tvec=tvec,
                             useExtrinsicGuess=1)[2]

        ###############################################################################################################
        # changing rvec to a matrix
        rvec2Mat = numpy.zeros((3, 3))
        dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

        RotMat = dst
        # using the inverse: (Rt-RtT)
        RotMat = numpy.matrix.transpose(dst)
        TranMat = (-1) * numpy.matmul(RotMat, tvec)

        datOrdAlt = numpy.zeros(12)
        RotMatCopy = numpy.copy(RotMat)
        # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

        datOrdAlt[0] = RotMat[0, 0]
        datOrdAlt[1] = RotMat[1, 0]
        datOrdAlt[2] = RotMat[2, 0]
        datOrdAlt[3] = RotMat[0, 1]
        datOrdAlt[4] = RotMat[1, 1]
        datOrdAlt[5] = RotMat[2, 1]
        datOrdAlt[6] = RotMat[0, 2]
        datOrdAlt[7] = RotMat[1, 2]
        datOrdAlt[8] = RotMat[2, 2]

        # camera origin
        datOrdAlt[9] = TranMat[0]
        datOrdAlt[10] = TranMat[1]
        datOrdAlt[11] = TranMat[2]


        if moviewatch==0:
            #fwheader.create_dataset('Camera1CalibrationValue', data=VikCam)
            del f1['F' + str(int(frame1num))]['CameraPos']
            f1['F' + str(int(frame1num))].create_dataset('CameraPos', data=datOrdAlt)
        if moviewatch==1:
            del f2['F' + str(int(frame1num)-frameDelay)]['CameraPos']
            f2['F' + str(int(frame1num)-frameDelay)].create_dataset('CameraPos', data=datOrdAlt)

        #return datOrdAlt,ThreeDMarkedPoints2, TwoDMarkedPoints2





            ######################################################3
    #
    # PNP routene to get camera info if FO is in frame
    #  THis is the recursion thing
    #
    def PnPtoFacePointsForFOinGoProsRecursion(self, ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num, cameraMatrix1,
                        distCoeffs1, f1,f2, frameDelay):
        # here is the group of 3D points I think



        TwoDMarkedPoints1 = []
        ThreeDMarkedPoints1 = []
        TwoDMarkedPoints2 = []
        ThreeDMarkedPoints2 = []

        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################

        #  Initial PnP


        # We are matching the 2D pixel points with the 3D points.   So we are going though the 3D points to find those that are
        # present.
        #print str(int(frame1num)) - frameDelay,str(int(frame1num)),"str(int(frame1num)) - frameDelay"
        Tobject="fiducialM_"

        # First frame
        moviewatch=0


        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if 1:
                    if moviewatch==0:
                        TwoDMarkedPoints1.append(
                            [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                             fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])

                ThreeDMarkedPoints1.append(ThreeDMarkedPoints[i])
            except:
                None


        ThreeDMarkedPointsq = numpy.array(ThreeDMarkedPoints1)
        TwoDMarkedPointsq = numpy.array(TwoDMarkedPoints1)
        #print ThreeDMarkedPoints1,TwoDMarkedPoints1
        # This is the open cv PnP algorithim
        rvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[1]
        tvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[2]

        rvec1=numpy.copy(rvec)
        tvec1=numpy.copy(tvec)

        if 1:  # Here we are just using solvePnP
            rvec = cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec,
                                tvec=tvec, useExtrinsicGuess=1)[1]
            tvec = \
                cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec,
                             tvec=tvec,
                             useExtrinsicGuess=1)[2]

        ###############################################################################################################
        # changing rvec to a matrix
        rvec2Mat = numpy.zeros((3, 3))
        dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

        RotMat = dst
        # using the inverse: (Rt-RtT)
        RotMat = numpy.matrix.transpose(dst)
        TranMat = (-1) * numpy.matmul(RotMat, tvec)

        datOrdAlt1 = numpy.zeros(12)

        # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

        datOrdAlt1[0] = RotMat[0, 0]
        datOrdAlt1[1] = RotMat[1, 0]
        datOrdAlt1[2] = RotMat[2, 0]
        datOrdAlt1[3] = RotMat[0, 1]
        datOrdAlt1[4] = RotMat[1, 1]
        datOrdAlt1[5] = RotMat[2, 1]
        datOrdAlt1[6] = RotMat[0, 2]
        datOrdAlt1[7] = RotMat[1, 2]
        datOrdAlt1[8] = RotMat[2, 2]

        # camera origin
        datOrdAlt1[9] = TranMat[0]
        datOrdAlt1[10] = TranMat[1]
        datOrdAlt1[11] = TranMat[2]



        #Second camera

        moviewatch=1
        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if 1:

                    if moviewatch == 1:
                        TwoDMarkedPoints2.append(
                            [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][0],
                             fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][1]])

                ThreeDMarkedPoints2.append(ThreeDMarkedPoints[i])
            except:
                None


        ThreeDMarkedPointsq = numpy.array(ThreeDMarkedPoints2)
        TwoDMarkedPointsq = numpy.array(TwoDMarkedPoints2)
        #print ThreeDMarkedPoints2, TwoDMarkedPoints2
        # This is the open cv PnP algorithim
        rvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[1]
        tvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[2]

        rvec2=numpy.copy(rvec)
        tvec2=numpy.copy(tvec)
        if 1:  # Here we are just using solvePnP
            rvec = cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec,
                                tvec=tvec, useExtrinsicGuess=1)[1]
            tvec = \
                cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec,
                             tvec=tvec,
                             useExtrinsicGuess=1)[2]

        ###############################################################################################################
        # changing rvec to a matrix
        rvec2Mat = numpy.zeros((3, 3))
        dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

        RotMat = dst
        # using the inverse: (Rt-RtT)
        RotMat = numpy.matrix.transpose(dst)
        TranMat = (-1) * numpy.matmul(RotMat, tvec)

        datOrdAlt2 = numpy.zeros(12)
        RotMatCopy = numpy.copy(RotMat)        # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

        datOrdAlt2[0] = RotMat[0, 0]
        datOrdAlt2[1] = RotMat[1, 0]
        datOrdAlt2[2] = RotMat[2, 0]
        datOrdAlt2[3] = RotMat[0, 1]
        datOrdAlt2[4] = RotMat[1, 1]
        datOrdAlt2[5] = RotMat[2, 1]
        datOrdAlt2[6] = RotMat[0, 2]
        datOrdAlt2[7] = RotMat[1, 2]
        datOrdAlt2[8] = RotMat[2, 2]

        # camera origin
        datOrdAlt2[9] = TranMat[0]
        datOrdAlt2[10] = TranMat[1]
        datOrdAlt2[11] = TranMat[2]



        for jjj in range(8):
            #put in + here????
            pointDic={}
            errorDic={}
            Tobject="insect"
            for insect in fwData[cameraNames[0]]:  # calling the first camera tracks insects

                if Tobject in insect:
                    try:
                        point1=numpy.zeros(2)
                        point2=numpy.zeros(2)

                        point1[0] = fwData[cameraNames[0]][insect][str(int(frame1num))][0]
                        point1[1] = fwData[cameraNames[0]][insect][str(int(frame1num))][1]

                        point2[0] = \
                            fwData[cameraNames[1]][insect][str(int(frame1num) - frameDelay)][0]
                        point2[1] = \
                            fwData[cameraNames[1]][insect][str(int(frame1num) - frameDelay)][1]


                        CamOrgn1, pointvect1 = self.theVectors(datOrdAlt1, point1, cameraMatrix1, distCoeffs1)
                        CamOrgn2, pointvect2 = self.theVectors(datOrdAlt2, point2, cameraMatrix1, distCoeffs1)

                        cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, CamOrgn2,
                                                                                  pointvect2)

                        cloPoAve = [(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                    (cloPo1[2] + cloPo2[2]) / 2]

                        cloPoError = numpy.linalg.norm(cloPo1 - cloPo2) / 2
                        errorDic[insect]=cloPoError
                        pointDic[insect]=cloPoAve
                        #print "did pass", insect
                    except:
                        None
                        #print "didn't pass", insect

            #redo PNP




            TwoDMarkedPoints1_n = TwoDMarkedPoints1[:]
            ThreeDMarkedPoints1_n = ThreeDMarkedPoints1[:]
            TwoDMarkedPoints2_n = TwoDMarkedPoints2[:]
            ThreeDMarkedPoints2_n = ThreeDMarkedPoints2[:]



            #print TwoDMarkedPoints1_n,"TwoDMarkedPoints1_n",TwoDMarkedPoints1

            inint=0
            for insect in pointDic:
                if insect=="insect21" or insect=="insect3" or insect=="insect20":
                    print pointDic[insect],errorDic[insect], insect
                    point1[0] = fwData[cameraNames[0]][insect][str(int(frame1num))][0]
                    point1[1] = fwData[cameraNames[0]][insect][str(int(frame1num))][1]
                    #print point1
                    TwoDMarkedPoints1_n.append(point1)
                    ThreeDMarkedPoints1_n.append(pointDic[insect])

                    point2[0] = \
                        fwData[cameraNames[1]][insect][str(int(frame1num) - frameDelay)][0]
                    point2[1] = \
                        fwData[cameraNames[1]][insect][str(int(frame1num) - frameDelay)][1]

                    TwoDMarkedPoints2_n.append(point2)
                    ThreeDMarkedPoints2_n.append(pointDic[insect])
                    inint+=1


            #first camera

            ThreeDMarkedPointsq = numpy.array(ThreeDMarkedPoints1_n)
            TwoDMarkedPointsq = numpy.array(TwoDMarkedPoints1_n)
            # print ThreeDMarkedPoints1,TwoDMarkedPoints1
            # This is the open cv PnP algorithim
            #rvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[1]
            #tvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[2]

            if 1:  # Here we are just using solvePnP
                rvec = cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec1,
                                    tvec=tvec1, useExtrinsicGuess=1)[1]
                tvec = \
                    cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec1,
                                 tvec=tvec1,
                                 useExtrinsicGuess=1)[2]

            ###############################################################################################################
            # changing rvec to a matrix
            rvec2Mat = numpy.zeros((3, 3))
            dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

            RotMat = dst
            # using the inverse: (Rt-RtT)
            RotMat = numpy.matrix.transpose(dst)
            TranMat = (-1) * numpy.matmul(RotMat, tvec)

            datOrdAlt1 = numpy.zeros(12)

            # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

            datOrdAlt1[0] = RotMat[0, 0]
            datOrdAlt1[1] = RotMat[1, 0]
            datOrdAlt1[2] = RotMat[2, 0]
            datOrdAlt1[3] = RotMat[0, 1]
            datOrdAlt1[4] = RotMat[1, 1]
            datOrdAlt1[5] = RotMat[2, 1]
            datOrdAlt1[6] = RotMat[0, 2]
            datOrdAlt1[7] = RotMat[1, 2]
            datOrdAlt1[8] = RotMat[2, 2]

            # camera origin
            datOrdAlt1[9] = TranMat[0]
            datOrdAlt1[10] = TranMat[1]
            datOrdAlt1[11] = TranMat[2]




            #second camera
            ThreeDMarkedPointsq = numpy.array(ThreeDMarkedPoints2_n)
            TwoDMarkedPointsq = numpy.array(TwoDMarkedPoints2_n)
            # print ThreeDMarkedPoints2, TwoDMarkedPoints2
            # This is the open cv PnP algorithim
            #rvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[1]
            #tvec = cv2.solvePnPRansac(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1)[2]

            if 1:  # Here we are just using solvePnP
                rvec = cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec2,
                                    tvec=tvec2, useExtrinsicGuess=1)[1]
                tvec = \
                    cv2.solvePnP(ThreeDMarkedPointsq, TwoDMarkedPointsq, cameraMatrix1, distCoeffs1, rvec=rvec2,
                                 tvec=tvec2,
                                 useExtrinsicGuess=1)[2]

            ###############################################################################################################
            # changing rvec to a matrix
            rvec2Mat = numpy.zeros((3, 3))
            dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

            RotMat = dst
            # using the inverse: (Rt-RtT)
            RotMat = numpy.matrix.transpose(dst)
            TranMat = (-1) * numpy.matmul(RotMat, tvec)

            datOrdAlt2 = numpy.zeros(12)
            RotMatCopy = numpy.copy(
                RotMat)  # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

            datOrdAlt2[0] = RotMat[0, 0]
            datOrdAlt2[1] = RotMat[1, 0]
            datOrdAlt2[2] = RotMat[2, 0]
            datOrdAlt2[3] = RotMat[0, 1]
            datOrdAlt2[4] = RotMat[1, 1]
            datOrdAlt2[5] = RotMat[2, 1]
            datOrdAlt2[6] = RotMat[0, 2]
            datOrdAlt2[7] = RotMat[1, 2]
            datOrdAlt2[8] = RotMat[2, 2]

            # camera origin
            datOrdAlt2[9] = TranMat[0]
            datOrdAlt2[10] = TranMat[1]
            datOrdAlt2[11] = TranMat[2]


        if 1:
            if 1:
                #fwheader.create_dataset('Camera1CalibrationValue', data=VikCam)
                del f1['F' + str(int(frame1num))]['CameraPos']
                f1['F' + str(int(frame1num))].create_dataset('CameraPos', data=datOrdAlt1)
            if 1:
                del f2['F' + str(int(frame1num)-frameDelay)]['CameraPos']
                f2['F' + str(int(frame1num)-frameDelay)].create_dataset('CameraPos', data=datOrdAlt2)

            #return datOrdAlt,ThreeDMarkedPoints2, TwoDMarkedPoints2








    ######################################################3
    #
    # Essential matrix routene to get camera info if FO is in frame
    #
    #
    def EssentialMatrixForFOinGoPros(self, ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num,
                                     cameraMatrix1,
                                     distCoeffs1, f1, f2, frameDelay):
        # here is the group of 3D points I think



        TwoDMarkedPoints = numpy.zeros((len(ThreeDMarkedPoints), 2))
        TwoDMarkedPoints2 = []
        TwoDMarkedPoints1 = []
        ThreeDMarkedPoints2 = []

        # We are matching the 2D pixel points with the 3D points.   So we are going though the 3D points to find those that are
        # present.
        # print str(int(frame1num)) - frameDelay,str(int(frame1num)),"str(int(frame1num)) - frameDelay"
        Tobject = "fiducialM_"
        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if 1:
                    fwData[cameraNames[0]][insectNum][str(int(frame1num))]
                    fwData[cameraNames[1]][insectNum][str(int(frame1num) - frameDelay)]
                    moviewatch =0
                    TwoDMarkedPoints1.append(
                        [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                         fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                    moviewatch =1
                    TwoDMarkedPoints2.append(
                        [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][0],
                         fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][1]])


            except:
                None

        Tobject = "insect"
        for i in range(len(ThreeDMarkedPoints)):
            insectNum = Tobject + str(int(i + 1))
            try:
                if 1:
                    fwData[cameraNames[0]][insectNum][str(int(frame1num))]
                    fwData[cameraNames[1]][insectNum][str(int(frame1num) - frameDelay)]
                    moviewatch = 0

                    TwoDMarkedPoints1.append(
                        [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                         fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                    moviewatch = 1
                    TwoDMarkedPoints2.append(
                        [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][0],
                         fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num) - frameDelay)][1]])


            except:
                None

        TwoDMarkedPoints1 = numpy.array(TwoDMarkedPoints1)

        TwoDMarkedPoints2 = numpy.array(TwoDMarkedPoints2)
        print TwoDMarkedPoints1, TwoDMarkedPoints2
        print len(TwoDMarkedPoints1),len(TwoDMarkedPoints2)
        focalLen=1114
        ppp=(976.2,560.5)
        # This is the open cv PnP algorithim

        EM, mask=cv2.findEssentialMat(TwoDMarkedPoints1,TwoDMarkedPoints2,focal=focalLen, pp=ppp,method=cv2.RANSAC)

        print cv2.decomposeEssentialMat(EM)
        pointsqq,RotMat,tvec, mask= cv2.recoverPose(EM,TwoDMarkedPoints1,TwoDMarkedPoints2,focal=focalLen, pp=ppp)

        print RotMat,tvec
        print pointsqq, "pointsqqq"
        ###############################################################################################################
        # changing rvec to a matrix

        if 0:
            rvec2Mat = numpy.zeros((3, 3))
            dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

            RotMat = dst
            # using the inverse: (Rt-RtT)
            RotMat = numpy.matrix.transpose(dst)
            TranMat = (-1) * numpy.matmul(RotMat, tvec)

        datOrdAlt = numpy.zeros(12)
        RotMatCopy = numpy.copy(RotMat)
        TranMat=tvec
        # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.

        datOrdAlt[0] = RotMat[0, 0]
        datOrdAlt[1] = RotMat[1, 0]
        datOrdAlt[2] = RotMat[2, 0]
        datOrdAlt[3] = RotMat[0, 1]
        datOrdAlt[4] = RotMat[1, 1]
        datOrdAlt[5] = RotMat[2, 1]
        datOrdAlt[6] = RotMat[0, 2]
        datOrdAlt[7] = RotMat[1, 2]
        datOrdAlt[8] = RotMat[2, 2]

        # camera origin
        datOrdAlt[9] = TranMat[0]
        datOrdAlt[10] = TranMat[1]
        datOrdAlt[11] = TranMat[2]

        datOrdAlt1=numpy.array([1,0,0,0,1,0,0,0,1,0,0,0])

        if 1:
            # fwheader.create_dataset('Camera1CalibrationValue', data=VikCam)
            del f1['F' + str(int(frame1num))]['CameraPos']
            f1['F' + str(int(frame1num))].create_dataset('CameraPos', data=datOrdAlt1)
        if 1:
            del f2['F' + str(int(frame1num) - frameDelay)]['CameraPos']
            f2['F' + str(int(frame1num) - frameDelay)].create_dataset('CameraPos', data=datOrdAlt)

            # return datOrdAlt,ThreeDMarkedPoints2, TwoDMarkedPoints2

            ################################################################################################################

            ################################################################################################################

            ################################################################################################################

            ################################################################################################################

            ################################################################################################################

            ################################################################################################################

    def SquidFitToPath(self, fwData, cameraNames, threeDobjNum, frame1num):


        #plotting with fiducial
        #RotMatMinClas,TranMatMinClas,MinClassData
        if 1:
            try:
                ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])

                ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

                RotMatMinClas=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                TranMatMinClas=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                MinClassData=numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatParams'])

            except:
                None
        pointssquid = []
        # [0,1,2]
        # [0,8,1]
        ## first is nose mantle apex, second is right eye with the apex being forward and third is the left eye with apex being forward

        pointssquidreal = numpy.array(
            [[0, 3.55998, -0.02321], [0.56313, -0.89247, 0.22152], [-0.56313, -0.89247, 0.22152]])
        pathArr = ["Squid" + str(threeDobjNum) + "_0_" + "Squid" + str(threeDobjNum) + "_0",
                   "Squid" + str(threeDobjNum) + "_1_" + "Squid" + str(threeDobjNum) + "_1",
                   "Squid" + str(threeDobjNum) + "_2_" + "Squid" + str(threeDobjNum) + "_2"]

        ####################################################################################################
        #for eq in range(20):  # going through all the squids...
        if 1: # just one squid
            eq=threeDobjNum-1

            pathArr = ["Squid" + str(eq+1) + "_0_" + "Squid" + str(eq+1) + "_0",
                       "Squid" + str(eq+1) + "_1_" + "Squid" + str(eq+1) + "_1",
                       "Squid" + str(eq+1) + "_2_" + "Squid" + str(eq+1) + "_2"]
            path0="Squid" + str(eq+1) + "_0_" + "Squid" + str(eq+1) + "_0"


            #################################################################################
            ################################################################################
            # recording the info....
            try:
                fwData[cameraNames[2]].create_group("Squid" + str(eq+1) + "_" + "Squid" + str(eq+1))
            except:
                del fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)]
                fwData[cameraNames[2]].create_group("Squid" + str(eq+1) + "_" + "Squid" + str(eq+1))



            fwDataKeys = fwData[cameraNames[2]][path0].keys()
            fwDataKeysInt = []
            #print len(fwDataKeys), "fwDataKeys"
            #if len(fwDataKeys) < 2:
            #    continue
            for s in fwDataKeys:
                if s != "FrameDelay":
                    fwDataKeysInt.append(int(s))
            while len(fwDataKeysInt)>0:
            #if 1:
                pointssquid = []
                fwDataKeysmin = min(fwDataKeysInt)
                fwDataKeysmax = max(fwDataKeysInt)
                #print fwDataKeysmin,fwDataKeysmax,eq
                #print fwDataKeysInt





                inX = []
                inY = []
                inZ = []
                # for jj in range(fwDataKeysmin, fwDataKeysmax + 1):
                IsthereaPoint = True
                jj = 0
                print fwDataKeysmin
                while IsthereaPoint is True:
                    if 1:
                        try:
                            InterPoint = numpy.array(fwData[cameraNames[2]][path0][str(int(fwDataKeysmin + jj))]['3Dpoint'])
                            inX.append(InterPoint[0])
                            inY.append(InterPoint[1])
                            inZ.append(InterPoint[2])
                            #print str(int(fwDataKeysmin + jj)),"str(int(fwDataKeysmin + jj))"
                            fwDataKeysInt.remove(fwDataKeysmin + jj)
                            jj += 1
                            #if jj>5:
                            #    IsthereaPoint = False

                        except:
                            IsthereaPoint = False
                            # inX.append(numpy.NaN)
                            # inY.append(numpy.NaN)
                            # inZ.append(numpy.NaN)
                jjMax=jj
                inX = numpy.array(inX)
                inY = numpy.array(inY)
                inZ = numpy.array(inZ)
                pointssquid.append([numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)])
                #print [numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)]

                for jju in range(2):
                    inX = []
                    inY = []
                    inZ = []
                    for jj in range(jjMax):
                        try:
                            InterPoint = numpy.array(
                                fwData[cameraNames[2]][pathArr[jju+1]][str(int(fwDataKeysmin + jj))]['3Dpoint'])
                            inX.append(InterPoint[0])
                            inY.append(InterPoint[1])
                            inZ.append(InterPoint[2])
                            jj += 1
                        except:

                            inX.append(numpy.NaN)
                            inY.append(numpy.NaN)
                            inZ.append(numpy.NaN)

                    inX = numpy.array(inX)
                    inY = numpy.array(inY)
                    inZ = numpy.array(inZ)
                    pointssquid.append([numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)])
                    #print [numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)]











                #print pointssquid
                pointssquid = numpy.array(pointssquid)

                def Rotationthing(SquidIn):
                    theta1 = SquidIn[0]
                    theta2 = SquidIn[1]
                    theta3 = SquidIn[2]
                    magnif = SquidIn[3]
                    #print magnif
                    TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                         [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                    roty = numpy.array(
                        [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                         [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                    rotz = numpy.array(
                        [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                         [0, 0, 1]])
                    # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                    RotMatSQ = numpy.matmul(rotz, numpy.matmul(roty, rotx))
                    PVV = 0
                    for sq in range(len(pointssquidreal)):
                        # print numpy.isnan(pointssquid[sq][0]),pointssquid[sq][0]
                        if numpy.isnan(pointssquid[sq][0]) == False:
                            PV1 = numpy.linalg.norm(
                                pointssquid[sq] - (TranMatSQ + magnif * numpy.matmul(RotMatSQ, pointssquidreal[sq])))
                            # print PV1,"why not here"
                            PVV += PV1 ** 2
                    PVV = numpy.sqrt(PVV)
                    #print PVV,SquidIn
                    return PVV

                def con(SquidIn):
                    theta1 = SquidIn[0]
                    theta2 = SquidIn[1]
                    theta3 = SquidIn[2]
                    magnif = SquidIn[3]
                    TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                         [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                    roty = numpy.array(
                        [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                         [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                    rotz = numpy.array(
                        [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                         [0, 0, 1]])
                    # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                    RotMatSQ = numpy.matmul(rotz, numpy.matmul(roty, rotx))
                    # print RotMatSQ[0][2],"the con"
                    if 1:
                        xhat=numpy.array([1,0,0])
                        xhatST=numpy.matmul(ResultRot.T,numpy.matmul(RotMatSQ,xhat))
                    #return RotMatSQ[0][2]
                    return xhatST[2]

                def con_gt0(SquidIn):
                    magnif = SquidIn[3]
                    if magnif<0:
                        return 1
                    else:
                        return 0


                def conboth(SquidIn):
                    theta1 = SquidIn[0]
                    theta2 = SquidIn[1]
                    theta3 = SquidIn[2]
                    magnif = SquidIn[3]
                    if magnif<0:
                        magnifneg=1
                    else:
                        magnifneg = 0

                    TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                         [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                    roty = numpy.array(
                        [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                         [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                    rotz = numpy.array(
                        [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                         [0, 0, 1]])
                    # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                    RotMatSQ = numpy.matmul(rotz, numpy.matmul(roty, rotx))
                    # print RotMatSQ[0][2],"the con"
                    return RotMatSQ[0][2]+magnifneg


                SquidIn = [0, 0, 0, 1, 0, 0, 0]
                SquidIn = [-1.50202908, 0.32472496, - 0.01857423,0.00412113, - 0.09854525, 0.19311158,0.904858]
                #res = minimize(Rotationthing, SquidIn)
                #SquidIn = res.x
                #SquidIn[3]=abs(SquidIn[3])
                #cons = {'type': 'eq', 'fun': con_gt0}
                #res = minimize(Rotationthing, SquidIn,constraints=cons)

                #print res.x
        #        cons = [{'type': 'eq', 'fun': con},{'type': 'eq', 'fun': con_gt0}]
                cons = [{'type': 'eq', 'fun': con}]

                #print con(res.x)
                bds=((-7,7),(-7,7),(-7,7),(0.001,0.05),(-1000,1000),(-1000,1000),(-1000,1000))
                #SquidIn = res.x  # [0, 0, 0, 0, 0, 0, 0]
                res = minimize(Rotationthing, SquidIn,bounds=bds, constraints=cons)

                print res.x
                #print con(res.x)
                SquidIn = res.x
                theta1 = SquidIn[0]
                theta2 = SquidIn[1]
                theta3 = SquidIn[2]
                magnif = SquidIn[3]
                TranMatSQ = numpy.array([SquidIn[4], SquidIn[5], SquidIn[6]])

                rotx = numpy.array(
                    [[1, 0, 0], [0, numpy.cos(theta1), numpy.sin(theta1)],
                     [0, -numpy.sin(theta1), numpy.cos(theta1)]])
                roty = numpy.array(
                    [[numpy.cos(theta2), 0, numpy.sin(theta2)], [0, 1, 0],
                     [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
                rotz = numpy.array(
                    [[numpy.cos(theta3), -numpy.sin(theta3), 0], [numpy.sin(theta3), numpy.cos(theta3), 0],
                     [0, 0, 1]])
                # rotx = self.rotation_matrix(xvect1, (-1 * theta))
                RotMatSQ = magnif * numpy.matmul(rotz, numpy.matmul(roty, rotx))
                #RotMatMinClas, TranMatMinClas, MinClassData
                TotRotSQ=numpy.matmul(RotMatMinClas/MinClassData[3],RotMatSQ/magnif)

                def isRotationMatrix(R):
                    Rt = numpy.transpose(R)
                    shouldBeIdentity = numpy.dot(Rt, R)
                    I = numpy.identity(3, dtype=R.dtype)
                    n = numpy.linalg.norm(I - shouldBeIdentity)
                    return n < 1e-6

                def rotationMatrixToEulerAngles(R):

                    assert (isRotationMatrix(R))

                    sy = numpy.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

                    singular = sy < 1e-6

                    if not singular:
                        x = numpy.arctan2(R[2, 1], R[2, 2])
                        y = numpy.arctan2(-R[2, 0], sy)
                        z = numpy.arctan2(R[1, 0], R[0, 0])
                    else:
                        x = numpy.arctan2(-R[1, 2], R[1, 1])
                        y = numpy.arctan2(-R[2, 0], sy)
                        z = 0

                    return numpy.array([x, y, z])
                TotEuler=rotationMatrixToEulerAngles(TotRotSQ)
                TotTrans=numpy.matmul(RotMatMinClas,TranMatSQ)+ TranMatMinClas
                FitDatawithFiducial=[TotEuler[0], TotEuler[1], TotEuler[2], MinClassData[3] * magnif, TotTrans[0], TotTrans[1], TotTrans[2]]
                print "[",TotEuler[0],",",TotEuler[1],",",TotEuler[2],",",MinClassData[3]*magnif,",",TotTrans[0],",",TotTrans[1],",",TotTrans[2],"],"

                i=fwDataKeysmin
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)].create_group(
                    str(int(i)))
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'FitData', data=SquidIn)
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'RotMatSQ', data=RotMatSQ)
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'TranMatSQ', data=RotMatSQ)
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'TotRotSQ', data=TotRotSQ)
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'TotTrans', data=TotTrans)
                fwData[cameraNames[2]]["Squid" + str(eq+1) + "_" + "Squid" + str(eq+1)][
                    str(int(i))].create_dataset(
                    'FitDatawithFiducial', data=FitDatawithFiducial)


            #aqa



        return RotMatSQ,TranMatSQ
        #return pointssquid
    ################################################################################################################

        ################################################################################################################

        ################################################################################################################



    def FitWaterSurfacetofromPointCloud(self,fwData, cameraNames,Water_Surface_Point_Cloud_Path):
        UseRotMatMinClas = False

        if 1:

            ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])

            ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

            try:
                RotMatMinClas = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                TranMatMinClas = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])
                UseRotMatMinClas = True
            except:
                None

        mapFile = Water_Surface_Point_Cloud_Path


        if 1:

            mapFileLoaded = open(mapFile)

            lines = mapFileLoaded.readlines()
            mapVect = numpy.zeros(3)
            xload = []
            yload = []
            mapFileLoaded = open(mapFile)

            lines = mapFileLoaded.readlines()
            for iu in range(2):
                xs = []
                ys = []
                zs = []
                for ii in lines:

                    if len(ii.split(";")) == 3:
                        mapVect = numpy.array(
                            [float(ii.split(";")[0]), float(ii.split(";")[1]), float(ii.split(";")[2])])
                    else:
                        mapVect = numpy.array(
                            [float(ii.split(" ")[0]), float(ii.split(" ")[1]), float(ii.split(" ")[2])])
                    if iu == 1:
                        if UseRotMatMinClas == False:
                            mapVect = numpy.matmul(ResultRot.T, mapVect) + ResultTran
                        else:
                            mapVect = numpy.matmul(RotMatMinClas, mapVect) + TranMatMinClas

                    xs.append(mapVect[0])
                    ys.append(mapVect[1])
                    zs.append(mapVect[2])

                # do fit
                tmp_A = []
                tmp_b = []
                for i in range(len(xs)):
                    tmp_A.append([xs[i], ys[i], 1])
                    tmp_b.append(zs[i])
                b = numpy.matrix(tmp_b).T
                A = numpy.matrix(tmp_A)
                fit = (A.T * A).I * A.T * b
                errors = b - A * fit
                residual = numpy.linalg.norm(errors)

                #print "solution:"
                #print "%f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
                #print "%f, %f, -1, %f" % (fit[0], fit[1], fit[2])
                if iu == 0:
                    WaterPlaneInputstring=str(float(fit[0]))+","+str(float(fit[1]))+",-1.0,"+str(float(fit[2]))+";"
                else:
                    WaterPlaneInputstring =WaterPlaneInputstring+str(float(fit[0]))+","+str(float(fit[1]))+",-1.0,"+str(float(fit[2]))
                #print "errors:"
                #print errors
                #print "residual:"
                #print residual
            print "Enter this into the Water_Surface_Plane column in the DualGoPros sheet in ProjectMain"
            print WaterPlaneInputstring
        ################################################################################################################

        ################################################################################################################






    def makeSTLfromPointCloud(self,fwData,cameraNames,h5WritePath,PointCLoudName,PointCLoudScale,Water_Surface_Plane,answer):

        UseRotMatMinClas = False
        if answer == 0:
            IncludeWaterPlane = False
            JustWaterPlane = False
        elif answer == 1:
            IncludeWaterPlane=True
            JustWaterPlane=False
        elif answer ==2:
            IncludeWaterPlane=True
            JustWaterPlane=True

        if Water_Surface_Plane == "":
            IncludeWaterPlane = False
            JustWaterPlane = False
            print "Water_Surface_Plane not found... not including water"


        if 1:

            ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])

            ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

            try:
                RotMatMinClas = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                TranMatMinClas = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])
                UseRotMatMinClas = True
            except:
                None
        MapDict = {}
        if 0:
            mapFile=PointCLoudName+".txt"
            stlWrite = PointCLoudName+".stl"
        if 1:
            mapFile=PointCLoudName
            stlWrite = PointCLoudName.split(".")[0]+".stl"

        if 1:
            ##########################
            #  scale is the variable that most needs to be adjusted
            scale = PointCLoudScale
            mapFileLoaded = open(mapFile)
            #mapFileWrite = open(mapFileWrite, "w")
            lines = mapFileLoaded.readlines()
            mapVect = numpy.zeros(3)
            xload = []
            yload = []
            for jj, ii in enumerate(lines):
                if jj > 1:
                    # If from xyz file from meshlab
                    #mapVect=numpy.array([float(ii.split(" ")[0]),float(ii.split(" ")[1]),float(ii.split(" ")[2])])

                    # if from ORB_SLAM
                    #mapVect = numpy.array([float(ii.split(";")[0]), float(ii.split(";")[1]), float(ii.split(";")[2])])
                    if len(ii.split(";"))==3:
                        mapVect = numpy.array(
                            [float(ii.split(";")[0]), float(ii.split(";")[1]), float(ii.split(";")[2])])
                    else:
                        mapVect = numpy.array(
                            [float(ii.split(" ")[0]), float(ii.split(" ")[1]), float(ii.split(" ")[2])])

                    if UseRotMatMinClas == False:
                        mapVect = numpy.matmul(ResultRot.T, mapVect) + ResultTran
                    else:
                        mapVect = numpy.matmul(RotMatMinClas, mapVect) + TranMatMinClas
                    mapVect = scale * mapVect
                    # MapDict[[int(mapVect[0]),int(mapVect[1])]]=[]
                    # MapDict[[int(mapVect[0]), int(mapVect[1])]].append(mapVect)

                    # using rounding to orgaize map points.
                    try:
                        MapDict[str(int(mapVect[0])) + ";" + str(int(mapVect[1]))].append(mapVect)
                    except:
                        MapDict[str(int(mapVect[0])) + ";" + str(int(mapVect[1]))] = []
                        MapDict[str(int(mapVect[0])) + ";" + str(int(mapVect[1]))].append(mapVect)
            print len(MapDict.keys())

            mapDicNew = {}
            newmap = []

            ###############################################
            ####  now we are taking the average of each bin
            for ii in MapDict.keys():
                indistict = []
                print len(MapDict[ii])
                if 1:
                    for z in range(len(MapDict[ii])):
                        # print MapDict[ii][z][2]
                        indistict.append(MapDict[ii][z][2])

                    indistict = numpy.array(indistict)

                    if len(indistict) > 1:
                        spliceaverage = numpy.average(indistict)
                        spliceaverage = numpy.median(indistict)
                        newpoint = numpy.array([float(ii.split(";")[0]), float(ii.split(";")[1]), spliceaverage])
                        newpoint = newpoint / scale
                        newpointstr = str(newpoint[0]) + ";" + str(newpoint[1]) + ";" + str(newpoint[2])
                        xload.append(float(ii.split(";")[0]))
                        yload.append(float(ii.split(";")[1]))
                        newmap.append(newpoint.tolist())
                        # newmap.append(newpointstr)
                        mapDicNew[ii] = newpointstr
                        #mapFileWrite.write(
                        #    str(newpoint[0]) + " " + str(newpoint[1]) + " " + str(newpoint[2]) + "\n")
            # print mapDicNew.keys()

            xload = numpy.array(xload)
            yload = numpy.array(yload)

            xof = 0
            yof = 0
            peace = 0
            #print int(xload.min()), int(xload.max())
            #print int(yload.min()), int(yload.max())
            xloadmin = int(xload.min())
            xloadmax = int(xload.max())
            yloadmin = int(yload.min())
            yloadmax = int(yload.max())
            #print xloadmax, xloadmin
            # xloadmin=int(xloadmin-(xloadmax-xloadmin)*(0.5))
            #print xloadmax, xloadmin

            #######################################################################
            # interpolation scheme
            if 1:
                for i in range(xloadmin, xloadmax):
                    for j in range(yloadmin, yloadmax):
                        try:
                            mapDicNew[str(i + xof) + ";" + str(j + yof)]
                        except:
                            continue
                        if (str(i + 2 + xof) + ";" + str(j + yof) in mapDicNew) and (
                                str(i + 1 + xof) + ";" + str(j + yof) not in mapDicNew):
                            zv1 = float(mapDicNew[str(i + xof) + ";" + str(j + yof)].split(";")[2])
                            zv2 = float(mapDicNew[str(i + 2 + xof) + ";" + str(j + yof)].split(";")[2])

                            mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)] = str((i + 1 + xof) / scale) + ";" + str(
                                (j + yof) / scale) + ";" + str((zv1 + zv2) / 2.0)
                        if (str(i + xof) + ";" + str(j + 2 + yof) in mapDicNew) and (
                                str(i + xof) + ";" + str(j + 1 + yof) not in mapDicNew):
                            zv1 = float(mapDicNew[str(i + xof) + ";" + str(j + yof)].split(";")[2])
                            zv2 = float(mapDicNew[str(i + xof) + ";" + str(j + 2 + yof)].split(";")[2])

                            mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)] = str((i + xof) / scale) + ";" + str(
                                (j + 1 + yof) / scale) + ";" + str((zv1 + zv2) / 2.0)

                        if (str(i + 3 + xof) + ";" + str(j + yof) in mapDicNew) and (
                                str(i + 1 + xof) + ";" + str(j + yof) not in mapDicNew) and (
                                str(i + 2 + xof) + ";" + str(j + yof) not in mapDicNew):
                            print "found one"
                            zv1 = float(mapDicNew[str(i + xof) + ";" + str(j + yof)].split(";")[2])
                            zv3 = float(mapDicNew[str(i + 3 + xof) + ";" + str(j + yof)].split(";")[2])

                            mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)] = str((i + 1 + xof) / scale) + ";" + str(
                                (j + yof) / scale) + ";" + str(zv1 + (zv3 - zv1) * (1.0 / 3.0))
                            mapDicNew[str(i + 2 + xof) + ";" + str(j + yof)] = str((i + 2 + xof) / scale) + ";" + str(
                                (j + yof) / scale) + ";" + str(zv1 + (zv3 - zv1) * (2.0 / 3.0))

                        if 1:
                            if (str(i + xof) + ";" + str(j + 3 + yof) in mapDicNew) and (
                                    str(i + xof) + ";" + str(j + 1 + yof) not in mapDicNew) and (
                                    str(i + xof) + ";" + str(j + 2 + yof) not in mapDicNew):
                                print "found one"
                                zv1 = float(mapDicNew[str(i + xof) + ";" + str(j + yof)].split(";")[2])
                                zv3 = float(mapDicNew[str(i + xof) + ";" + str(j + 3 + yof)].split(";")[2])

                                mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)] = str((i + xof) / scale) + ";" + str(
                                    (j + 1 + yof) / scale) + ";" + str(zv1 + (zv3 - zv1) * (1.0 / 3.0))

                                mapDicNew[str(i + xof) + ";" + str(j + 2 + yof)] = str((i + xof) / scale) + ";" + str(
                                    (j + 2 + yof) / scale) + ";" + str(zv1 + (zv3 - zv1) * (2.0 / 3.0))

                            #######################################################################

            #            putting in water

            if IncludeWaterPlane==True:  # we just make this 0 to not make water
                if Water_Surface_Plane != "":
                    abcd = []
                    for ic in range(4):
                        abcd.append(float(Water_Surface_Plane.split(";")[1].split(",")[ic]))
                # abcd = [-0.012840, -0.056112, -1, -9.821203]
                # abcd = [0.070603, -0.264929, -1, 1.439623 ]#horse pond???
                # abcd without transfor
                # 0.166865 x + -3.466464 y + 1.416294 = z
                # abcd=[0.166865,-3.466464,-1,1.416294]
                #abcd = [-0.010277, 0.000218, -1, -16.543996]  # neon skimmer  # last one to be used

                mapDicNewW = {}
                # distance=abs(InterPoint[0]*abcd[0]+InterPoint[1]*abcd[1]+InterPoint[2]*abcd[2]+abcd[3])/numpy.sqrt(abcd[0]**2+abcd[1]**2+abcd[2]**2)

                for i in range(xloadmin, xloadmax):
                    for j in range(yloadmin, yloadmax):
                        try:
                            mapDicNew[str(i + xof) + ";" + str(j + yof)]
                        except:
                            zzz = (i + xof) * abcd[0] / scale + (j + yof) * abcd[1] / scale + abcd[3]
                            ## was mapDicNewW
                            if JustWaterPlane==True:
                                mapDicNew[str(i + xof) + ";" + str(j + yof)] = str((i + xof) / scale) + ";" + str(
                                    (j + yof) / scale) + ";" + str(zzz)
                            else:
                                mapDicNew[str(i + xof) + ";" + str(j + yof)] = str((i + xof) / scale) + ";" + str(
                                    (j + yof) / scale) + ";" + str(zzz)

                # this needs to be done better.
                #
                # If we want water in the STL you need to take out the W.
                #
                #
                if JustWaterPlane == True:
                    mapDicNew = mapDicNewW
            ##############

            #######################################################################
            # now we are making the STL

            for i in range(xloadmin, xloadmax):
                for j in range(yloadmin, yloadmax):
                    try:  # checking if we have all four points.
                        mapDicNew[str(i + xof) + ";" + str(j + yof)]
                        mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)]
                        mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)]
                        mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)]
                    except:
                        continue
                    print "gotone", i + xof, j + yof

                    if peace == 1:
                        # ids = map(id, vertex)
                        if mapDicNew[str(i + xof) + ";" + str(j + yof)] in vertex:
                            # print "in", vertexnp.tolist().index(list(mapDicNew[str(i + xof) + ";" + str(j + yof)]))
                            # ZeroF=vertexnp.tolist().index(list(mapDicNew[str(i + xof) + ";" + str(j + yof)]))
                            ZeroF = vertex.index(mapDicNew[str(i + xof) + ";" + str(j + yof)])
                        else:
                            vertex.append(mapDicNew[str(i + xof) + ";" + str(j + yof)])
                            ZeroF = len(vertex) - 1

                        if mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)] in vertex:
                            # OneF=vertexnp.tolist().index(list(mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)]))
                            OneF = vertex.index(mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)])
                        else:
                            vertex.append(mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)])
                            OneF = len(vertex) - 1

                        if mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)] in vertex:
                            # print vertex,mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)]
                            # TwoF = vertexnp.tolist().index(list(mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)]))
                            TwoF = vertex.index(mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)])
                        else:
                            vertex.append(mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)])
                            TwoF = len(vertex) - 1

                        if mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)] in vertex:
                            # ThreeF = vertexnp.tolist().index(list(mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)]))
                            ThreeF = vertex.index(mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)])
                        else:
                            vertex.append(mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)])
                            ThreeF = len(vertex) - 1
                        faces.append([ZeroF, OneF, TwoF])
                        faces.append([TwoF, ThreeF, ZeroF])
                        # vertexnp = numpy.array(vertex)

                    if peace == 0:
                        peace = 1
                        vertex = [mapDicNew[str(i + xof) + ";" + str(j + yof)],
                                  mapDicNew[str(i + 1 + xof) + ";" + str(j + yof)],
                                  mapDicNew[str(i + 1 + xof) + ";" + str(j + 1 + yof)],
                                  mapDicNew[str(i + xof) + ";" + str(j + 1 + yof)]]
                        # vertexnp=numpy.array(vertex)
                        faces = [[0, 1, 2], [2, 3, 0]]

            vertexnp = []
            for ii in vertex:
                vertexnp.append([float(ii.split(";")[0]), float(ii.split(";")[1]), float(ii.split(";")[2])])
            vertex = numpy.array(vertexnp)
            faces = numpy.array(faces)
            cube = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for iw, f in enumerate(faces):
                for jw in range(3):
                    cube.vectors[iw][jw] = vertex[f[jw], :]

            # Write the mesh to file "cube.stl"
            cube.save(stlWrite)










        ################################################################################################################




    def PnPtoOutlinePoints(self,TwoDMarkedPoints2, datOrdAlt, cameraMatrix1,distCoeffs1, obbTree,ThreeDMarkedPoints2,Tobject,fwData,cameraNames,moviewatch,frame1num,Ptarlen,width,height,makeInt):


    ####################################################################  # CircleArray,Outline,twoAve=self.getMeshOutline(TwoDMarkedPoints2,datOrdAlt, cameraMatrix1, distCoeffs1,caster)

        ###############################################################################################
        ############new pnp with edges

        if 1:
            # get mesh outline????
            CircleArray, Outline, twoAve = self.getMeshOutline(TwoDMarkedPoints2, datOrdAlt, cameraMatrix1,
                                                               distCoeffs1, obbTree)

            TwoDMarkedPointsSurface = []
            # definging the 3D points
            ThreeDMarkedPoints3 = []
            #print ThreeDMarkedPoints2,"ThreeDMarkedPoints2 first"
            for vv in ThreeDMarkedPoints2:
                ThreeDMarkedPoints3.append(vv)

            # these are the ones currently used.
            TwoDMarkedPoints3 = []
            for vv in TwoDMarkedPoints2:
                TwoDMarkedPoints3.append(vv)

            corEdgepnts = []

            # Going through the boundary points
            for i in range(20, 39):

                insectNum = Tobject + str(int(i + 1))
                try:
                    # TwoDMarkedPointsSurface.append(
                    #    [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                    #     fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                    point2Boundary = [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                                      fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]]
                    # adding them here.
                    TwoDMarkedPoints3.append(point2Boundary)
                    point2Boundary = numpy.array(point2Boundary)
                    #print point2Boundary,"point2Boundary",insectNum


                    madeIt = True
                except:
                    madeIt = False
                #print madeIt,"madeit"
                minDisToBoundary = []
                if madeIt == True:
                    # what is outline???
                    for v2 in Outline:
                        v = numpy.array([v2[0], v2[1]])
                        minDisToBoundary.append(numpy.linalg.norm(v - point2Boundary))

                    minDisToBoundary = numpy.array(minDisToBoundary)

                    N1 = numpy.array(Outline[minDisToBoundary.argsort()[0]])
                    N2 = numpy.array(Outline[minDisToBoundary.argsort()[1]])
                    N21 = N2 - N1
                    vN1 = point2Boundary - N1

                    closestPoint = (vN1.dot(N21) / ((numpy.linalg.norm(N21)) ** 2)) * N21 + N1

                    CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, closestPoint, cameraMatrix1,
                                                             distCoeffs1)

                    pSource = CamOrgn23
                    pTarget = CamOrgn23 + Ptarlen * pointvect23
                    pointsIntersection, cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                    OutlineInterations = 100
                    # finding the closest point whether or not it goes.
                    if not pointsIntersection:  # so it needs to go to the center
                        for three in range(0, OutlineInterations):
                            interpoint = closestPoint + three * (twoAve - closestPoint) / OutlineInterations

                            CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1,
                                                                     distCoeffs1)
                            pSource = CamOrgn23
                            pTarget = CamOrgn23 + Ptarlen * pointvect23
                            pointsIntersection, cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                            if not pointsIntersection:
                                None
                            else:
                                CorespondingEdgePoint = closestPoint + (three) * (
                                    twoAve - closestPoint) / OutlineInterations
                                corEdgepnts.append(CorespondingEdgePoint)
                                ThreeDMarkedPoints3.append(pointsIntersection[0])
                                break

                    else:
                        for three in range(0, OutlineInterations):
                            interpoint = closestPoint + three * (closestPoint - twoAve) / OutlineInterations

                            CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, interpoint, cameraMatrix1,
                                                                     distCoeffs1)

                            pSource = CamOrgn23
                            pTarget = CamOrgn23 + Ptarlen * pointvect23
                            pointsIntersection, cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)

                            if not pointsIntersection:
                                CorespondingEdgePoint = closestPoint + (three - 1) * (
                                    closestPoint - twoAve) / OutlineInterations
                                ThreeDMarkedPoints3.append(pointsIntersectionSave[0])
                                corEdgepnts.append(CorespondingEdgePoint)
                                break
                            else:
                                None
                            pointsIntersectionSave = pointsIntersection

            ThreeDMarkedPoints3 = numpy.array(ThreeDMarkedPoints3)
            TwoDMarkedPoints3 = numpy.array(TwoDMarkedPoints3)

            RotMat = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
            TranMat = numpy.zeros(3)

            pcMat3 = numpy.transpose(ThreeDMarkedPoints3)

            MeshProj3 = self.ReturnMeshProjection(pcMat3, RotMat, TranMat, datOrdAlt, cameraMatrix1,
                                                  distCoeffs1, width,
                                                  height, makeInt)

            MeshProj3 = numpy.array(MeshProj3)

            MeshProj3Length, MeshProj3Width = MeshProj3.shape
            ReprojectionErrors = numpy.zeros((int(MeshProj3Length)))
            for vv in range(int(MeshProj3Length)):
                ReprojectionErrors[vv] = numpy.linalg.norm(MeshProj3[vv] - TwoDMarkedPoints3[vv])



            rvec = cv2.solvePnPRansac(ThreeDMarkedPoints3, TwoDMarkedPoints3, cameraMatrix1, distCoeffs1)[1]
            tvec = cv2.solvePnPRansac(ThreeDMarkedPoints3, TwoDMarkedPoints3, cameraMatrix1, distCoeffs1)[2]

            if 1:
                rvec = \
                    cv2.solvePnP(ThreeDMarkedPoints3, TwoDMarkedPoints3, cameraMatrix1, distCoeffs1,
                                 rvec=rvec,
                                 tvec=tvec, useExtrinsicGuess=1)[1]
                tvec = \
                    cv2.solvePnP(ThreeDMarkedPoints3, TwoDMarkedPoints3, cameraMatrix1, distCoeffs1,
                                 rvec=rvec,
                                 tvec=tvec,
                                 useExtrinsicGuess=1)[2]

            ###############################################################################################################
            # changing rvec to a matrix
            rvec2Mat = numpy.zeros((3, 3))
            dst, jacobian = cv2.Rodrigues(rvec, rvec2Mat)

            RotMat = dst
            # using the inverse: (Rt-RtT)
            RotMat = numpy.matrix.transpose(dst)
            # RotMat = numpy.linalg.inv(dst)
            TranMat = (-1) * numpy.matmul(RotMat, tvec)
            datOrdAlt = numpy.zeros(12)

            # remeber that the vertical collums are what corresponds to xvect, yvect, zvect.
            RotMatCopy=numpy.copy(RotMat)
            datOrdAlt[0] = RotMat[0, 0]
            datOrdAlt[1] = RotMat[1, 0]
            datOrdAlt[2] = RotMat[2, 0]
            datOrdAlt[3] = RotMat[0, 1]
            datOrdAlt[4] = RotMat[1, 1]
            datOrdAlt[5] = RotMat[2, 1]
            datOrdAlt[6] = RotMat[0, 2]
            datOrdAlt[7] = RotMat[1, 2]
            datOrdAlt[8] = RotMat[2, 2]

            # camera origin
            datOrdAlt[9] = TranMat[0]
            datOrdAlt[10] = TranMat[1]
            datOrdAlt[11] = TranMat[2]

            RotMat = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
            TranMat = numpy.zeros(3)

            pcMat3 = numpy.transpose(ThreeDMarkedPoints3)

            MeshProj3 = self.ReturnMeshProjection(pcMat3, RotMat, TranMat, datOrdAlt, cameraMatrix1,
                                                  distCoeffs1, width,
                                                  height, makeInt)

            MeshProj3 = numpy.array(MeshProj3)

            MeshProj3Length, MeshProj3Width = MeshProj3.shape
            ReprojectionErrors = numpy.zeros((int(MeshProj3Length)))
            for vv in range(int(MeshProj3Length)):
                ReprojectionErrors[vv] = numpy.linalg.norm(MeshProj3[vv] - TwoDMarkedPoints3[vv])

                # print "reprojection Error after", ReprojectionErrors.mean(), ReprojectionErrors.std()

        CircleArray, Outline, twoAve = self.getMeshOutline(TwoDMarkedPoints2, datOrdAlt, cameraMatrix1,
                                                           distCoeffs1, obbTree)

        ##################3 printing out a lot of things here

        ##############################3
        #### trying to adjust the pitch of the squid to get zero roll.
        self.datOrdAltGlobe = numpy.copy(datOrdAlt)



        #######################################################################################333    roll
        res = minimize(self.AdjustRolltoMinimizeRoll, 0, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})
        print res.x[0], "res.x"
        AdjustSquidPitch = 0
        CameraRoll, RotAroundY, datOrdAltAlt = self.AdjustRolltoMinimizeRollOut(res.x[0])



        ### here we are looking at roll???  zxz
        cameraOrginW = numpy.array([datOrdAltAlt[9], datOrdAltAlt[10], datOrdAltAlt[11]])

        Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
        cameraZ = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], datOrdAltAlt[8]])
        cameraZX = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], 0])
        cameraPitch = (180 / 3.1415926) * numpy.arccos(
            numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))

        cameraPitch8 = (180 / 3.1415926) * numpy.arctan(datOrdAltAlt[8] / numpy.linalg.norm(cameraZX))
        print "squid Roll", res.x[0]
        print "camera pitch", cameraPitch8
        print "Asimuth", Asimuth

        cameraZ = numpy.array([-datOrdAltAlt[9], -datOrdAltAlt[10], -datOrdAltAlt[11]])
        cameraZX = numpy.array([-datOrdAltAlt[9], -datOrdAltAlt[10], 0])

        zvect1 = numpy.zeros(3)

        yvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)

        xvect1[0] = datOrdAltAlt[0]
        xvect1[1] = datOrdAltAlt[1]
        xvect1[2] = datOrdAltAlt[2]

        yvect1[0] = datOrdAltAlt[3]
        yvect1[1] = datOrdAltAlt[4]
        yvect1[2] = datOrdAltAlt[5]

        zvect1[0] = datOrdAltAlt[6]
        zvect1[1] = datOrdAltAlt[7]
        zvect1[2] = datOrdAltAlt[8]

        zhat = numpy.array([0, 0, 1.0])
        camHrizontal = numpy.cross(zvect1, zhat)
        camHrizontal = numpy.array(camHrizontal)
        camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
        CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
            numpy.linalg.norm(xvect1) * numpy.linalg.norm(
                camHrizontal))) * 180.0 / 3.1415926

        print "camera Roll", CameraRoll
        cameraPitch = (180 / 3.1415926) * numpy.arctan(
            -datOrdAltAlt[11] / numpy.linalg.norm(cameraZX))
        print "camera pitch from camorig", cameraPitch
        print "camera pitch differece", cameraPitch - cameraPitch8



        ######################################################################################   pitch
        res = minimize(self.AdjustPitchtoMinimizeRoll, 0, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})

        AdjustSquidPitch = 0
        CameraRoll, RotAroundY, datOrdAltAlt = self.AdjustPitchtoMinimizeRollOut(res.x[0])
        resx0=res.x[0]

        fishRoll, fishRelAzimuth, fishPitch = self.goingThoughVariousCameraParameters(fwData, frame1num, datOrdAlt,
                                                                                     datOrdAltAlt, CameraRoll, RotAroundY,
                                                                                     resx0, RotMatCopy)



        return datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt,Outline,corEdgepnts,resx0,fishRoll








    def watchVidandPicFrames(self):


        ############################################################################################################
        # declaring a few variables.
        CameraOriginArray=[]
        FillMapDicframe=0
        MapErrorDicMeauredPoint={}
        MapErrorDicMapPoint={}
        delayfraction=0
        thePlatform=platform.system()
        thePlatformNum=0
        if thePlatform=='Windows':
            thePlatformNum = 0
        elif thePlatform=='Linux':
            thePlatformNum = 1
        elif  thePlatform=='Darwin':
            thePlatformNum = 2



        PointCLoudScale=1
        NoPlottingError=True
        UseRotMatMinClas=False
        place2SaveImages = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/amberenvi"
        savefunction = 2
        SingleFramesImages = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/imagestoMovies/SquidVid"
        KalmanOverlay=False
        FillMapDic=False
        DictA={}
        PlotDesciptiveDic={}
        MovieFileNotFound=False
        ThreeDlineDicBUse=True
        PointCloudFromORBPoints=[]
        PointCloudIdentified=False
        UseFullGoPro6Video=""
        h5SquidDelay=0
        DoRewindInZ=True
        if 1:#  this was put here because of the quiid stuff.
            Frames2Average=0
            FrameSkipInc=0
        if 0:
            Frames2Average = 10
            FrameSkipInc = 240

        frame1numStore2=-1
        ViewDualGoProSquid=False
        stlMax=0
        threeDobjNum=0
        threeDobjNumMax=0
        Track3DobjectInDualGP=True
        viewTrackingThings=True
        saveImagesAsIs=False
        UseBackgroundORBstl=False
        DontUpdatePoint=False
        ContinueClickFrameMode = True
        settingTheRewindView = False
        RVvalue=5
        ThereIsSTLData=False
        setFrame=1
        MovieNamePrevious=""
        ViewVideoInstedofSquid=False
        mirrorCameraA=["mirrorCamera1","mirrorCamera2"]
        rottag=0
        viewAolp = False
        frame1numStore=-1
        AdjustSquidPitch=0
        towardOrAwayFromSun=""
        PanelNum=0
        frame1numtrack2=-1
        useH5 = False
        insectColor = [[.1, .2, .6], [.9, .6, .4], [.5, .2, .6], [.1, .9, .2], [.4, .7, .2], [.3, .9, .7], [.2, .3, .9],
                       [.1, .4, .2]]
        insectColor = numpy.array(insectColor) * 255
        insectColor = numpy.concatenate((insectColor, insectColor, insectColor, insectColor, insectColor, insectColor),
                                        axis=0)
        insectColor = numpy.concatenate((insectColor, insectColor, insectColor, insectColor, insectColor, insectColor),
                                        axis=0)
        insectnumit = 0
        WindowWidth = 600  # having some problemes with this it was 600 now it is 540  use 540 for selecting fish frames from nov 2018 trip


        movWin1x = 70
        movWin1y = 0
        movWin2x = 950
        movWin2y = 0
        bitmap1 = []
        bitmap2 = []
        UseforwardFrameSubtraction = False
        p1 = None


        magZoom = .44444
        moviewatch = 0

        ClickedError = {}
        ClickedDicB = {}
        ThreeDlineDicB1={}
        ThreeDlineDicB2 = {}

        DicBInt = 0
        DicBInt1 = 0
        DicBInt2 = 0

        insectMaxNum = 1
        point1fromH5 = numpy.zeros(2)
        point2fromH5 = numpy.zeros(2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        movieRewind = False
        theapp = animateFromVideo()
        ClickIncrimentMode = False
        checkFramesWithoutH5 = True
        jnum = 0
        jnumMax = 4
        ErrorCircle = 3
        UseNumbersForError = False
        ClickedErrorInd = False
        UseFWError = True
        IsInsectEmpty = ""
        clicked1m1=numpy.zeros(3)
        MouseHasNotMovedIn=0
        MouseMoved=False
        Ptarlen = 10000
        RoiStringIndex=0
        indextrack=0
        viewSquidMeshStuff=True
        keysImages=["keys1.PNG","keys2.PNG","keys3.PNG","keys4.PNG"]
        keysImagesIndex=0
        Water_Surface_Plane=""

        ###   rotation things

        RotMat = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

        RotMat = numpy.array(RotMat)
        TranMat = numpy.array([-0.34, -0.67, 0.03])
        newProjMesh = []

        TransStep = .1
        zvect1 = numpy.zeros(3)
        xvect1 = numpy.zeros(3)
        yvect1 = numpy.zeros(3)
        theta = .02
        scale = 1.1
        cameraMatrix1 = numpy.zeros((3, 3))
        distCoeffs1 = numpy.zeros(5)

        cameraMatrix2 = numpy.zeros((3, 3))
        distCoeffs2 = numpy.zeros(5)
        cameraNames = ["camera1", "camera2", "combined"]
        Tobject = "insect"

        ############# somewhat relateed to squid work.  what are these doing here
        frame1numtrack = 0
        frame2numtrack = 0
        SquidIndex = 1
        makeInt = True

        CircleArray = numpy.zeros(1)
        RoiStringArray=["ROI1","ROI2","ROI3","ROI4","ROI5","ROI6","ROI7","ROI8","ROI9","ROI10"]
        RoiString = "ROI10"
        datOrdAlt = numpy.array([1, 0, 0, 0, -1, 0, 0, 0, -1, 0,-150, 800])  #looks like the top.  cuttle
        point1track = numpy.zeros(2)
        point2track = numpy.zeros(2)

        viewGuidingLines=True

        corEdgepnts = []



        if 0:
            root = tk.Tk()
            root.withdraw()

            width_px = root.winfo_screenwidth()
            height_px = root.winfo_screenheight()
            print width_px,height_px
            root.destroy()


        # excel interface
        ###############################################################################################################


        #main work books:
        workbook = xlrd.open_workbook('ProjectMain.xlsx')
        worksheetGP = workbook.sheet_by_name("DualGoPros")
        worksheetPath = workbook.sheet_by_name("Paths")
        worksheetORB = workbook.sheet_by_name("ORB_projects")
        worksheetPolmp4 = workbook.sheet_by_name("SingleVideos")
        worksheetImageC = workbook.sheet_by_name("ImageCompilations")
        worksheetMain=workbook.sheet_by_name("Main")


        #headers
        worksheetORBHeader = dict(zip(worksheetORB.row_values(0), range(len(worksheetORB.row_values(0)))))
        worksheetGPHeader = dict(zip(worksheetGP.row_values(0), range(len(worksheetGP.row_values(0)))))
        worksheetPolmp4Header = dict(zip(worksheetPolmp4.row_values(0), range(len(worksheetPolmp4.row_values(0)))))
        worksheetPathHeader = dict(zip(worksheetPath.row_values(0), range(len(worksheetPath.row_values(0)))))
        worksheetImageCHeader = dict(zip(worksheetImageC.row_values(0), range(len(worksheetImageC.row_values(0)))))
        worksheetMainHeader=dict(zip(worksheetMain.row_values(0), range(len(worksheetMain.row_values(0)))))


        #list of names
        worksheetORBNames = dict(zip(worksheetORB.col_values(0), range(len(worksheetORB.col_values(0)))))
        worksheetGPNames = dict(zip(worksheetGP.col_values(0), range(len(worksheetGP.col_values(0)))))
        worksheetPolmp4Names = dict(zip(worksheetPolmp4.col_values(0), range(len(worksheetPolmp4.col_values(0)))))


        #getting the specific project to run
        projectWBxfinder=dict(zip(workbook.sheet_by_name("Main").col_values(worksheetMainHeader["ToUse"]),
                            range(len(workbook.sheet_by_name("Main").col_values(worksheetMainHeader["ToUse"])))))

        projectWorkbook= str(workbook.sheet_by_name("Main").cell(projectWBxfinder["*"], worksheetMainHeader["projectWorkbook"]).value)

        projName = dict(zip(workbook.sheet_by_name(projectWorkbook).col_values(0),
                            range(len(workbook.sheet_by_name(projectWorkbook).col_values(0)))))
        projPaths = dict(zip(workbook.sheet_by_name("Paths").col_values(0),
                             range(len(workbook.sheet_by_name("Paths").col_values(0)))))
        projHeader = dict(zip(workbook.sheet_by_name(projectWorkbook).row_values(0),
                              range(len(workbook.sheet_by_name(projectWorkbook).row_values(0)))))


        if projectWorkbook == "ORB_projects":
            print "made it to orb projects"
            #basic initialization
            ORBprojHeader = dict(zip(workbook.sheet_by_name(projectWorkbook).row_values(0),
                                  range(len(workbook.sheet_by_name(projectWorkbook).row_values(0)))))
            projectNamexfinder = dict(zip(workbook.sheet_by_name(projectWorkbook).col_values(ORBprojHeader["ToUse"]),
                                          range(
                                              len(workbook.sheet_by_name(projectWorkbook).col_values(
                                                  ORBprojHeader["ToUse"])))))
            ORBprojectName = str(
                workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["name"]).value)
            #getting the orb project name
            print ORBprojectName


            ##################################
            #Getting paths


            projectPaths = str(
                workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["path"]).value)

            pathVid = projPaths[projectPaths]
            GoProprojPaths = dict(zip(workbook.sheet_by_name("DualGoPros").col_values(0),
                                 range(len(workbook.sheet_by_name("DualGoPros").col_values(0)))))
            GoProprojHeader = dict(zip(workbook.sheet_by_name("DualGoPros").row_values(0),
                                  range(len(workbook.sheet_by_name("DualGoPros").row_values(0)))))
            GoProprojectName=str(
                workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["link_to_video"]).value)

            filename1vidName=str(workbook.sheet_by_name("DualGoPros").cell(GoProprojPaths[GoProprojectName], GoProprojHeader["FileName"]).value)#.split(".")[0]+".mp4"#FileName2 switched with FileName on 200225
            filename2vidName= str(workbook.sheet_by_name("DualGoPros").cell(GoProprojPaths[GoProprojectName], GoProprojHeader["FileName2"]).value)#.split(".")[0]+".mp4"
            filename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_videos"]).value) + "/" + filename1vidName

            filename2 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_videos"]).value) + "/" + filename2vidName
            FiletoORB_SLAM2 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Path to ORB_SLAM2"]).value)
            FiletoORB_SLAM2_output = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Path to ORB_SLAM2 output"]).value)

            print filename1,filename2,FiletoORB_SLAM2


            if str(
                workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["MakeH5"]).value)=="y":
                print "making H5"
                filewrite=FiletoORB_SLAM2_output+"/"+str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Map_metadata_name1"]).value)
                filename=FiletoORB_SLAM2_output+"/"+str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Camera1_metadata"]).value)
                self.ConvertMapDataintoH5(filename, filewrite)
                print "making H5 2"
                filewrite=FiletoORB_SLAM2_output+"/"+str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Map_metadata_name2"]).value)
                filename=FiletoORB_SLAM2_output+"/"+str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Camera2_metadata"]).value)
                self.ConvertMapDataintoH5(filename, filewrite)
                return
            else:


                ################################################
                #now working on YAML files
                YAMLpath=FiletoORB_SLAM2 + "/Examples/Monocular/YAMLfromExcel.yaml"
                #YAMLpath = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/YAMLtrial.yaml"
                ORBYAMLName = str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["MakeYaml"]).value)
                #print ORBYAMLName
                YAMLprojHeader = dict(zip(workbook.sheet_by_name("Yaml_files").row_values(0),
                                      range(len(workbook.sheet_by_name("Yaml_files").row_values(0)))))

                worksheetYAMLNames = dict(zip(workbook.sheet_by_name("Yaml_files").col_values(0), range(len(workbook.sheet_by_name("Yaml_files").col_values(0)))))

                YAMLprojDic = dict(zip(workbook.sheet_by_name("Yaml_files").row_values(0),zip(workbook.sheet_by_name("Yaml_files").row_values(
                    worksheetYAMLNames[ORBYAMLName]))))

                #print YAMLprojDic

                DicORBdetails={}
                DicORBdetails["map_in"]=str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Map_In"]).value)
                DicORBdetails["map_out"]=str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Map_Out"]).value)
                if DicORBdetails["map_out"]=="":
                    DicORBdetails["map_out"]=DicORBdetails["map_in"]
                DicORBdetails["ORB_pointcloud_output"] = str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["ORB_pointcloud_output"]).value)
                DicORBdetails["Camera1_metadata"]=str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Camera1_metadata"]).value)
                DicORBdetails["Camera2_metadata"]=str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Camera2_metadata"]).value)
                DicORBdetails["MakeH5"]=str(
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["MakeH5"]).value)
                DicORBdetails["FiletoORB_SLAM2_output"]=FiletoORB_SLAM2_output
                try:
                    DicORBdetails["Starting_frame"] = int(
                        workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"],
                                                                     ORBprojHeader["Starting_frame"]).value)
                except:
                    DicORBdetails["Starting_frame"] = 1000

                    #map file out
                #starting frame number
                #key frame trajectory file
                #Camera 1 metadata
                #camera 2 metadata




                self.MakeYaml(YAMLprojDic,YAMLpath,DicORBdetails)



                ##########################################
                #Running ORB_SLAM2

                Use_Map_File_Str=str((
                    workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], ORBprojHeader["Use_Map_File_for_ORB_SLAM2"]).value))
                if Use_Map_File_Str=="y":
                    Use_Map_File="1"
                else:
                    Use_Map_File="0"

                bigstring2 = FiletoORB_SLAM2 + "/Examples/Monocular/mono_tum " + FiletoORB_SLAM2 + "/Vocabulary/ORBvoc.bin " + \
                             YAMLpath+" " + filename1+" " \
                             + filename2+" "+Use_Map_File
                print bigstring2

                try:
                    os.system(bigstring2)
                except:
                    print "ORB_SLAM2 did not respond"

                return






        projectPathxfinder = dict(zip(workbook.sheet_by_name(projectWorkbook).col_values(projHeader["ToUse"]),
                                    range(
                                        len(workbook.sheet_by_name(projectWorkbook).col_values(projHeader["ToUse"])))))

        projectPaths = str(
            workbook.sheet_by_name(projectWorkbook).cell(projectPathxfinder["*"], projHeader["path"]).value)

        pathVid=projPaths[projectPaths]
        #looking for the video or folder of images

        projectNamexfinder = dict(zip(workbook.sheet_by_name(projectWorkbook).col_values(projHeader["ToUse"]),
                                      range(
                                          len(workbook.sheet_by_name(projectWorkbook).col_values(
                                              projHeader["ToUse"])))))
        projectName = str(
            workbook.sheet_by_name(projectWorkbook).cell(projectNamexfinder["*"], projHeader["name"]).value)





        if projectWorkbook == "ImageCompilations":
            filename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_ImageFolder"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["FileName"]).value)
            filename2 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_ImageFolder"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["FileName2"]).value)#FileName2 switched with FileName on 200225

            if os.path.isdir(filename1):
                print "no file1"
            else:
                if os.path.isdir(filename2):
                    filename1 = filename2
                else:
                    print "this project is not going to work"
            AustraliaSingleFrames=filename2
        elif projectWorkbook == "SingleVideos":
            filename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Paths_to_videos"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["FileName"]).value)
            filename2 = filename1

        else:
            filename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Paths_to_videos"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["FileName"]).value)#FileName2 switched with FileName on 200225
            filename2 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Paths_to_videos"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["FileName2"]).value)

            if os.path.isfile(filename1):
                print "no file1"
            else:
                if os.path.isfile(filename2):
                    filename1 = filename2
                else:
                    print "this project is not going to work"

        print filename1, filename2, "filename1,filename2"





        if projectWorkbook == "SingleVideos":

            UseSquidExcel = False
            viewSquid = True
            initialFrameRaw = (
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Initial_Frame_to_Start"]).value)
            if initialFrameRaw=="":
                initialFrame=300
            else:
                initialFrame=int(initialFrameRaw)
            usePnpProj = False
            UseBlankCamera = True
            savefunctionRaw = workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],
                                                                              projHeader["Save_Function"]).value
            if savefunctionRaw=="":
                savefunction=0
            else:
                savefunction=int(savefunctionRaw)

            place2SaveImages= str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_ImageFolder"]).value)
            print "place2SaveImages",place2SaveImages
            print savefunction,"savefunction"
        if projectWorkbook == "DualGoPros":  # no squid singles

            UseSquidExcel = False
            viewSquid = False
            #initialFrame = 8816  # 37000
            initialFrame = int(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Initial_Frame_to_Start"]).value)
            UseBlankCamera = False
            usePnpProj = False
            Water_Surface_Plane=workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Water_Surface_Plane"]).value
            UseFullGoPro6Video=workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["SquidVideo"]).value

        if projectWorkbook == "ImageCompilations":  # for sqid work



            UseSquidExcel = True
            viewSquid = True
            initialFrame = 20
            usePnpProj = False
            UseBlankCamera = True
            insectMaxNum = 44





        if projectWorkbook == "ImageCompilations":
            Excelfilename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader[
                "Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["excel file"]).value)

            rb = xlrd.open_workbook(Excelfilename1)
            xlrownum = len(rb.sheet_by_index(0).col_values(0))
            wb = copy(rb)
            s = wb.get_sheet(0)

        if projectWorkbook == "SingleVideos":
            imagProj = str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ImageCompilationProject"]).value)
            if imagProj != "":
                ImagprojName = dict(zip(workbook.sheet_by_name("ImageCompilations").col_values(0),
                                        range(len(workbook.sheet_by_name("ImageCompilations").col_values(0)))))

                Excelfilename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                    workbook.sheet_by_name("ImageCompilations").cell(ImagprojName[imagProj], worksheetImageCHeader["excel file"]).value)

                rb = xlrd.open_workbook(Excelfilename1)
                print rb, "rb"
                xlrownum = len(rb.sheet_by_index(0).col_values(0))
                wb = copy(rb)
                print wb, "wb"
                s = wb.get_sheet(0)
                AustraliaSingleFrames = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_ImageFolder"]).value) + "/" + str(
                    workbook.sheet_by_name("ImageCompilations").cell(ImagprojName[imagProj], worksheetImageCHeader["FileName"]).value)
                PanelNum = int(workbook.sheet_by_name("ImageCompilations").cell(ImagprojName[imagProj], worksheetImageCHeader["Panel"]).value)






        ########  Looking for the H5 files.   why is this a try?
        try:
            #this is from go pro stuff with two h5 files
            h5filename1 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Map_metadata_name1"]).value)#Map_metadata_name1 switched with Map_metadata_name2 on 200225
            h5filename2 = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Map_metadata_name2"]).value)
            #why is this not normally colored, it is rewritten down there
            h5filenamewrite = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["output_file"]).value)
            useH5 = True
        except:
            useH5 = False
            checkFramesWithoutH5 = True

        # load frame number

        try:
            if str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Time_delay"]).value) != "":
                frameDelay = int(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Time_delay"]).value)
            else:
                frameDelay = 0
        except:
            frameDelay = 0

        try:
            if str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],
                                                                projHeader["Subframe_delay"]).value) != "":
                delayfraction = float(
                    workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["Subframe_delay"]).value)
            else:
                delayfraction = 0
        except:
            delayfraction = 0

        h5filewitename = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["output_file"]).value)
        h5WritePath = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value)
        h5filenamewrite = h5WritePath + "/" + h5filewitename
        h5filewitenameBU = h5filewitename.split(".")[0] + "_BU.h5"
        h5filenamewriteBU = h5WritePath + "/" + h5filewitenameBU


        try:
            PointCLoudPath = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Map_pointcloud_name"]).value)
            PointCLoudName = str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Map_pointcloud_name"]).value)
            PointCLoudScale = float(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Map_pointcloud_scale"]).value)

            if PointCLoudName!="":
                PointCloudIdentified=True
        except:
            None
        try:
            Water_Surface_Point_Cloud_Path = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value) + "/" + str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Water_Surface_Point_Cloud"]).value)

            Water_Surface_Point_Cloud_Name = str(
                workbook.sheet_by_name("ORB_projects").cell(
                    worksheetORBNames[workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["ORB_Name"]).value],
                    worksheetORBHeader["Water_Surface_Point_Cloud"]).value)
            if Water_Surface_Point_Cloud_Name!="":
                Water_Surface_identified=True
        except:
            None


        if useH5 == True:
            print h5filename1, h5filename2, "h5 names h5filename1,h5filename2"
            if 0:
                f1 = h5py.File(h5filename1, 'a')
                f2 = h5py.File(h5filename2, 'a')
            if 1:
                f1 = h5py.File(h5filename1, 'r')
                f2 = h5py.File(h5filename2, 'r')

            ################################    Camera Calibration info
            cameraMatrix1[0, 0] = f1['header']['CameraCalibrationValue'][0]
            cameraMatrix1[0, 2] = f1['header']['CameraCalibrationValue'][2]
            cameraMatrix1[1, 1] = f1['header']['CameraCalibrationValue'][1]
            cameraMatrix1[1, 2] = f1['header']['CameraCalibrationValue'][3]
            cameraMatrix1[2, 2] = 1

            distCoeffs1[0] = f1['header']['CameraCalibrationValue'][4]
            distCoeffs1[1] = f1['header']['CameraCalibrationValue'][5]
            distCoeffs1[2] = f1['header']['CameraCalibrationValue'][6]
            distCoeffs1[3] = f1['header']['CameraCalibrationValue'][7]
            distCoeffs1[4] = f1['header']['CameraCalibrationValue'][8]

            cameraMatrix2[0, 0] = f2['header']['CameraCalibrationValue'][0]
            cameraMatrix2[0, 2] = f2['header']['CameraCalibrationValue'][2]
            cameraMatrix2[1, 1] = f2['header']['CameraCalibrationValue'][1]
            cameraMatrix2[1, 2] = f2['header']['CameraCalibrationValue'][3]
            cameraMatrix2[2, 2] = 1

            distCoeffs2[0] = f2['header']['CameraCalibrationValue'][4]
            distCoeffs2[1] = f2['header']['CameraCalibrationValue'][5]
            distCoeffs2[2] = f2['header']['CameraCalibrationValue'][6]
            distCoeffs2[3] = f2['header']['CameraCalibrationValue'][7]
            distCoeffs2[4] = f2['header']['CameraCalibrationValue'][8]

            print f2['header']['CameraCalibrationValue'][:],"f2['header']['CameraCalibrationValue']"

            if UseFullGoPro6Video=="a":#will need to intergrate this but this is for viewing bigger MP4 files over the reduced ones from GoPro6
                ################################    Camera Calibration info
                print "made an a"
                cameraMatrix1[0, 0] = 1113.9
                cameraMatrix1[0, 2] = 976.1
                cameraMatrix1[1, 1] = 1114.8
                cameraMatrix1[1, 2] = 560.5
                cameraMatrix1[2, 2] = 1

                distCoeffs1[0] = -0.13956291
                distCoeffs1[1] = 0.09491485
                distCoeffs1[2] =0.00654849
                distCoeffs1[3] = 0.00368825
                distCoeffs1[4] = -0.0095185

                cameraMatrix2[0, 0] =1113.9
                cameraMatrix2[0, 2] =976.1
                cameraMatrix2[1, 1] =1114.8
                cameraMatrix2[1, 2] =560.5
                cameraMatrix2[2, 2] = 1

                distCoeffs2[0] = -0.13956291
                distCoeffs2[1] = 0.09491485
                distCoeffs2[2] =0.00654849
                distCoeffs2[3] = 0.00368825
                distCoeffs2[4] = -0.0095185
                h5SquidDelay=302




                #####################################################################################
            ###  write into data file h5data set up
        print h5filenamewrite, "h5filenamewrite"
        print h5filewitename, h5WritePath, os.path.isdir(h5WritePath)
        if h5filewitename!="" and h5filewitename.split(".")[1] == "h5" and os.path.isdir(h5WritePath):
            print "yes the h5 write", h5filenamewrite, os.path.isfile(h5filenamewrite)

            if os.path.isfile(h5filenamewrite):
                fw = h5py.File(h5filenamewrite, 'a')
                fwheader = fw.get('header')
                fwData = fw.get('data')
                print fwData, "fwData"
                if useH5 == False:
                    VikCam = numpy.zeros(0)
                    VikCam = fwheader['Camera1CalibrationValue'][:]

                    cameraMatrix1[0, 0] = VikCam[0]
                    cameraMatrix1[0, 2] = VikCam[2]
                    cameraMatrix1[1, 1] = VikCam[1]
                    cameraMatrix1[1, 2] = VikCam[3]
                    cameraMatrix1[2, 2] = 1

                    distCoeffs1[0] = VikCam[4]
                    distCoeffs1[1] = VikCam[5]
                    distCoeffs1[2] = VikCam[6]
                    distCoeffs1[3] = VikCam[7]
                    distCoeffs1[4] = VikCam[8]
                    distCoeffs2 = distCoeffs1
                    cameraMatrix2 = cameraMatrix1
                    print cameraMatrix1, distCoeffs1, "from h5"

            else:
                fw = h5py.File(h5filenamewrite, 'w')
                fwheader = fw.create_group('header')
                if useH5 == True:
                    fwheader.create_dataset('Camera1CalibrationValue', data=f1['header']['CameraCalibrationValue'])
                    fwheader.create_dataset('Camera2CalibrationValue', data=f2['header']['CameraCalibrationValue'])
                else:
                    VikCam = numpy.zeros(0)
                    VikCam = numpy.array(
                        [1898.54, 1910.83, 612.521, 691.126, -0.534567, 2.32161, -0.00956921, 0.00146657, -7.23781])
                    fwheader.create_dataset('Camera1CalibrationValue', data=VikCam)
                    fwheader.create_dataset('Camera2CalibrationValue', data=VikCam)

                    cameraMatrix1[0, 0] = VikCam[0]
                    cameraMatrix1[0, 2] = VikCam[2]
                    cameraMatrix1[1, 1] = VikCam[1]
                    cameraMatrix1[1, 2] = VikCam[3]
                    cameraMatrix1[2, 2] = 1

                    distCoeffs1[0] = VikCam[4]
                    distCoeffs1[1] = VikCam[5]
                    distCoeffs1[2] = VikCam[6]
                    distCoeffs1[3] = VikCam[7]
                    distCoeffs1[4] = VikCam[8]
                    distCoeffs2 = distCoeffs1
                    cameraMatrix2 = cameraMatrix1
                    print cameraMatrix1, distCoeffs1

                fwData = fw.create_group('data')

                for ii in range(2):
                    fwData.create_group(cameraNames[ii])
                    fwData[cameraNames[ii]].create_group("insect0")
                    fwData[cameraNames[ii]].create_group("fiducial0_")
                    print cameraNames[ii]
                fwData.create_group(cameraNames[2])





        #this was a problem on 8/7/2019
        #we did a change on how we view STL bringing STL into the other side of this project
        if viewSquid==True:
            try: #qwq
                filenameSTL = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],  projHeader["STLfile"]).value)
                print filenameSTL,"filenameSTL"
                #if filenameSTL =="STLfile":#before 8/7/2019
                if filenameSTL !="":#after 8/7/2019
                    viewSquid = True
                else:
                    viewSquid = False
            except:
                viewSquid=False

        print viewSquid,"ViewSquid"
##################################################################################################
        ###########   getting the box

        if viewSquid == False:

            PointsOnObjects = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["Fiducial_Box"]).value)
            if projectWorkbook=="DualGoPros":#qwq
                FiducialFromAnotherProject=str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["Fiducial_Box_Information_from_Other_Project"]).value)
            worksheetPOO = workbook.sheet_by_name("PointsOnObjects")
            worksheetPOOHeader = dict(zip(worksheetPOO.row_values(0), range(len(worksheetPOO.row_values(0)))))
            print worksheetPOOHeader, "worksheetPOOHeader"
            POOindex = worksheetPOOHeader[PointsOnObjects]
            # print POOindex
            qq = 2
            Box = []
            while str(workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex).value) != "":
                Box.append([float(workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex).value),
                                           float(
                                               workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex + 1).value),
                                           float(
                                               workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex + 2).value)])
                qq += 1

            Box = numpy.array(Box)
            print "Box!!!!!",Box

        ##################################################################################################
        ###########   getting the box

        if viewSquid == False:
            try:
                filenameSTL = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["STLfile"]).value)

                if filenameSTL=="":
                    UseBackgroundORBstl=False
                else:
                    UseBackgroundORBstl = True
            except:
                UseBackgroundORBstl = False

        ##################################################################################################################
        #########  getting the stl mark points.

        if viewSquid == True:

            # three or 6 panels
            PanelNum = int(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["Panel"]).value)

        try:
            PointsOnObjects = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],projHeader["PointsONObjects"]).value)
            worksheetPOO = workbook.sheet_by_name("PointsOnObjects")
            worksheetPOOHeader = dict(zip(worksheetPOO.row_values(0), range(len(worksheetPOO.row_values(0)))))
            print worksheetPOOHeader, "worksheetPOOHeader"
            POOindex = worksheetPOOHeader[PointsOnObjects]
            # print POOindex
            qq = 2
            ThreeDMarkedPoints = []
            while str(workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex).value) != "":
                ThreeDMarkedPoints.append([float(workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex).value),
                                           float(
                                               workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex + 1).value),
                                           float(
                                               workbook.sheet_by_name("PointsOnObjects").cell(qq, POOindex + 2).value)])
                qq += 1

            ThreeDMarkedPoints = numpy.array(ThreeDMarkedPoints)
            stlMax =numpy.abs(ThreeDMarkedPoints).max()
            print stlMax,"stlMax"
        except:
            None

        if viewSquid == True or UseBackgroundORBstl== True: # this is in just the dual go pro
            ###################################################################################################
                #########################3
                # loading in meshes

            #putting these in maps
            #filenameSTL = str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], 8).value)
            filenameSTLwp = str(workbook.sheet_by_name("Paths").cell(pathVid, worksheetPathHeader["Paths_to_H5"]).value)+"/"+str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], projHeader["STLfile"]).value)

            readerSTL = vtk.vtkSTLReader()
            readerSTL.SetFileName(filenameSTLwp)
            readerSTL.Update()

            polydata = readerSTL.GetOutput()

            # If there are no points in 'vtkPolyData' something went wrong
            if polydata.GetNumberOfPoints() == 0:
                raise ValueError(
                    "No point data could be loaded from '" + filenameSTL)

            mesh = polydata

            ################ normals

            # Create a new 'vtkPolyDataNormals' and connect to the 'earth' sphere
            normalsCalcEarth = vtk.vtkPolyDataNormals()
            normalsCalcEarth.SetInputConnection(readerSTL.GetOutputPort())

            normalsCalcEarth.ComputePointNormalsOn()
            # Enable normal calculation at cell centers
            normalsCalcEarth.ComputeCellNormalsOn()
            # Disable splitting of sharp edges
            normalsCalcEarth.SplittingOff()
            # Disable global flipping of normal orientation
            normalsCalcEarth.FlipNormalsOff()
            # Enable automatic determination of correct normal orientation
            normalsCalcEarth.AutoOrientNormalsOn()

            # normalsCalcEarth.ConsistencyOn()
            # Perform calculation

            normalsCalcEarth.Update()

            obbTree = vtk.vtkOBBTree()
            obbTree.SetDataSet(mesh)
            obbTree.BuildLocator()
            if stlMax!=0:
                pcMat = self.STLsurfacePycastVTK(obbTree,stlMax)
                print pcMat, "vtk"
            else:
                pcMat = self.STLsurfacePycastVTK(obbTree,100)
                print pcMat, "vtk"

            print "Made obbTree!!!!!!!!!!!!!!!!!!!!!!!!"
            # Graph=vtk.vtkGraph()
            idList = vtk.vtkIdList()

            # print "vtkgraph",vtk.vtkGraph.GetVertices(mesh)
            print polydata.GetNumberOfPoints(), "number of point"
            print polydata.GetNumberOfCells(), "number of point"

            print polydata.GetCellData(), "cells"
            print polydata.GetCell(0).GetPoints().GetPoint(0)
            print polydata.GetCell(0).GetPoints().GetPoint(1)
            print polydata.GetCell(0).GetPoints().GetPoint(2)


            squidVert = []

            for i in range(polydata.GetNumberOfPoints()):
                squidVert.append(polydata.GetPoint(i))



            ###########################################################################################################

        print frameDelay, "frameDelay"




                ###############################################################################################

        filenameSplit1 = filename1.split("/")
        ActualFileName1 = filenameSplit1[-1].split(".")
        GoproName1 = filenameSplit1[-2] + "_" + ActualFileName1[0]
        print GoproName1,"GoproName1"

        filenameSplit2 = filename2.split("/")
        ActualFileName2 = filenameSplit2[-1].split(".")

        GoproName2 = filenameSplit2[-2] + "_" + ActualFileName2[0]
        print GoproName2,"GoproName2"
        self.captureFile1 = filename1
        self.captureFile2 = filename2
        if PanelNum == 4:
            WindowWidth = 544  # having some problemes with this it was 600 now it is 540  use 540 for selecting fish frames from nov 2018 trip
            #WindowWidth = 540  # just put this in here while doing an australia video project 190824
        else:
            WindowWidth = 600  # having some problemes with this it was 600 now it is 540  use 540 for selecting fish frames from nov 2018 trip

        if UseSquidExcel==False:
            try:
                self.capture1 = cv2.VideoCapture(self.captureFile1)

                self.fps1 = self.capture1.get(cv2.CAP_PROP_FPS) / 2.0
                length1 = int(self.capture1.get(cv2.CAP_PROP_FRAME_COUNT))

                self.waitTime1 = (1000.0 / self.fps1) / 2
                print self.fps1,length1
                self.isPaused = True
                self.MovieIndex = []

                self.capture2 = cv2.VideoCapture(self.captureFile2)
                length2 = int(self.capture2.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps2 = self.capture2.get(cv2.CAP_PROP_FPS) / 2.0
                self.waitTime2 = (1000.0 / self.fps2) * 2
                print self.fps2

                self.capture1.set(cv2.CAP_PROP_POS_FRAMES, initialFrame + frameDelay)
                self.capture2.set(cv2.CAP_PROP_POS_FRAMES, initialFrame)

                ret, frame1 = self.capture1.read()
                ret, frame2 = self.capture2.read()

                height, width, channels = frame1.shape


                if viewAolp==True:

                    if PanelNum == 3:
                        height, width=720,1280

                    if PanelNum == 6:
                        height, width = 720, 1280
                    if PanelNum == 4:
                        height, width = 1080,1920

                frameratio = height / float(width)
                self.point1[0] = width / 2
                self.point1[1] = height / 2

                self.point2[0] = width / 2
                self.point2[1] = height / 2

                magZoom = 480 / width
            except:
                MovieFileNotFound=True
        if UseSquidExcel == True or MovieFileNotFound==True:
            if 0:#PanelNum==4:
                self.point1[0] = 1920.0 / 2
                self.point1[1] = 1080.0 / 2

                self.point2[0] = 1920.0 / 2
                self.point2[1] = 1080.0 / 2
                frameratio = 1080.0  / 1920.0
                magZoom = float(WindowWidth) / 1080.0

            if 1:#else:
                self.point1[0] = 1280 / 2
                self.point1[1] = 720 / 2

                self.point2[0] = 1280 / 2
                self.point2[1] = 720 / 2
                frameratio = 720.0 / 1280.0
                magZoom = float(WindowWidth)/720.0
            #magZoom = 1
            print PanelNum



            point1track = self.point1
            frame1num=1
            frame2num=1
            self.waitTime1=2000.0/11.0
            self.waitTime2 = 2000.0 / 11.0
            frame1=0
            self.isPaused = True


        ######################################################################################################33

        ######################################################################################################33

        ######################################################################################################33

        if 1:
            try:
                look4Max=[]
                for wwe in range(2):
                    Looking4IntMax1Num=[]
                    for comb in fwData[cameraNames[0]]:
                        if comb.split("t")[0]=="insec":
                            Looking4IntMax1Num.append(int(comb.split("t")[1]))

                    Looking4IntMax1Num=numpy.array(Looking4IntMax1Num)
                    look4Max.append(numpy.max(Looking4IntMax1Num))

                look4Max=numpy.array(look4Max)
                insectMaxNum=numpy.max(look4Max)
            except:
                insectMaxNum=1



        ######################################################################################################33
        # the meat part of the program.   Using keyboard controls etc.
        #############################################################################################





        Kkeys = cv2.imread(keysImages[keysImagesIndex])
        cv2.imshow("Keys1", Kkeys)
        cv2.moveWindow("Keys1", 1400, 0)
        ########### start loop
        while frame1 is not None:



            ####################################
            #####   loading frames

            if self.isPaused == False and MovieFileNotFound==False:
                if movieRewind == False:
                    if DontUpdatePoint == False:
                        ret, frame1 = self.capture1.read()
                        ret, frame2 = self.capture2.read()
                    else:
                        if moviewatch == 0:
                            ret, frame1 = self.capture1.read()
                        if moviewatch == 1:
                            ret, frame2 = self.capture2.read()

                else:
                    self.rewind(5, self.capture1)
                    ret, frame1 = self.capture1.read()

                    self.rewind(5, self.capture2)
                    ret, frame2 = self.capture2.read()

            else:
                None

            if UseSquidExcel == False and MovieFileNotFound==False:
                frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                if viewAolp==True and frame1numStore!=frame1num:
                    as_array = numpy.asarray(frame1[:, :])
                    if PanelNum == 3:
                        xcorner1 = 1322
                        ycorner1 = 720
                        resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                    if PanelNum == 6:
                        xcorner1 = 25
                        ycorner1 = 0
                        resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                    if PanelNum == 4:
                        xcorner1 = 1051  # I had 1053, but Missaels calculations were 1051
                        ycorner1 = 0
                        resized_image = as_array[ycorner1:ycorner1 + 1080, xcorner1:xcorner1 + 1920]  # for 1080 images
                        frameratio = 1 / 1.7777777777
                    frame1 = resized_image
                    frame1numStore=frame1num

                    #print frame1num,frame2num
            if UseSquidExcel == True and ViewVideoInstedofSquid==False:  # we are bootlegging the program it is thinking it is running a video but we are running these insted.
                PNGPath = rb.sheet_by_index(0).cell(SquidIndex, 4).value
                SquidExcelDetails = [str(int(SquidIndex)), str(rb.sheet_by_index(0).cell(SquidIndex, 0).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 1).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 2).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 3).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 4).value)]

                PNGPath = AustraliaSingleFrames + "/" + PNGPath
                frame = cv2.imread(PNGPath)
                as_array = numpy.asarray(frame[:, :])


                ActualFileName1 = str(rb.sheet_by_index(0).cell(SquidIndex, 1).value).split(".")
                ActualFileName1FN=float((rb.sheet_by_index(0).cell(SquidIndex, 2).value))


                # cropping the images from multipanel videos.
                if PanelNum==3:
                    xcorner1 = 683
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                if PanelNum==6:
                    xcorner1 = 25
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                if PanelNum==4:

                    xcorner1 = 1051
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 1080, xcorner1:xcorner1 + 1920]  # for 1080 images
                    frameratio=1/1.7777777777

                    if 1:
                        mag=720.0/1080.0
                        xRsize = int(mag * 1920)
                        yRsize = int(mag * 1080)

                        resized_image = cv2.resize(resized_image, (xRsize, yRsize))


                frame1 = resized_image


                if 0:#undistort image
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix1, distCoeffs1, (xRsize, yRsize), 1, (xRsize, yRsize))

                    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, None, newcameramtx, (xRsize, yRsize), 5)

                    frame1 = cv2.remap(frame1, mapx, mapy, cv2.INTER_LINEAR)


                # putting the SquidIndex as the frame number  which is the value on the excell sheet linking the paths to the squid.
                frame1num = SquidIndex
                frame2num = SquidIndex
                frame1numtrack = frame1num

###########################################################################################################################
            # Jun 3 2019


            if UseSquidExcel == True and ViewVideoInstedofSquid==True:  # we are bootlegging the program it is thinking it is running a video but we are running these insted.
                PNGPath = rb.sheet_by_index(0).cell(SquidIndex, 4).value
                MovieName = rb.sheet_by_index(0).cell(SquidIndex, 1).value
                framenumOnMovie = rb.sheet_by_index(0).cell(SquidIndex, 2).value
                SquidExcelDetails = [str(int(SquidIndex)), str(rb.sheet_by_index(0).cell(SquidIndex, 0).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 1).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 2).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 3).value),
                                     str(rb.sheet_by_index(0).cell(SquidIndex, 4).value)]

                MoviePath="F:/Belize/FinalMovies/"
                if MovieName!=MovieNamePrevious:
                    self.capture1 = cv2.VideoCapture(MoviePath+MovieName)

                    self.fps1 = self.capture1.get(cv2.CAP_PROP_FPS) / 2.0
                    length1 = int(self.capture1.get(cv2.CAP_PROP_FRAME_COUNT))

                    self.waitTime1 = (1000.0 / self.fps1) / 2
                    print self.fps1, length1
                    self.isPaused = True


                ############## Set this and then let it run.
                framenumOnMoviebound=25
                if setFrame==1:
                    self.capture1.set(cv2.CAP_PROP_POS_FRAMES, framenumOnMovie-framenumOnMoviebound)
                    ret, frame = self.capture1.read()
                    frame2watch = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                    setFrame=0

                if frame2watch< framenumOnMovie+framenumOnMoviebound:
                    ret, frame = self.capture1.read()
                    frame2watch = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                    if frame2watch==framenumOnMovie:
                        viewSquidMeshStuff=True
                    else:
                        viewSquidMeshStuff=False
                as_array = numpy.asarray(frame[:, :])


                # cropping the images from multi panel videos.

                if PanelNum==3:
                    xcorner1 = 683
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                if PanelNum==6:
                    xcorner1 = 25
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 720, xcorner1:xcorner1 + 1280]  # for 1080 images

                if PanelNum==4:

                    xcorner1 = 1051 # I had 1053, but Missaels calculations were 1051
                    ycorner1 = 0
                    resized_image = as_array[ycorner1:ycorner1 + 1080, xcorner1:xcorner1 + 1920]  # for 1080 images
                    frameratio=1/1.7777777777

                    if 1:
                        mag=720.0/1080.0
                        xRsize = int(mag * 1920)
                        yRsize = int(mag * 1080)

                        resized_image = cv2.resize(resized_image, (xRsize, yRsize))


                frame1 = resized_image
                frame1num = SquidIndex
                frame2num = SquidIndex
                frame1numtrack = frame1num
                MovieNamePrevious = MovieName


            if MovieFileNotFound==True:
                frame1 = cv2.imread("FileNotFound.png")


            height, width, channels = frame1.shape
             ##################################################################################
            ####### geting ORB_Camera loaction

            try:
                dat1 = numpy.array(f1['F' + str(int(frame1num-h5SquidDelay))]['CameraPos'][:])

                dat2 = numpy.array(f2['F' + str(int(frame2num-h5SquidDelay))]['CameraPos'][:])
                useH5 = True
                #print h5SquidDelay
            except:  # was key error
                # print "frames not recorded"
                useH5 = False
                if checkFramesWithoutH5 == True:
                    dat1 = numpy.zeros(12)
                    dat2 = numpy.zeros(12)
                    dat1 = numpy.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -3])
                    # datOrdAlt=dat1
                    dat2 = dat1
                    useH5 = False
                    if usePnpProj == True:
                        dat1 = datOrdAlt
                        dat2 = dat1
                else:

                    ##### use???????????????
                    self.fastforward(1, self.capture1)
                    ret, frame1 = self.capture1.read()

                    self.fastforward(1, self.capture2)
                    ret, frame2 = self.capture2.read()
                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                        cv2.CAP_PROP_POS_FRAMES)

                    continue
            del bitmap1
            del bitmap2


            ###################################################################3################################
            # getting the central point values to adjust the frame position or to select a point.
            if (frame1num == frame1numtrack and numpy.linalg.norm(self.point1 - point1track) > 0.00001) or (frame2num == frame2numtrack and numpy.linalg.norm(self.point2 - point2track) > 0.00001) or DontUpdatePoint==True:
                None
            else:
                if viewTrackingThings==True:
                    try:  # h5data
                        if moviewatch == 0:
                            self.point1[0] = \
                                fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][str(int(frame1num))][0]
                            self.point1[1] = \
                                fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][str(int(frame1num))][1]
                            point1track = numpy.copy(self.point1)
                            frame1numtrack = frame1num

                        if moviewatch == 1:
                            self.point2[0] = \
                                fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][str(int(frame2num))][0]
                            self.point2[1] = \
                                fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][str(int(frame2num))][1]
                            point2track = numpy.copy(self.point2)
                            frame2numtrack = frame1num
                    except:  # KeyError or any
                        None
                #print self.point1,"self.point1????"

            ####################################################################################
            # we transform the frame to zoom in or to pan
            if moviewatch == 2:
                windowsize = [WindowWidth, int(WindowWidth * frameratio)]
                bitmap1, self.KeepTrack1 = self.TransformFrame(frame1, self.point1, magZoom, windowsize)
                bitmap2, self.KeepTrack2 = self.TransformFrame(frame2, self.point2, magZoom, windowsize)
            elif moviewatch == 0:
                windowsize = [WindowWidth * 2, int(WindowWidth * frameratio * 2)]
                bitmap1, self.KeepTrack1 = self.TransformFrame(frame1, self.point1, magZoom, windowsize)
                bitmap2 = bitmap1
                # bitmap2, self.KeepTrack2 = self.TransformFrame(frame2, self.point2, magZoom,windowsize)
            elif moviewatch == 1:
                windowsize = [WindowWidth * 2, int(WindowWidth * frameratio * 2)]
                # bitmap1, self.KeepTrack1 = self.TransformFrame(frame1, self.point1, magZoom,windowsize)
                bitmap2, self.KeepTrack2 = self.TransformFrame(frame2, self.point2, magZoom, windowsize)
                bitmap1 = bitmap2
            bit1Height, bit1Width, channels = bitmap1.shape




            #######################################################################################
            ## subtracting forward frame so to better view a moving object

            if UseforwardFrameSubtraction == True:

                frameAdvance = 7

                if moviewatch == 0 or moviewatch == 2:
                    self.fastforward(frameAdvance + 1, self.capture1)

                    ret, frame1adv = self.capture1.read()
                    self.rewind(frameAdvance, self.capture1)
                    bitmap1advg, KeepTrack1adv = self.TransformFrame(frame1adv, self.point1, magZoom, windowsize)
                    bitmap1adv = numpy.copy(bitmap1advg)
                    bitmapgray = cv2.cvtColor(bitmap1, cv2.COLOR_BGR2GRAY)
                    bitmapadvgray = cv2.cvtColor(bitmap1adv, cv2.COLOR_BGR2GRAY)
                    del frame1adv, bitmap1advg, ret, KeepTrack1adv

                if moviewatch == 1 or moviewatch == 2:
                    self.fastforward(frameAdvance + 1, self.capture2)
                    ret, frame2adv = self.capture2.read()
                    self.rewind(frameAdvance, self.capture2)

                    bitmap2advg, KeepTrack2adv = self.TransformFrame(frame2adv, self.point2, magZoom, windowsize)

                    bitmap2adv = numpy.copy(bitmap2advg)
                    bitmapgray = cv2.cvtColor(bitmap2, cv2.COLOR_BGR2GRAY)
                    bitmapadvgray = cv2.cvtColor(bitmap2adv, cv2.COLOR_BGR2GRAY)
                    del frame2adv, bitmap2advg, ret, KeepTrack2adv


                # Find size of image1
                sz = bitmapgray.shape

                # Define the motion model
                warp_mode = cv2.MOTION_EUCLIDEAN

                # Define 2x3 or 3x3 matrices and initialize the matrix to identity
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    warp_matrix = numpy.eye(3, 3, dtype=numpy.float32)
                else:
                    warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)

                # Specify the number of iterations.
                number_of_iterations = 50;

                # Specify the threshold of the increment
                # in the correlation coefficient between two iterations
                termination_eps = 1e-10;

                # Define termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

                # Run the ECC algorithm. The results are stored in warp_matrix.
                (cc, warp_matrix) = cv2.findTransformECC(bitmapgray, bitmapadvgray, warp_matrix, warp_mode, criteria)

                if 0:
                    if warp_mode == cv2.MOTION_HOMOGRAPHY:
                        # Use warpPerspective for Homography
                        im2_aligned = cv2.warpPerspective(bitmap1adv, warp_matrix, (sz[1], sz[0]),
                                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    else:
                        # Use warpAffine for Translation, Euclidean and Affine
                        im2_aligned = cv2.warpAffine(bitmap1adv, warp_matrix, (sz[1], sz[0]),
                                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
                if 1:
                    if warp_mode == cv2.MOTION_HOMOGRAPHY:
                        # Use warpPerspective for Homography
                        im2_aligned = cv2.warpPerspective(bitmapadvgray, warp_matrix, (sz[1], sz[0]),
                                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    else:
                        # Use warpAffine for Translation, Euclidean and Affine
                        im2_aligned = cv2.warpAffine(bitmapadvgray, warp_matrix, (sz[1], sz[0]),
                                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

                bitmapgray=bitmapgray.astype(numpy.float32)
                im2_aligned = im2_aligned.astype(numpy.float32)

                im2_aligned = im2_aligned - bitmapgray + 10.0
                im2_aligned=im2_aligned*5
                im2_aligned[im2_aligned<0]=-im2_aligned[im2_aligned<0]
                im2_aligned[im2_aligned>255]=255
                im2_aligned = im2_aligned.astype(numpy.uint8)
                im2_aligned = cv2.cvtColor(im2_aligned, cv2.COLOR_GRAY2BGR)

                if moviewatch == 0 or moviewatch == 2:
                    bitmap1=im2_aligned
                if moviewatch == 1 or moviewatch == 2:
                    bitmap2=im2_aligned


            ###################################################################################################
            #
            if viewSquid == True:  # so this is all for squid

                if UseBlankCamera == False:

                    if usePnpProj == True:
                        MeshProj = newProjMesh
                        projPoints=newProjMesh
                    else:
                        MeshProj = self.ReturnMeshProjection(pcMat, RotMat, TranMat, dat1, cameraMatrix1, distCoeffs1,
                                                             width, height, makeInt)
                        # where fo I put this?
                        projPoints,insectPointArray = self.ReturnPointsection(ThreeDMarkedPoints, RotMat, TranMat, dat1, cameraMatrix1,
                                                             distCoeffs1,
                                                             makeInt, Tobject,frame1num,fwData)

                else:
                    if usePnpProj == True:  # is is after we press f and this is using datOrdAlt and not rotMat
                        RotMat = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
                        TranMat = numpy.zeros(3)
                        dat1 = datOrdAlt
                        MeshProj = self.ReturnMeshProjection(pcMat, RotMat, TranMat, datOrdAlt, cameraMatrix1,
                                                             distCoeffs1,
                                                             width, height, makeInt)
                        # where fo I put this?
                        projPoints,insectPointArray = self.ReturnPointsection(ThreeDMarkedPoints, RotMat, TranMat, datOrdAlt, cameraMatrix1,
                                                             distCoeffs1,
                                                             makeInt, Tobject,frame1num,fwData)


                    else:  # this is using rotations and not the datOrd.
                        #                        datOrd = numpy.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
                        datOrd = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

                        if viewSquidMeshStuff==True:
                            try:
                                if indextrack!=SquidIndex:
                                    fwSTL = fw['STL']
                                    ThereIsSTLData=True
                                    datOrdAlt=fwSTL[str(int(frame1num))][filenameSTL]["datOrdAlt"][:]
                                    if 1: #qwq
                                        try:
                                            datOrdAltAlt = fwSTL[str(int(frame1num))][filenameSTL]["datOrdAltAlt"][:]
                                            AdjustSquidPitch = fwSTL[str(int(frame1num))][filenameSTL][
                                                "PitchTheta"].value
                                        except:
                                            datOrdAltAlt = fwSTL[str(int(frame1num))][filenameSTL]["datOrdAlt"][:]
                                            AdjustSquidPitch = 0
                                        resx0=AdjustSquidPitch
                                        fishRoll, fishRelAzimuth, fishPitch = self.goingThoughVariousCameraParameters(
                                            fwData, frame1num, datOrdAlt,
                                            datOrdAltAlt, CameraRoll, RotAroundY,
                                            resx0, RotMatCopy)


                                    TwoDMarkedPoints2 = []
                                    ThreeDMarkedPoints2 = []

                                    for i in range(len(ThreeDMarkedPoints)):
                                        insectNum = Tobject + str(int(i + 1))
                                        try:
                                            TwoDMarkedPoints2.append(
                                                [fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][0],
                                                 fwData[cameraNames[moviewatch]][insectNum][str(int(frame1num))][1]])
                                        except:
                                            None

                                    TwoDMarkedPoints2 = numpy.array(TwoDMarkedPoints2)
                                    CircleArray, Outline, twoAve = self.getMeshOutline(TwoDMarkedPoints2, datOrdAlt,
                                                                                       cameraMatrix1,
                                                                                       distCoeffs1, obbTree)
                                    indextrack=SquidIndex
                                    #print indextrack, SquidIndex, "indextrack,SquidIndex at the end"

                            except:
                            #if 0:
                                ThereIsSTLData = False



                        datOrd=datOrdAlt


                        if 0:
                            RotMat[0, 0]=datOrdAlt[0]
                            RotMat[1, 0]=datOrdAlt[1]
                            RotMat[2, 0]=datOrdAlt[2]
                            RotMat[0, 1]=datOrdAlt[3]
                            RotMat[1, 1]=datOrdAlt[4]
                            RotMat[2, 1]=datOrdAlt[5]
                            RotMat[0, 2]=datOrdAlt[6]
                            RotMat[1, 2]=datOrdAlt[7]
                            RotMat[2, 2]=datOrdAlt[8]

                            # camera origin
                            TranMat[0]=datOrdAlt[9]
                            TranMat[1]=datOrdAlt[10]
                            TranMat[2]=datOrdAlt[11]
                        RotMat = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
                        TranMat = numpy.zeros(3)
                        #print "here ord"
                        MeshProj = self.ReturnMeshProjection(pcMat, RotMat, TranMat, datOrdAlt, cameraMatrix1, distCoeffs1,
                                                             width, height, makeInt)
                        projPoints,insectPointArray = self.ReturnPointsection(ThreeDMarkedPoints, RotMat, TranMat, datOrdAlt, cameraMatrix1,
                                                             distCoeffs1,
                                                             makeInt, Tobject,frame1num,fwData)







                    ################################  marking anotations on images

            if FillMapDic==True:
                if self.isPaused == False:
                    if FillMapDicframe!=0 or FillMapDicframe!=frame1num:
                        MapErrorDicMeauredPoint, MapErrorDicMapPoint,CameraOriginArray = self.FillMapDictionaries(fwData, cameraNames, f1, f2,
                                                                                                frame1num, h5SquidDelay,
                                                                                                frame2num, dat2,
                                                                                                cameraMatrix2, distCoeffs2,
                                                                                                dat1, cameraMatrix1,
                                                                                                distCoeffs1,
                                                                                                MapErrorDicMeauredPoint,
                                                                                                MapErrorDicMapPoint,CameraOriginArray)

                        FillMapDicframe=frame1num


            #############################################################################################
            #This is a functionalitiy that overlays the paths onto the image.   This requires at this point specifying the
            #paths to be ploted.   There are arrows that give the path direction.
            #KalmanOverlay=False
            if KalmanOverlay == True:


                Cam2Kalarr=[]


                if 1:  # amber wings
                    Cam2KalMax = 2.5
                    Cam2KalMin = .35
                    maxlevel = 1
                    minlevel = .1
                if 0:# insect one
                    InsectOrder = [
                        "insect0_insect0",
                        "insect1_insect1",
                        "insect2_insect2",
                        "insect4_insect4",
                        "insect5_insect5",
                        "insect6_insect6",
                        "insect8_insect8",
                        "insect3_insect3",
                        "insect7_insect7",
                    ]
                    Cam2KalMax = 1.2556
                    Cam2KalMin = 0.2780
                    maxlevel = 1
                    minlevel = .1
                if 0:#fish one
                    InsectOrder = [
                        "insect0_insect0",
                        "insect3_insect3",
                        "insect1_insect1",
                        "insect4_insect4",
                        "insect5_insect5",
                        "insect8_insect8",
                        "insect2_insect2",
                        "insect6_insect6",
                        "insect7_insect7",
                        "insect9_insect9",
                        "insect10_insect10",
                        "insect11_insect11",
                        "insect12_insect12"
                    ]
                    Cam2KalMax = 3.2828
                    Cam2KalMin = 0.924888
                    maxlevel = 1
                    minlevel = .1
                if moviewatch==0:
                    datt=dat1
                if moviewatch==1:
                    datt=dat2

                arroworline = 0

                # getting all the path stuff
                #for insectCom in range(len(InsectOrder)):

                    #if 1:
                        #for jj in fwData[cameraNames[2]][InsectOrder[insectCom]]:

                colorarray = plt.cm.jet(numpy.linspace(0, 1, len(PlotDesciptiveDic["Insectnumbers1"])))
                for j in range(len(PlotDesciptiveDic["Insectnumbers1"])):
                    insectCom=j
                    ###################################
                    # Initializing the dictionary data
                    for ii in range(PlotDesciptiveDic["insectNumGroupNumber"]):

                        for i in range(len(DictA["dt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)])-1):
                            try:
                                ResultRot = numpy.array(
                                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])
                                ResultRot=ResultRot.T

                                ResultTran = numpy.array(
                                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                            except:
                                ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])#l;l

                                ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

                            xto = DictA["xt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i]
                            yto = DictA["yt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i]
                            zto = DictA["zt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i]

                            xKal = numpy.array([xto,yto,zto])


                            xKalA0 = numpy.matmul(numpy.linalg.inv(ResultRot.T), xKal-ResultTran)
                            cameraOrgn1 = numpy.array([datt[9], datt[10], datt[11]])
                            Cam2Kal = numpy.linalg.norm(cameraOrgn1 - xKalA0)
                            xKalA0 = numpy.transpose(numpy.array([xKalA0]))

                            xt1 = DictA["xt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i+1]
                            yt1 = DictA["yt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i+1]
                            zt1 = DictA["zt" + str(
                                PlotDesciptiveDic["Insectnumbers" + str(ii + 1)][j]) + "_" + str(
                                ii + 1)][i+1]

                            xKal = numpy.array([xt1, yt1, zt1])

                            xKalA1 = numpy.matmul(numpy.linalg.inv(ResultRot.T), xKal - ResultTran)

                            xKalA1 = numpy.transpose(numpy.array([xKalA1]))

                            #if it>74 and it<82:
                            if 0:
                                try:
                                    InterPointC = numpy.array(fwData[cameraNames[2]][InsectOrder[insectCom]][jj]['MovingAveragePoint'])

                                    cameraOrgn1 = numpy.array([datt[9], datt[10], datt[11]])
                                    Cam2Kal=numpy.linalg.norm(cameraOrgn1-InterPointC)
                                    Cam2Kalarr.append(Cam2Kal)
                                    InterPointC = numpy.transpose(numpy.array([InterPointC]))

                                    InterPointC2 = numpy.array(
                                        fwData[cameraNames[2]][InsectOrder[insectCom]][str(int(jj)+1)]['MovingAveragePoint'])
                                    InterPointC2 = numpy.transpose(numpy.array([InterPointC2]))

                                except:
                                    continue
                            if 1:
                                RotMatC = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
                                TranMatC = numpy.zeros(3)
                                #projecting the points
                                #seems like there are few things to process for different cameras.
                                InterPointCPro = self.ReturnMeshProjection(xKalA0, RotMatC, TranMatC, datt,
                                                                     cameraMatrix1, distCoeffs1,
                                                                     width, height, makeInt)
                                xo, yo = self.TransformPoint(InterPointCPro[0][0], InterPointCPro[0][1], self.KeepTrack1)

                                if 1:
                                    InterPointC2Pro = self.ReturnMeshProjection(xKalA1, RotMatC, TranMatC, datt,
                                                                               cameraMatrix1, distCoeffs1,
                                                                               width, height, makeInt)
                                    xo2, yo2 = self.TransformPoint(InterPointC2Pro[0][0], InterPointC2Pro[0][1],
                                                                 self.KeepTrack1)

                                #making bounds.
                                bbb=(minlevel-Cam2KalMax*maxlevel/Cam2KalMin)/(1-Cam2KalMax/Cam2KalMin)
                                aaa=(maxlevel-bbb)/Cam2KalMin
                                Cam2KalAdd = Cam2Kal *aaa+bbb

                                #checking if we want to display this
                                if Cam2KalAdd>maxlevel:
                                    Cam2KalAdd=maxlevel
                                if Cam2KalAdd<minlevel:
                                    Cam2KalAdd=minlevel
                                if 0:
                                    cv2.circle(bitmap1, (int(xo), int(yo)), int(Cam2KalAdd * 10),
                                               (int(colorarray[insectCom][0] * 255 * Cam2KalAdd), int(colorarray[insectCom][1] * 255 * Cam2KalAdd),
                                                int(colorarray[insectCom][2] * 255 * Cam2KalAdd)), -1)

                                if xo>0 and yo>0 and xo2>0 and yo2>0 and xo<width and xo2<width and yo<height and yo2<height:

                                    if arroworline<3:
                                        cv2.line(bitmap1, (int(xo), int(yo)),(int(xo2), int(yo2)), thickness=int(Cam2KalAdd * 3),
                                                   color=(int(colorarray[insectCom][0] * 255 * Cam2KalAdd),
                                                    int(colorarray[insectCom][1] * 255 * Cam2KalAdd),
                                                    int(colorarray[insectCom][2] * 255 * Cam2KalAdd)))
                                        arroworline+=1
                                    else:
                                        if numpy.sqrt((xo - xo2) ** 2 + (yo - yo2) ** 2) > 10:
                                            print xo, yo, xo2, yo2,numpy.sqrt((xo - xo2) ** 2 + (yo - yo2) ** 2)
                                        else:
                                            cv2.arrowedLine(bitmap1, (int(xo), int(yo)),(int(xo2), int(yo2)), thickness=int(Cam2KalAdd * 3),
                                                       color=(int(colorarray[insectCom][0] * 255 * Cam2KalAdd),
                                                        int(colorarray[insectCom][1] * 255 * Cam2KalAdd),
                                                        int(colorarray[insectCom][2] * 255 * Cam2KalAdd)),tipLength=5)
                                        arroworline=0



                #Cam2Kalarr=numpy.array(Cam2Kalarr)
                #print Cam2Kalarr.max()
                #print Cam2Kalarr.min()

            #################################################################################################
            #Here we are plotting the VSLAM lines that solve the triangulation algorithim from the point selected in the
            #coincident frame.
            #viewGuidingLines=False
            if useH5 == True and viewTrackingThings==True:
                if viewGuidingLines==True:  # moviewatch==0 and moviewatch == 2:
                    point1n = numpy.zeros(2)
                    point2n = numpy.zeros(2)
                    pointsIn2frames=False
                    try:
                        frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                        point1n[0] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][0]
                        point1n[1] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][1]
                        pointsIn2frames = True
                    except:
                        None
                    try:
                        frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                        point2n[0] = \
                        fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(frame1num) - frameDelay)][0]
                        point2n[1] = \
                        fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(frame1num) - frameDelay)][1]
                        pointsIn2frames = True
                    except:
                        None

                    if pointsIn2frames:

                        #line1to2, line2to1 = self.returnMutualPointProjection2(dat1, dat2, cameraMatrix1,
                        #                                             distCoeffs1,
                        #                                             cameraMatrix2, distCoeffs2,
                        #                                             width, height, point1n, point2n)
                        line1to2, line2to1 = self.returnMutualPointProjection(dat1, dat2, cameraMatrix1, distCoeffs1,
                                                                              cameraMatrix2, distCoeffs2,
                                                                              width, height, point1n, point2n)
                        # line1to2, line2to1
                        if moviewatch == 0 or moviewatch == 2:

                            for ii in range(len(line2to1)):
                                xo, yo = self.TransformPoint(line2to1[ii][0], line2to1[ii][1], self.KeepTrack1)
                                cv2.circle(bitmap1, (int(xo), int(yo)), 2, (0, 0, 255), -1)
                        if moviewatch == 1 or moviewatch == 2:
                            for ii in range(len(line1to2)):
                                xo, yo = self.TransformPoint(line1to2[ii][0], line1to2[ii][1], self.KeepTrack2)
                                cv2.circle(bitmap2, (int(xo), int(yo)), 2, (0, 0, 255), -1)



            if ViewDualGoProSquid==True:
                ##################################################################
                #Here we are plotting the projection of the mesh onto the image
                #print "here ViewDualGoProSquid"
                if 1:
                    if moviewatch == 0:
                        MeshProj = self.ReturnMeshProjection(pcMat, RotMatSQ, TranMatSQ, dat1, cameraMatrix1,
                                                             distCoeffs1,
                                                             width, height, makeInt)
                    if moviewatch == 1:
                        MeshProj = self.ReturnMeshProjection(pcMat, RotMatSQ, TranMatSQ, dat2, cameraMatrix1,
                                                             distCoeffs1,
                                                             width, height, makeInt)

                    #print "MeshProj", MeshProj

                if moviewatch == 0 or moviewatch == 2:
                    for ii in range(len(MeshProj)):#mesh projection
                        xo, yo = self.TransformPoint(MeshProj[ii][0], MeshProj[ii][1], self.KeepTrack1)
                        cv2.circle(bitmap1, (int(xo), int(yo)), 2, (0, 0, 255), -1)
                if moviewatch == 1 or moviewatch == 2:

                    for ii in range(len(MeshProj)):#mesh projection
                        xo, yo = self.TransformPoint(MeshProj[ii][0], MeshProj[ii][1], self.KeepTrack2)
                        cv2.circle(bitmap2, (int(xo), int(yo)), 2, (0, 0, 255), -1)
                pointssquidreal = numpy.array(
                    [[0, 3.55998, -0.02321], [0.56313, -0.89247, 0.22152], [-0.56313, -0.89247, 0.22152]])
                if moviewatch==0:
                    pointssquidreal=numpy.transpose(pointssquidreal)
                    pointssquidrealProj = self.ReturnMeshProjection(pointssquidreal, RotMatSQ, TranMatSQ, dat1, cameraMatrix1,
                                                         distCoeffs1,
                                                         width, height, makeInt)

                    xo, yo = self.TransformPoint(pointssquidrealProj[0][0], pointssquidrealProj[0][1], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (255, 0, 255), -1)

                    xo, yo = self.TransformPoint(pointssquidrealProj[1][0], pointssquidrealProj[1][1], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (255, 0, 0), -1)

                    xo, yo = self.TransformPoint(pointssquidrealProj[2][0], pointssquidrealProj[2][1], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (0, 255, 0), -1)


            ######################################################################################################
            if viewSquid == True and viewSquidMeshStuff==True:# and ThereIsSTLData==True:



                ##################################################################
                #projected points of the marked points
                for ii in range(len(projPoints)):#
                    xo, yo = self.TransformPoint(projPoints[ii][0], projPoints[ii][1], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (50, 255, 50), -1)

                ##################################################################
                #Here we are plotting the center of the frame as defined by the camera calibration in the cameraMatrix
                if 1:
                    xo, yo = self.TransformPoint(cameraMatrix1[0, 2], cameraMatrix1[1, 2], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (2, 30, 240), -1)

                ##################################################################
                #Here we are plotting the projection of the mesh onto the image
                for ii in range(len(MeshProj)):#mesh projection
                    xo, yo = self.TransformPoint(MeshProj[ii][0], MeshProj[ii][1], self.KeepTrack1)
                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (0, 0, 255), -1)


                ##################################################################
                #
                if not newProjMesh:
                    None
                else:
                    for ii in range(len(newProjMesh)):
                        xo, yo = self.TransformPoint(newProjMesh[ii][0], newProjMesh[ii][1], self.KeepTrack1)
                        # cv2.circle(bitmap1, (int(xo), int(yo)), 2, (0, 255, 0), -1)


                ##################################################################
                #Here we are plotting the outline of the mesh.   outline is the standard outline (plotted in Blue) and corEdgepnts are the
                #values of the points that are used for the edge PnP fit. They are in purple
                if len(CircleArray) > 1:

                    for ii in range(len(CircleArray)):#outline of the mesh
                        xo, yo = self.TransformPoint(Outline[ii][0], Outline[ii][1], self.KeepTrack1)
                        cv2.circle(bitmap1, (int(xo), int(yo)), 2, (255, 0, 0), -1)
                    for ii in range(len(corEdgepnts)):
                        xo, yo = self.TransformPoint(corEdgepnts[ii][0], corEdgepnts[ii][1], self.KeepTrack1)
                        cv2.circle(bitmap1, (int(xo), int(yo)), 2, (200, 30, 130), -1)


                ##################################################################
                #Here we are plotting the pixels in the selected ROI over the mesh in light blue
                if 1:
                    try:
                        fwROI = fw['ROI']
                        fwSTL = fw['STL']
                        for ik in fwROI[RoiString]["ROI cell list"]:
                            try:
                                for pixarr in fwSTL[str(int(frame1num))][filenameSTL][str(int(ik))]["PixelArray"]:
                                    xo, yo = self.TransformPoint(pixarr[0], pixarr[1], self.KeepTrack1)
                                    cv2.circle(bitmap1, (int(xo), int(yo)), 2, (250, 200, 20), -1)
                            except:
                                None
                    except:
                        None



            ###########################################################################
            ###########################################################################
            if UseSquidExcel==True:
                if 1:

                    if frame1num!=frame1numStore2:
                        try:
                            print "FishHeading", fwData["direction"][str(frame1num)]["FishHeading"].value
                            print "FishPitch", fwData["direction"][str(frame1num)]["FishPitch"].value
                            print "FishRoll",fwData["direction"][str(frame1num)]["FishRoll"].value
                            print "FishCameraRelAzimuth", fwData["direction"][str(frame1num)]["FishCameraRelAzimuth"].value
                            print "SolarHeadingFromTime", fwData["direction"][str(frame1num)]["SolarHeadingFromTime"].value
                            print "FishSolarHeading", fwData["direction"][str(frame1num)]["FishSolarHeading"].value
                        except:
                            None
                        frame1numStore2=frame1num
                    else:
                        None

            ###########################################################################
            ###########################################################################
            #the folowing are preparing and ploting the text on the image.


            ########################################################################
            #toward or away from sun
            if UseSquidExcel==True:
                try:
                    fwData["direction"]
                except:
                    fwData.create_group("direction")
                try:
                    towardOrAwayFromSun = fwData["direction"][str(frame1num)]["TorA"][0]
                except:
                    None
                    if 0:
                        fwData["direction"].create_group(str(frame1num))
                        fwData["direction"][str(frame1num)].create_dataset("TorA",data=["?"])
                        towardOrAwayFromSun="?"


            if 1:
                try:
                    mirrorA = fwData[cameraNames[moviewatch]][mirrorCameraA[moviewatch]].value
                    #print mirrorA
                    mirrorIndicator=0
                    for mr in range(len(mirrorA)):
                        if mirrorA[mr][1] == insectnumit:
                            insectR = "insect" + str(mirrorA[mr][0])
                            MirrorString="Mir. from "+insectR
                            mirrorIndicator+=1
                            break
                        if mirrorA[mr][0] == insectnumit:
                            insectR = "insect" + str(mirrorA[mr][1])
                            MirrorString="Mir. to "+insectR
                            mirrorIndicator+=1
                            break
                    if mirrorIndicator==0:
                        MirrorString=""

                except:
                    MirrorString = ""



            #################################################################################################
            ## looking if the there is any insect data
            if moviewatch == 0 or moviewatch == 2:
                try:
                    insectkeys = fwData[cameraNames[0]][Tobject + str(int(insectnumit))].keys()
                    if not insectkeys:
                        IsInsectEmpty = "Empty"
                    else:
                        IsInsectEmpty = ""
                except:
                    None

            if moviewatch == 1 or moviewatch == 2:
                try:
                    insectkeys = fwData[cameraNames[1]][Tobject + str(int(insectnumit))].keys()
                    if not insectkeys:
                        IsInsectEmpty = "Empty"
                    else:
                        IsInsectEmpty = ""
                except:
                    None


            #################################################################################################

            if UseNumbersForError == False:
                ErrorMode = ""
            else:
                ErrorMode = "Mes. Err."

            #################################################################################################

            if ClickIncrimentMode==True:
                ClickingMode="Mouse Mv"
            else:
                ClickingMode=""


            ############################################################################
            if viewTrackingThings==True: #write text
                if useH5==True:
                    useH5Str="h5good"
                else:
                    useH5Str = ""
                if moviewatch == 0 or moviewatch == 2:
                    messagestring = Tobject + str(int(insectnumit)) + " " + IsInsectEmpty + " " + cameraNames[
                        moviewatch] + " " + str(int(frame1num)) + " " + ErrorMode + " " + ClickingMode+" "+useH5Str+" "+MirrorString
                    if UseSquidExcel == True:
                        messagestring += " " + RoiString+ " "+towardOrAwayFromSun
                    cv2.putText(bitmap1, str(messagestring), (int(5), int(30)), font, 1,
                                (10, 120, 250), 1, cv2.LINE_AA)
                if moviewatch == 1 or moviewatch == 2:
                    messagestring = Tobject + str(int(insectnumit)) + " " + IsInsectEmpty + " " + cameraNames[
                        moviewatch] + " " + str(int(frame2num)) + " " + ErrorMode + " " + ClickingMode+" "+useH5Str+" "+MirrorString
                    if UseSquidExcel == True:
                        messagestring += " " + RoiString+ " "+towardOrAwayFromSun
                    cv2.putText(bitmap2, str(messagestring), (int(5), int(30)), font, 1,
                                (10, 120, 250), 1, cv2.LINE_AA)


            #################################################################################################

            if viewSquidMeshStuff and viewTrackingThings==True: #write text:
                for ii in range(insectMaxNum):

                    if moviewatch == 0 or moviewatch == 2:
                        try:
                            point1fromH5[0] = fwData[cameraNames[0]][Tobject + str(int(ii))][str(int(frame1num))][0]

                            point1fromH5[1] = fwData[cameraNames[0]][Tobject + str(int(ii))][str(int(frame1num))][1]
                            if UseFWError == True:
                                ErrorCircle = fwData[cameraNames[0]][Tobject + str(int(ii))][str(int(frame1num))][2]
                                if ErrorCircle == 0:
                                    ErrorCircle = 3

                            xo, yo = self.TransformPoint(point1fromH5[0], point1fromH5[1], self.KeepTrack1)
                            cv2.circle(bitmap1, (int(xo), int(yo)), int(ErrorCircle * self.KeepTrack1[5]),
                                       (insectColor[ii][0], insectColor[ii][1], insectColor[ii][2]), 1)
                            font2 = cv2.FONT_HERSHEY_SIMPLEX

                            cv2.putText(bitmap1, str(int(ii)), (int(xo + 40), int(yo + 40)), font2, 1,
                                        (insectColor[ii][0], insectColor[ii][1], insectColor[ii][2]), 2, cv2.LINE_AA)


                        except:
                            None

                    if moviewatch == 1 or moviewatch == 2:

                        try:
                            point2fromH5[0] = \
                                fwData[cameraNames[1]][Tobject + str(int(ii))][str(int(frame2num))][0]

                            point2fromH5[1] = \
                                fwData[cameraNames[1]][Tobject + str(int(ii))][str(int(frame2num))][1]
                            if UseFWError == True:
                                ErrorCircle = fwData[cameraNames[1]][Tobject + str(int(ii))][str(int(frame2num))][2]
                                if ErrorCircle == 0:
                                    ErrorCircle = 3

                            xo, yo = self.TransformPoint(point2fromH5[0], point2fromH5[1], self.KeepTrack2)
                            cv2.circle(bitmap2, (int(xo), int(yo)), int(ErrorCircle * self.KeepTrack2[5]),
                                       (insectColor[ii][0], insectColor[ii][1], insectColor[ii][2]), 1)

                            font2 = cv2.FONT_HERSHEY_SIMPLEX

                            cv2.putText(bitmap2, str(int(ii)), (int(xo + 40), int(yo + 40)), font2, 1,
                                        (insectColor[ii][0], insectColor[ii][1], insectColor[ii][2]), 2, cv2.LINE_AA)


                        except:
                            None

            #################################################################################################

            if UseSquidExcel==True and DicBInt>0:
                for i in range(DicBInt):
                    if i>=1:
                        xo, yo = self.TransformPoint((ClickedDicB[i-1][0]),(ClickedDicB[i-1][1]), self.KeepTrack1)
                        xo2, yo2 = self.TransformPoint((ClickedDicB[i][0]),(ClickedDicB[i][1]), self.KeepTrack1)
                        cv2.line(bitmap1,(int(xo), int(yo)),(int(xo2), int(yo2)),(100,200,0),5)
            elif ThreeDlineDicBUse==True:
                if moviewatch==0:
                    for i in range(DicBInt1):
                        if i>=1:
                            xo, yo = self.TransformPoint((ThreeDlineDicB1[i-1][0]),(ThreeDlineDicB1[i-1][1]), self.KeepTrack1)
                            xo2, yo2 = self.TransformPoint((ThreeDlineDicB1[i][0]),(ThreeDlineDicB1[i][1]), self.KeepTrack1)
                            cv2.line(bitmap1,(int(xo), int(yo)),(int(xo2), int(yo2)),(100,200,0),5)
                if moviewatch == 1:
                    for i in range(DicBInt2):
                        if i>=1:
                            xo, yo = self.TransformPoint((ThreeDlineDicB2[i-1][0]),(ThreeDlineDicB2[i-1][1]), self.KeepTrack2)
                            xo2, yo2 = self.TransformPoint((ThreeDlineDicB2[i][0]),(ThreeDlineDicB2[i][1]), self.KeepTrack2)
                            cv2.line(bitmap2,(int(xo), int(yo)),(int(xo2), int(yo2)),(100,200,0),5)


            #################################################################################################
            #################################################################################################
            #here we are making the images visible.

            if moviewatch == 0 or moviewatch == 2:
                xo, yo = self.TransformPoint(self.point1[0], self.point1[1], self.KeepTrack1)
                movWin1xx = movWin1x
                movWin1yy = movWin1y

                cv2.imshow("movie1", bitmap1)
                cv2.moveWindow("movie1", movWin1xx, movWin1yy)
                cv2.setMouseCallback('movie1', theapp.select_point, param=(self.point1, self.KeepTrack1, self.clicked1))

            if moviewatch == 1 or moviewatch == 2:
                xo, yo = self.TransformPoint(self.point2[0], self.point2[1], self.KeepTrack2)
                movWin2xx = movWin1x
                movWin2yy = movWin1y

                cv2.imshow("movie2", bitmap2)
                cv2.moveWindow("movie2", movWin2xx, movWin2yy)
                cv2.setMouseCallback("movie2", theapp.select_point, param=(self.point2, self.KeepTrack2, self.clicked2))



            #################################################################################################

            if 1:
                #########################################################################################
                #this is a functionality that makes the video small when z is pressed and then rewinds and plays a few frames
                #to make the targe more visible.
                if 1:#new motion thing
                    if ClickIncrimentMode == True and DoRewindInZ==True and ContinueClickFrameMode==False:
                        if 1:
                            if settingTheRewindView==True: # here we are taking a few frames to rewind before another point is selected  so that movement can be seen.
                                if moviewatch == 0:
                                    RVframenumInit = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                                    self.rewind(RVvalue, self.capture1) # just rewinds a little bit
                                    ret, frame1 = self.capture1.read()
                                if moviewatch == 1:
                                    RVframenumInit = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                                    self.rewind(RVvalue, self.capture2)
                                    ret, frame2 = self.capture2.read()
                                settingTheRewindView=False #it gets ready to go through the regular process of the loop to display the frames.
                                self.isPaused = False
                                DontUpdatePoint = True  # I don't think this has developed enitrely yet.
                                viewSquidMeshStuff=False # turns off annotations duing the replay
                            else:
                                if moviewatch == 0:# no rewinding is happening
                                    RVframenum=self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                                if moviewatch == 1:
                                    RVframenum = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                                print RVframenum,RVframenumInit,"RVframenum,RVframenumInit"
                                print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                                    cv2.CAP_PROP_POS_FRAMES),self.capture1.get(cv2.CAP_PROP_POS_FRAMES)-self.capture2.get(
                                    cv2.CAP_PROP_POS_FRAMES)
                                if RVframenum==RVframenumInit:
                                    self.isPaused = True
                                    ContinueClickFrameMode=True
                                    DontUpdatePoint=False
                                    viewSquidMeshStuff=True
                    elif ClickIncrimentMode == True and DoRewindInZ==True and ContinueClickFrameMode == True:
                        self.isPaused = True
                    elif ClickIncrimentMode == True and DoRewindInZ==False:
                        self.isPaused = True

                #########################################################################################
                #we were thinging of making one that goes between cameras.
                if 1: #normal one camera at a time
                    if ClickIncrimentMode == True and ContinueClickFrameMode== True:  # looking to see if the mouse has moved.

                        if numpy.linalg.norm(self.clicked1-clicked1m1)>0.0001 or numpy.linalg.norm(self.clicked2-clicked2m1)>0.0001:

                            MouseMoved=True
                            MouseHasNotMovedIn=0

                        else:
                            MouseHasNotMovedIn+=1
                            MouseMoved=False

                        if MouseMoved==False and MouseHasNotMovedIn==5:
                            self.isPaused = False
                            print "made it"
                            print frame2num
                            if moviewatch == 0:

                                frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)

                                try:
                                    g2 = fwData[cameraNames[0]][Tobject + str(int(insectnumit))].create_dataset(
                                        str(int(frame1num)), data=[self.clicked1[1], self.clicked1[2], 4, 4])

                                except: # recording the data
                                    fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][0] = \
                                        self.clicked1[1]
                                    fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][1] = \
                                        self.clicked1[2]


                                if FrameSkipInc > 1 and Frames2Average > 1 and frame2num -(int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc) - 1)==0:
                                    try:
                                        previouspnt = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][
                                            str(int(frame1num - FrameSkipInc))]
                                    except:
                                        previouspnt = [self.clicked2[1],self.clicked2[2]]
                                else:
                                    try: # predicting the next point
                                        previouspnt=fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num-1))]
                                    except:
                                        previouspnt = [self.clicked1[1],self.clicked1[2]]
                                xpixdif=self.clicked1[1]-previouspnt[0]
                                ypixdif=self.clicked1[2]-previouspnt[1]
                                self.point1[0]=self.clicked1[1]+xpixdif
                                self.point1[1]=self.clicked1[2]+ypixdif


                            if moviewatch == 1:

                                frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)

                                try:
                                    g2 = fwData[cameraNames[1]][Tobject + str(int(insectnumit))].create_dataset(
                                        str(int(frame2num)), data=[self.clicked2[1], self.clicked2[2], 4, 4])

                                except:
                                    fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(frame2num))][0] = \
                                        self.clicked2[1]
                                    fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(frame2num))][1] = \
                                        self.clicked2[2]

                                if FrameSkipInc > 1 and Frames2Average > 1 and frame2num -(int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc) - 1)==0:
                                    try:
                                        previouspnt = fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                            str(int(frame2num - FrameSkipInc))]
                                    except:
                                        previouspnt = [self.clicked2[1],self.clicked2[2]]

                                else:

                                    try:
                                        previouspnt = fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                            str(int(frame2num - 1))]
                                    except:
                                        previouspnt = [self.clicked2[1],self.clicked2[2]]
                                xpixdif = self.clicked2[1] - previouspnt[0]
                                ypixdif = self.clicked2[2] - previouspnt[1]
                                self.point2[0] = self.clicked2[1] + xpixdif
                                self.point2[1] = self.clicked2[2] + ypixdif
                            if DoRewindInZ==True:
                                ContinueClickFrameMode=False
                                settingTheRewindView=True
                            #Frames2Average
                            if FrameSkipInc>1 and Frames2Average>1:
                                print frame2num,(int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc) - 1),Frames2Average
                                if frame2num-(int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc) - 1)>Frames2Average:
                                    if 0:
                                        print "incrament to next"

                                        self.capture1.set(cv2.CAP_PROP_POS_FRAMES, int(
                                            numpy.floor((frame2num+ FrameSkipInc) / FrameSkipInc) * FrameSkipInc) - 1 + frameDelay)
                                        #ret, frame1 = self.capture1.read()

                                        self.capture2.set(cv2.CAP_PROP_POS_FRAMES,
                                                          int(numpy.floor((frame2num+ FrameSkipInc) / FrameSkipInc) * FrameSkipInc) - 1)
                                        #ret, frame2 = self.capture2.read()

                                        #ret, frame2 = self.capture2.read()
                                    if 1:
                                        print "done"

            clicked1m1=numpy.copy(self.clicked1)
            clicked2m1 = numpy.copy(self.clicked2)



            #####################################################################
            ##
            if 0:
                #qwq
                if frame1numtrack2!=frame1num:
                    # save things
                    frame1numtrack2=frame1num
                    saveFile = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/ImagesForSquidAnalysis/AolpIMUGraphs/"
                    aolpBKArr = numpy.load(saveFile + ActualFileName1[0] + "_aolpBKArr" + ".npy")
                    greenIntenArr = numpy.load(saveFile + ActualFileName1[0] + "_greenIntenArr" + ".npy")
                    bkgrndindexArr = numpy.load(saveFile + ActualFileName1[0] + "_bkgrndindexArr" + ".npy")

                    fig5 = plt.figure(figsize=(6, 6))
                    plt.subplot(211)
                    plt.plot(bkgrndindexArr, aolpBKArr, label="", linestyle="", marker=".")
                    plt.plot([ActualFileName1FN/20.0+20],[0],label="",color=(0,1,0), linestyle="", marker="x")
                    plt.subplot(212)
                    plt.plot(bkgrndindexArr, greenIntenArr, label="", linestyle="", marker=".")
                    plt.show()

            if 0:
                print "file name", ActualFileName1[0]
                print "index", ActualFileName1FN/20.0



            #####################################################################################################
            #This functionality is turned on by pressing s when saveImagesAsIs is set for 2.   This will save the rendered
            #images of the analysis program with all the visible anotations to the folder specified in SingleFramesImages.
            if saveImagesAsIs==True:
                #qwq


                PlotDesciptiveDicMade = False
                #pngFileName = SingleFramesImages + "/" + ActualFileName1[0] + "_" + str(int(frame1num)) + ".png"
                try:
                    print PlotDesciptiveDic["Path2SaveImages"]
                    PlotDesciptiveDicMade=True
                except:
                    print"Plot function has not been initiated for save folder"

                if PlotDesciptiveDicMade==True:
                    if PlotDesciptiveDic["Path2SaveImages"]!="":
                        print "Saving Images"
                        pngFileName = PlotDesciptiveDic["Path2SaveImages"] +"/"+"AnnotatedImages" + "/" + ActualFileName1[0] + "_" + str(int(frame1num)) + ".png"

                        #if not os.path.isdir(SingleFramesImages):
                        #    os.mkdir(SingleFramesImages)
                        if not os.path.isdir(PlotDesciptiveDic["Path2SaveImages"] +"/"+"AnnotatedImages"):
                            os.mkdir(PlotDesciptiveDic["Path2SaveImages"] +"/"+"AnnotatedImages")

                        cv2.imwrite(
                            pngFileName,
                            bitmap1)
                    else:
                        print "no save folder specified"


            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################


            #########################################################################################
            ########    the keyboard interation




            #Basic protocols for using this program:

            #For the VSLAM tracking part:

            #Get all of the information correct on




            #For the mesh fitting or Squid part:



            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################
            ######################################################################################################################
            #print "looped"
            c = cv2.waitKey(int(self.waitTime1)) % 0x100



            #############################################################
            #here we use tkinter to make a simple dialog that takes the user to the frame desired.
            #opens a dialog into which you can put in the desired frame you want to move to. This does not work on a mac.
            if c is 120:  # 120 is x, choose SquidIndex  #  does not work on mac
                if 0:
                    root = tk.Tk()
                    root.withdraw()
                    answer = int(tkSimpleDialog.askinteger("Input", "Input Frame number", parent=root))
                    if UseSquidExcel==False:
                        FrameMax=length1
                    if UseSquidExcel==True:
                        FrameMax=xlrownum
                    while answer < 0 or answer > FrameMax or answer is None:
                        tkMessageBox.showerror("Error", "Input a number in range.")
                        answer = int(tkSimpleDialog.askinteger("Input", "Input Frame number", parent=root))
                if 1:
                    answer=raw_input("Type the frame you wish to go to and press enter: ")
                    isnotandint=True
                    if UseSquidExcel==False:
                        FrameMax=length1
                    if UseSquidExcel==True:
                        FrameMax=xlrownum
                    while isnotandint:
                        try:
                            answer=int(answer)
                            if  answer < 0 or answer > FrameMax:
                                answer = raw_input("please enter an integer within the range of 0 to "+str(FrameMax)+" for the frame number and press enter: ")
                            else:
                                isnotandint=False
                        except:
                            answer = raw_input("please enter an integer for the frame number and press enter: ")

                if UseSquidExcel == False:
                    self.capture1.set(cv2.CAP_PROP_POS_FRAMES, answer-1 + frameDelay)
                    ret, frame1 = self.capture1.read()

                    self.capture2.set(cv2.CAP_PROP_POS_FRAMES, answer-1)
                    ret, frame2 = self.capture2.read()

                    frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                    frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)

                    print "FRAME:",frame2num,frame1num

                if UseSquidExcel == True:
                    SquidIndex = answer
                    setFrame = 1


            #############################################################
            #sets the zoom to none
            if c is 103:  # 103 is g  restores zoom to 1
                magZoom=1.0

            #############################################################
            #deletes the selected point for the selected insect on the current frame only.
            if c is 35:  # 35 is end   deletes an individual insect number
                print "made it to 35"
                if moviewatch == 0 or moviewatch == 2:
                    try:
                        del fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][
                            str(int(frame1num))]
                    except:
                        None
                if moviewatch == 1 or moviewatch == 2:
                    try:
                        del fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))][
                            str(int(frame2num))]
                    except:
                        None

            #############################################################
            #Deletes the entire path of the current insect number on the current video frame, and sets the insect number
            # to empty.
            if c is 42:  # 42 is *  deletes and entire insect number
                del fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))]
                try:
                    fwData[cameraNames[moviewatch]].create_group(Tobject + str(int(insectnumit)))
                    # print "made it", insectMaxNum, insectnumit
                except:
                    fwData[cameraNames[moviewatch]].get(Tobject + str(int(insectnumit)))
                    # print "it has it? ", insectMaxNum, insectnumit

            #############################################################
            #closes out the program, properly and safely closing all the H5 files and saving a backup of the H5 file.
            if c is 27:  # is ESC
                self.EndVideo = True
                if useH5 == True:
                    f1.close()
                    f2.close()
                    fw.close()
                    shutil.copy(h5filenamewrite, h5filenamewriteBU)

                break

            #############################################################
            #is rewind 250 frames for both cameras
            if c is 44:  # , or <
                if UseSquidExcel == True:
                    if SquidIndex < 31:
                        SquidIndex = 1
                    else:
                        SquidIndex -= 30
                else:
                    if FrameSkipInc > 1:
                        print "REWINDIN'"
                        if int(numpy.floor(frame2num  / FrameSkipInc) * FrameSkipInc)!=frame2num:
                            self.capture1.set(cv2.CAP_PROP_POS_FRAMES, int(
                                numpy.floor((frame2num) / FrameSkipInc) * FrameSkipInc) - 1 + frameDelay)
                            ret, frame1 = self.capture1.read()

                            self.capture2.set(cv2.CAP_PROP_POS_FRAMES,
                                              int(numpy.floor(
                                                  (frame2num) / FrameSkipInc) * FrameSkipInc) - 1)
                            ret, frame2 = self.capture2.read()
                        else:

                            self.capture1.set(cv2.CAP_PROP_POS_FRAMES, int(
                                numpy.floor((frame2num - FrameSkipInc) / FrameSkipInc) * FrameSkipInc) -1 + frameDelay)
                            ret, frame1 = self.capture1.read()

                            self.capture2.set(cv2.CAP_PROP_POS_FRAMES,
                                              int(numpy.floor(
                                                  (frame2num - FrameSkipInc) / FrameSkipInc) * FrameSkipInc)-1 )
                            ret, frame2 = self.capture2.read()

                    else:
                        print "REWINDIN'"
                        self.rewind(250, self.capture1)
                        ret, frame1 = self.capture1.read()

                        self.rewind(250, self.capture2)
                        ret, frame2 = self.capture2.read()

                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                        cv2.CAP_PROP_POS_FRAMES)

            #############################################################
            #is fast forward 250 frames for both cameras
            if c is 46:  # is . or >
                if UseSquidExcel == True:
                    SquidIndex += 30
                else:
                    if FrameSkipInc>1:

                        print "FASTFORWARD"
                        self.capture1.set(cv2.CAP_PROP_POS_FRAMES, int(
                            numpy.floor((frame2num + FrameSkipInc) / FrameSkipInc) * FrameSkipInc) -1 + frameDelay)
                        # ret, frame1 = self.capture1.read()
                        ret, frame1 = self.capture1.read()
                        self.capture2.set(cv2.CAP_PROP_POS_FRAMES,
                                          int(numpy.floor(
                                              (frame2num + FrameSkipInc) / FrameSkipInc) * FrameSkipInc)-1 )
                        ret, frame2 = self.capture2.read()
                    else:
                        print "FASTFORWARD"
                        self.fastforward(250, self.capture1)
                        ret, frame1 = self.capture1.read()

                        self.fastforward(250, self.capture2)
                        ret, frame2 = self.capture2.read()

                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                        cv2.CAP_PROP_POS_FRAMES)

            #############################################################
            #goes backward one frame on the video for both cameras.
            if c is 108:  # is , or is l
                if UseSquidExcel == True:
                    setFrame=1
                    if SquidIndex == 1:
                        SquidIndex = 1
                    else:
                        SquidIndex -= 1
                else:

                    print "REWINDIN'"
                    self.rewind(1, self.capture1)
                    ret, frame1 = self.capture1.read()

                    self.rewind(1, self.capture2)
                    ret, frame2 = self.capture2.read()

                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                        cv2.CAP_PROP_POS_FRAMES)

            #############################################################
            #goes forward one frame on the video for both cameras.
            if c is 59:  # is ;
                if UseSquidExcel == True:
                    SquidIndex += 1
                    setFrame=1

                else:
                    print "FASTFORWARD"
                    #self.fastforward(1, self.capture1)
                    ret, frame1 = self.capture1.read()

                    #self.fastforward(1, self.capture2)
                    ret, frame2 = self.capture2.read()

                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                        cv2.CAP_PROP_POS_FRAMES)


            #############################################################
            #de-increments the current insect number, the insect number is the number used for tracking a particular animal.
            if c is 91:  # 91 is [
                if insectnumit == 0:
                    None
                else:
                    insectnumit -= 1
                    try:
                        fwData[cameraNames[0]].create_group(Tobject + str(int(insectnumit)))
                    except:
                        None
                    if UseSquidExcel == False:
                        try:
                            fwData[cameraNames[1]].create_group(Tobject + str(int(insectnumit)))
                        except:
                            None

            #############################################################
            #increments the current insect number.
            if c is 93:  # 93 is ]
                insectnumit += 1


                try:
                    fwData[cameraNames[0]].create_group(Tobject + str(int(insectnumit)))
                except:
                    None
                if UseSquidExcel == False:
                    try:
                        fwData[cameraNames[1]].create_group(Tobject + str(int(insectnumit)))
                    except:
                        None

                if insectMaxNum <= insectnumit:
                    insectMaxNum = insectnumit + 1


            #############################################################
            if Track3DobjectInDualGP==True:
                if c is 50:  # 50 is 2
                    print "2 is pressed"
                    if threeDobjNum == 0:
                        Tobject="insect"
                    else:
                        threeDobjNum -= 1
                        if threeDobjNum == 0:
                            Tobject="insect"
                        else:
                            Tobject="Squid"+str(int(threeDobjNum))+"_"
                        try:
                            fwData[cameraNames[0]].create_group(Tobject + str(int(insectnumit)))
                        except:
                            None
                        if UseSquidExcel == False:
                            try:
                                fwData[cameraNames[1]].create_group(Tobject + str(int(insectnumit)))
                            except:
                                None


                #############################################################
                if c is 51:  # 51 is 3
                    print "3 is pressed"
                    threeDobjNum += 1
                    Tobject = "Squid" + str(int(threeDobjNum)) + "_"
                    try:
                        fwData[cameraNames[0]].create_group(Tobject + str(int(insectnumit)))
                    except:
                        None
                    if UseSquidExcel == False:
                        try:
                            fwData[cameraNames[1]].create_group(Tobject + str(int(insectnumit)))
                        except:
                            None

                    if threeDobjNumMax <= threeDobjNum:
                        threeDobjNumMax = threeDobjNum + 1

            #############################################################
            #Zooms into the image.
            if c is 110:  # is n
                magZoom += .25
                print magZoom

            #############################################################
            #Zooms out of the image.
            if c is 109:  # m
                magZoom -= .25
                print magZoom
                if magZoom < .2:
                    magZoom = 0


            #############################################################
            #selects the center point of the frame as a tracked point for the object.
            if c is 39:  # 38 is '
                ErrorOrPoint = 1
                if moviewatch == 0 or moviewatch == 2:
                    if UseSquidExcel == True:
                        frame1num = SquidIndex
                    else:
                        frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                    if ErrorOrPoint == 1:
                        try:
                            g2 = fwData[cameraNames[0]][Tobject + str(int(insectnumit))].create_dataset(
                                str(int(frame1num)), data=[self.point1[0], self.point1[1], 4, 4])

                        except:
                            fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][0] = \
                                self.point1[0]
                            fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][1] = \
                                self.point1[1]
                    elif ErrorOrPoint == 0:

                        try:
                            fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][2] = \
                                4
                            fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(frame1num))][3] = \
                                4
                        except:
                            print "None found"

                if moviewatch == 1 or moviewatch == 2:
                    frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                    if ErrorOrPoint == 1:
                        try:
                            g2 = fwData[cameraNames[1]][
                                Tobject + str(int(insectnumit))].create_dataset(str(int(frame2num)),
                                                                                data=[self.point2[0],
                                                                                      self.point2[1], 4,
                                                                                      4])

                        except:
                            fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                str(int(frame2num))][0] = self.point2[0]
                            fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                str(int(frame2num))][1] = \
                                self.point2[1]
                    elif ErrorOrPoint == 0:

                        try:
                            fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                str(int(frame2num))][2] = \
                                4
                            fwData[cameraNames[1]][Tobject + str(int(insectnumit))][
                                str(int(frame2num))][3] = \
                                4
                        except:
                            print "None found"
                if UseSquidExcel == True:
                    if insectnumit>0 and insectnumit<43:
                        insectnumit+=1
            #######################################################################


            #############################################################
            #goes from tracking mode to error mode.  The error mode is not currently being used and may be deleted.
            if c is 47:  ## c is /
                UseNumbersForError = not UseNumbersForError


            #######################################################################
            #######################################################################
            #######################################################################
            #######################################################################
            #######################################################################
            #######################################################################
            #######################################################################
            #######################################################################


            if UseSquidExcel == False:

                #############################################################
                #plots the 3D path of the current insect number.
                if c is 100:  # 100 is d  makes a plot
                    #cv2.destroyAllWindows()
                    self.Plot3Dinsect(fwData, RotMat, TranMat, insectnumit)

                #############################################################
                #pauses or lets the video run.
                if c is 32:  # is SPACE BAR
                    if self.isPaused == True:
                        self.isPaused = False
                        #self.waitTime1 = (1000.0 / self.fps1) / 2
                    else:
                        self.isPaused = True
                        #self.waitTime1 = (1000.0 / self.fps1) / 2  # 1000.0

                #############################################################
                #toggles between object tracking mode and fiducial object tracking mode.
                if c is 8:  # 8 is backspace
                    if Tobject == "insect":
                        Tobject = "fiducial0_"
                        try:
                            fwData[cameraNames[moviewatch]].create_group(Tobject + str(int(insectnumit)))
                        except:
                            None
                    elif Tobject == "fiducial0_":
                        Tobject = "fiducialM_"
                        try:
                            fwData[cameraNames[moviewatch]].create_group(Tobject + str(int(insectnumit)))
                        except:
                            None
                    elif Tobject == "fiducialM_":
                        Tobject = "insect"

                #############################################################
                #Set the program to rewind mode where the video goes backwards during normal tracking procedures. This
                # is much slower than the non-rewind mode because opencv does not have a real rewind feature.
                if c is 107:  # 107 is k    rewind
                    if UseSquidExcel==False:
                        movieRewind ^= True

                if c is 49:  # 49 is 1
                    viewTrackingThings = not viewTrackingThings
                    print viewTrackingThings


                if c is 102:  # 102 is f
                    KalmanOverlay = not KalmanOverlay
                    savefunction=2
                    print KalmanOverlay
                #############################################################
                if c is 45:  # 65 is -
                    print "-"
                    if 1:

                        #self.PnPtoFacePointsForFOinGoProsRecursion(Box, fwData, cameraNames, moviewatch,
                        self.PnPtoFacePointsForFOinGoPros(Box, fwData, cameraNames, moviewatch,
                                                     frame1num, cameraMatrix1,
                                                     distCoeffs1, f1, f2,frameDelay)
                    if 0:
                        self.EssentialMatrixForFOinGoPros(Box, fwData, cameraNames, moviewatch,
                                                          frame1num, cameraMatrix1,
                                                          distCoeffs1, f1, f2, frameDelay)

                if c is 48:# 48 is 0
                    print "0 was pressed"

                    if Water_Surface_identified==True:
                        print "Water_Surface_identified"
                        self.FitWaterSurfacetofromPointCloud(fwData, cameraNames,Water_Surface_Point_Cloud_Path)

                    else:
                        print "Water surface not Identified"

                #############################################################
                if c is 61:  # 61 is =
                    print "="
                    if PointCloudIdentified==True:
                        print "PointCloudIdentified"
                        answer = raw_input("If you want an STL from just the point cloud enter 0 \n If you want an STL "
                                           "from the point cloud and the water plane enter 1 \n If you want an STL of "
                                           "the water plane without the point cloud enter 2 \n ")
                        isnotandint = True

                        while isnotandint:
                            try:
                                answer = int(answer)
                                if answer < 0 or answer > 2:
                                    answer = raw_input("please enter an integer within the range of 0 to 2 and press enter: ")
                                else:
                                    isnotandint = False
                            except:
                                answer = raw_input("please enter an integer within the range of 0 to 2 and press enter: ")

                        self.makeSTLfromPointCloud(fwData, cameraNames,h5WritePath,PointCLoudPath,PointCLoudScale,Water_Surface_Plane,answer)

                    else:
                        print "PointCloud not Identified"
                        #self.PnPtoFacePointsForFOinGoProsRecursion(Box, fwData, cameraNames, moviewatch,
                        #                                       frame1num, cameraMatrix1,
                        #                                       distCoeffs1, f1, f2, frameDelay)

                #####################################################################
                if c is 52:  # 52 is 4
                    print "4 was pressed"
                    RotMatSQ, TranMatSQ=self.SquidFitToPath(fwData, cameraNames,threeDobjNum,frame1num)
                    #MeshProjPre = self.SquidFitToPath(fwData, cameraNames, threeDobjNum, frame1num)

                    if 1:# this is how we print out the data for Blender
                        eq=0
                        frameKeys=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)].keys()
                        print frameKeys
                        for FK in frameKeys:
                            for eq in range(23):
                                try:
                                    FDWF=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                        FK]['FitDatawithFiducial']


                                    print "[",FDWF[0],",",FDWF[1],",",FDWF[2],",",FDWF[3],",",FDWF[4],",",FDWF[5],",",FDWF[6],"]"+","
                                except:
                                    None
                            print ""

                    if 0:# this is how we print out the data for Blender
                        eq=0
                        frameKeys=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)].keys()
                        pointssquidrealWidened = numpy.array(
                            [[0, 3.55998, -0.02321], [0.76313, -0.89247, 0.22152], [-0.76313, -0.89247, 0.22152]])
                        print frameKeys
                        for FK in frameKeys:
                            for eq in range(23):
                                try:


                                    TotRotSQ=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                        FK]['TotRotSQ']
                                    TotTrans=fwData[cameraNames[2]]["Squid" + str(eq + 1) + "_" + "Squid" + str(eq + 1)][
                                        FK]['TotTrans']
                                    print FK,eq
                                    print "eye origin",numpy.matmul(TotRotSQ,pointssquidrealWidened[1])+TotTrans
                                    xshat=[1,0,0]
                                    yshat=[0,1,0]
                                    zshat=[0,0,1]
                                    print "zshat", (numpy.matmul(TotRotSQ,zshat))/numpy.linalg.norm(numpy.matmul(TotRotSQ,zshat))
                                    print "yshat", (numpy.matmul(TotRotSQ,yshat))/numpy.linalg.norm(numpy.matmul(TotRotSQ,yshat))
                                    print "xshat", (numpy.matmul(TotRotSQ,xshat))/numpy.linalg.norm(numpy.matmul(TotRotSQ,xshat))

                                    print "[",FDWF[0],",",FDWF[1],",",FDWF[2],",",FDWF[3],",",FDWF[4],",",FDWF[5],",",FDWF[6],"]"+","
                                except:
                                    None
                            print ""



                    if 0:
                        RotMatSQ=numpy.array([[1,0,0],[0,1,0],[0,0,1]])
                        TranMatSQ=numpy.array([0,0,0])
                        pcMat=[]
                        print MeshProjPre
                        for ie in range(len(MeshProjPre)):
                            print MeshProjPre[ie][0],numpy.isnan(MeshProjPre[ie][0])
                            if numpy.isnan(MeshProjPre[ie][0])==False:
                                print MeshProjPre[ie]
                                pcMat.append(MeshProjPre[ie])

                        pcMat=numpy.array(pcMat)
                        pcMat=numpy.transpose(pcMat)
                        print pcMat, pcMat.shape
                    ViewDualGoProSquid=True


                #############################################################
                #switches between the two coincident cameras.
                if c is 118:  # 118 is v #chagning video
                    if moviewatch == 0:
                        moviewatch = 1
                        print self.capture2.get(cv2.CAP_PROP_POS_FRAMES), "capture2"
                        cv2.destroyAllWindows()
                    else:
                        moviewatch = 0
                        print self.capture1.get(cv2.CAP_PROP_POS_FRAMES), "capture1"
                        cv2.destroyAllWindows()


                #############################################################
                #sets the program to an ergonomic tracking mode where the video frame gets smaller right around the
                # focal point and the object is selected from the mouse position when the mouse moves and then stops
                # for a few seconds.  The object is selected and incremented to the next frame with the projected
                # position of the object calculated from the previous two frames.  As is does so it plays through
                # several frames before the next frame so the user can see the movement in the video, making the object
                # easier to see.
                if c is 122:  # 122 is z
                    if ClickIncrimentMode == False:
                        ClickIncrimentMode = True
                        MouseHasNotMovedIn=70
                        WindowWidthPrev=WindowWidth
                        WindowWidth=200

                        if FrameSkipInc>1:

                            if int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc)!=frame2num:
                                self.capture1.set(cv2.CAP_PROP_POS_FRAMES,  int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc)-1+ frameDelay)
                                ret, frame1 = self.capture1.read()

                                self.capture2.set(cv2.CAP_PROP_POS_FRAMES,  int(numpy.floor(frame2num/FrameSkipInc)*FrameSkipInc) - 1)
                                ret, frame2 = self.capture2.read()

                    else:
                        ClickIncrimentMode = False
                        WindowWidth=WindowWidthPrev



                #############################################################
                #goes forward one frame only in camera1 leaving camera2 in the same place.
                if c is 111:  # 111 is o
                    print "REWINDIN'"
                    self.rewind(1, self.capture1)
                    ret, frame1 = self.capture1.read()

                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(cv2.CAP_PROP_POS_FRAMES)

                #############################################################
                #goes backward one frame only in camera1 leaving camera2 in the same place.
                if c is 112:  # 112 is p
                    ret, frame1 = self.capture1.read()
                    print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(cv2.CAP_PROP_POS_FRAMES)



                #############################################################
                #This is a function used to save various images from video files along with other information
                if c is 115:  # 115 is s    or write text file  or excel and images

                    ##savefunction has different functionalities.
                    #savefunction = 2


                    #################################
                    #this savefunction
                    # This is a functionalitiy just saves the current image.  It can be used with KalmanOverlay
                    # that overlays the paths onto the image.   This requires at this point specifying the
                    # paths to be ploted.   There are arrows that give the path direction.
                    if savefunction == 0:
                        frameSaveForLater = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                        # HowManyFrames=1277
                        HowManyFrames = 1  # just for one
                        frame2use = frameSaveForLater
                        #place2SaveImages = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/amberenvi"
                        if moviewatch == 0 or moviewatch == 2:
                            pngFileName = place2SaveImages + "/" + ActualFileName1[0] + "_" + str(
                                int(frame2use)) + ".png"
                            cv2.imwrite(
                                pngFileName,
                                bitmap1)

                        if moviewatch == 1 or moviewatch == 2:
                            pngFileName = place2SaveImages + "/" + ActualFileName2[0] + "_" + str(
                                int(frame2use)) + ".png"
                            cv2.imwrite(
                                pngFileName,
                                bitmap2)



                    #################################
                    # This savefunction turns on a functionality in the program loop.     This will save the rendered
                    # images of the analysis program with all the visible anotations to the folder specified in SingleFramesImages.

                    elif savefunction == 2:

                        if saveImagesAsIs==True:
                            saveImagesAsIs=False
                        else:
                            saveImagesAsIs = True
                    elif savefunction == 3:
                        self.MakeVideoFromGoPro6Video(capture1)
                    #################################
                    #this savefunction is in association with object STL annotations.   It saves the image and writes informaation down in
                    #an excell file.
                    elif savefunction == 1:
                        frameSaveForLater = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                        #HowManyFrames=1277
                        HowManyFrames = 1 #just for one
                        frame2use=frameSaveForLater
                        print frame2use,"frame2use"



                        for hh in range(HowManyFrames):

                            cap = self.capture1
                            s.write(xlrownum, 1,
                                    str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], 1).value))

                            #here we want to write stuff the name.
                            rbb = xlrd.open_workbook(Excelfilename1)
                            filenamerepeat = rbb.sheet_by_index(0).col_values(4).count(
                                ActualFileName1[0] + "_" + str(int(frame2use)) + ".png")

                            print filenamerepeat,ActualFileName1[0] + "_" + str(int(frame2use)) + ".png"
                            if filenamerepeat==0:
                                s.write(xlrownum, 0,
                                        str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],
                                                                                     0).value) + "_" + str(int(frame2use)))
                            else:
                                s.write(xlrownum, 0,
                                        str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName],
                                                                                     0).value) + "_" + str(int(frame2use))+"_"+str(filenamerepeat))

                            s.write(xlrownum, 2,
                                    frame1num)
                            pngFileName = AustraliaSingleFrames + "/" + ActualFileName1[0] + "_" + str(int(frame2use)) + ".png"
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame2use - 1)
                            s.write(xlrownum, 3,
                                    pngFileName)
                            s.write(xlrownum, 4,
                                    ActualFileName1[0] + "_" + str(int(frame2use)) + ".png")
                            s.write(xlrownum, 5,
                                    AustraliaSingleFrames)

                            if projectWorkbook == "SingleVideos" and PanelNum == 4:
                                as_array = numpy.asarray(frame1[:, :])
                                frameToRead1 = as_array[50:150, 40:950]  # for 1080 images
                                frameToRead2 = as_array[150:250, 40:350]  # for 1080 images
                                textPT = pytesseract.image_to_string(frameToRead1)
                                textPT2 = pytesseract.image_to_string(frameToRead2)
                                print textPT, "textpt", textPT2
                                if textPT2.split(":")[0] == "Frame":
                                    textPT2Num = int(textPT2.split(":")[1]) - 1
                                else:
                                    textPT2Num = "No good"

                                print textPT2Num
                                s.write(xlrownum, 6,
                                        textPT)
                                s.write(xlrownum, 7,
                                        textPT2Num)


                            ret, frameOut = cap.read()
                            print frame2use, cap.get(cv2.CAP_PROP_POS_FRAMES)

                            if not os.path.isdir(AustraliaSingleFrames):
                                os.mkdir(AustraliaSingleFrames)
                            cv2.imwrite(
                                pngFileName,
                                frameOut)
                            print AustraliaSingleFrames
                            xlrownum += 1

                            print "saved ", xlrownum
                            wb.save(Excelfilename1)
                            frame2use+=1

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frameSaveForLater)
                        print "frame2use current", cap.get(cv2.CAP_PROP_POS_FRAMES)

                    elif savefunction == 4:
                        frameSaveForLater = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                        # HowManyFrames=1277
                        HowManyFrames = 1  # just for one
                        frame2use = frameSaveForLater
                        if moviewatch == 0 or moviewatch == 2:
                            pngFileName = place2SaveImages + "/" + ActualFileName1[0] + "_" + str(
                               int(frame2use)) + ".png"
                            cv2.imwrite(
                               pngFileName,
                               frame1)

                #############################################################
                #this selects for the forward subtraction functionality where the image before is image registered for the
                #current image and subtracted from the current frame.   This is not always helpful.   The registration does not
                #work too well.
                if c is 98:  # 98 is b
                    if UseforwardFrameSubtraction==0:
                        UseforwardFrameSubtraction = 1
                    else:
                        UseforwardFrameSubtraction=0

                #############################################################
                #Each insect must be paired with its counterpart in the coincident frame and labled the same.  This will then
                #calulate the 3D positions of the paths from both insect tracks.
                if c is 43:  # 43 is +
                    if threeDobjNum==0:
                        self.MergeAndComputeInsectTracks(fwData, frameDelay, f1, cameraMatrix1, distCoeffs1, f2, cameraMatrix2,
                                                         distCoeffs2, Tobject,Box,Water_Surface_Plane,h5SquidDelay,delayfraction)
                    elif "Squid" in Tobject:
                        print "Squid here"
                        TobjectKeep=Tobject
                        Tobject="insect"
                        self.MergeAndComputeInsectTracks(fwData, frameDelay, f1, cameraMatrix1, distCoeffs1, f2, cameraMatrix2,
                                                         distCoeffs2, Tobject,Box,Water_Surface_Plane,h5SquidDelay,delayfraction)
                        for po in range(1,threeDobjNumMax+1):
                            Tobject = "Squid" + str(int(po)) + "_"
                            self.MergeAndComputeInsectTracks(fwData, frameDelay, f1, cameraMatrix1, distCoeffs1, f2,
                                                             cameraMatrix2,
                                                             distCoeffs2, Tobject, Box, Water_Surface_Plane,h5SquidDelay,delayfraction)
                        Tobject=TobjectKeep

                #############################################################
                #The reflections of animals can be tracked. If you have both the insect and the reflection of the
                # insect in the frame but not the coincident frame. This uses the calculated plane of the water, taken
                # from the orb_map and fitted to a plane. The plane needs to be manually inputted into the program.
                #
                if c is 119 : # 119  is w
                    None
                    #MapErrorDicMapPoint[fd] = numpy.matmul(ResultRot.T, MapPointfmd) + ResultTran
                    #MapErrorDicMeauredPoint[fd].append(cloPoAveTran)
                    self.analyizeMapErrorDics(CameraOriginArray,MapErrorDicMeauredPoint,MapErrorDicMapPoint)

                ########################################################
                if 1:

                    if c is 113:  # 113 is q
                        now = datetime.now()
                        current_time = now.strftime("%Y%d%m_%H%M%S")
                        #print current_time,"current_time"
                        #Initialations of the Plot
                        workbook2 = xlrd.open_workbook('ProjectMain.xlsx')
                        worksheetPlot = workbook2.sheet_by_name("PlottingSheet")
                        worksheetPlotHeader= dict(zip(worksheetPlot.row_values(0), range(len(worksheetPlot.row_values(0)))))
                        #print worksheetPlotHeader
                        AsterPosition=worksheetPlotHeader["*"]
                        if (workbook2.sheet_by_name("PlottingSheet").cell(
                            0, AsterPosition-1).value)==projectName:
                            NoPlottingError=True
                            PlotNamePosition=AsterPosition-1


                            PlotNamePositionHeader=dict(zip(worksheetPlot.col_values(PlotNamePosition), range(len(worksheetPlot.col_values(PlotNamePosition)))))
                            #print PlotNamePositionHeader

                            ###############
                            #tracking the lenth of the headers:
                            for i in range(PlotNamePosition,PlotNamePosition+50):
                                #print "i",i
                                try:
                                    if workbook2.sheet_by_name("PlottingSheet").cell(PlotNamePositionHeader["ToUse"], i).value=="":
                                        PlotNamePositionStop=i-1
                                        #print PlotNamePositionStop,"PlotNamePositionStop"
                                        break
                                except:
                                    PlotNamePositionStop = i - 1
                                    break

                            #PlotNameInsectHeader=dict(zip(worksheetPlot.row_values(PlotNamePositionHeader["ToUse"]), range(len(worksheetPlot.row_values(PlotNamePositionHeader["ToUse"])))))
                            PlotNameInsectHeader = dict(zip(worksheetPlot.row_values(PlotNamePositionHeader["ToUse"])[PlotNamePosition:PlotNamePositionStop+1],
                                                            range(PlotNamePosition,PlotNamePositionStop+1)))

                            #print "PlotNameInsectHeader",PlotNameInsectHeader


                            PointofReference=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Point of Reference"], PlotNamePosition + 1).value)
                            # getting lists of insect numbers to use
                            insectNumGroupNumber=int(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Number of simultaneous insects"],PlotNamePosition+1).value)
                            PathColorGradientString=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Path color gradient"],PlotNamePosition+1).value)
                            ConnectLinesColorGradientString=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Coincident lines color gradient"],PlotNamePosition+1).value)


                            OutputCVSType=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Output CVS"],PlotNamePosition+1).value)
                            OutputCVSName=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Output CVS name"],PlotNamePosition+1).value)
                            OutputCVSMeshLab=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Output CVS for Meshlab"],PlotNamePosition+1).value)

                            plot3D=False
                            plotKin=False
                            if (workbook2.sheet_by_name("PlottingSheet").cell(PlotNamePositionHeader["Type of plot"], PlotNamePosition + 2).value)=="*":
                                plot3D=True
                            if (workbook2.sheet_by_name("PlottingSheet").cell(PlotNamePositionHeader["Type of plot"], PlotNamePosition + 4).value)=="*":
                                plotKin=True

                            PlotDesciptiveDic={}

                            PlotDesciptiveDic["OutputCVSMeshLab"]=OutputCVSMeshLab
                            PlotDesciptiveDic["insectNumGroupNumber"]=insectNumGroupNumber
                            PlotDesciptiveDic["PathColorGradientString"] = PathColorGradientString
                            PlotDesciptiveDic["ConnectLinesColorGradientString"]=ConnectLinesColorGradientString
                            PlotDesciptiveDic["UseLines"]  = (workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Mark Coincident Points With Lines"], PlotNamePosition + 1).value)
                            PlotDesciptiveDic["Water_Surface_Plane"]=Water_Surface_Plane
                            PlotDesciptiveDic["OutputCVSType"]=OutputCVSType
                            PlotDesciptiveDic["OutputCVSName"] = OutputCVSName
                            PlotDesciptiveDic["PathOfSTLtoDisplay"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Path of STL to display"],PlotNamePosition+1).value)
                            PlotDesciptiveDic["DisplaySTLin3Dplot"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Display STL in 3D plot"],PlotNamePosition+1).value)


                            PlotDesciptiveDic["ColorGradientDistinguishingMultiplePaths"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Color gradient distinguishing multiple paths"],PlotNamePosition+1).value)


                            PlotDesciptiveDic["Path2SaveImages"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Output path"],PlotNamePosition+1).value)
                            PlotDesciptiveDic["SaveImages"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Save images"],PlotNamePosition+1).value)
                            PlotDesciptiveDic["SaveFileName"]=(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["Save file name"],PlotNamePosition+1).value)

                            PlotDesciptiveDic["current_time"]=current_time

                            if len(PointofReference)>1:
                                PlotDesciptiveDic["PointofReference"] =[float(PointofReference.split(",")[0]),
                                                                        float(PointofReference.split(",")[1]),
                                                                        float(PointofReference.split(",")[2])]

                            else:
                                PlotDesciptiveDic["PointofReference"] = [0.0,0.0,0.0]

                            PlotDesciptiveDic["3DPlotViewpoint"]=str(workbook2.sheet_by_name("PlottingSheet").cell(
                                PlotNamePositionHeader["3D plot viewpoint"],PlotNamePosition+1).value)

                            for i in range(insectNumGroupNumber):
                                PlotDesciptiveDic["Insectnumbers"+str(i+1)]=[]
                                PlotDesciptiveDic["InsectLabels" + str(i + 1)] = []
                                PlotDesciptiveDic["InsectColors" + str(i + 1)] = []
                                PlotDesciptiveDic["InsectMultColors" + str(i + 1)] = []
                                PlotDesciptiveDic["InsectSizes" + str(i + 1)] = []

                            PlotDesciptiveDic["OtherProjects"] = []
                            PlotDesciptiveDic["StartFrame"] = []
                            PlotDesciptiveDic["EndFrame"] = []
                            for ik in range(int(PlotNamePositionHeader["ToUse"]) + 1,len(worksheetPlot.col_values(PlotNamePosition))):

                                if workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNamePosition).value=="*":
                                    for i in range(insectNumGroupNumber):

                                        insectNumfromExcel=int(workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNameInsectHeader["object numbers "+str(i+1)]).value)
                                        if str(insectNumfromExcel) in PlotDesciptiveDic["Insectnumbers"+str(i+1)]:
                                            for hi in range(100):
                                                if str(insectNumfromExcel)+"-"+str(hi) not in PlotDesciptiveDic["Insectnumbers"+str(i+1)]:
                                                    PlotDesciptiveDic["Insectnumbers" + str(i + 1)].append(
                                                        str(insectNumfromExcel)+"-"+str(hi))
                                                    break
                                        else:
                                            PlotDesciptiveDic["Insectnumbers"+str(i+1)].append(str(insectNumfromExcel))
                                        #PlotDesciptiveDic["InsectPlaceHolders" + str(i + 1)].append(ik)
                                        PlotDesciptiveDic["InsectLabels" + str(i + 1)].append((workbook2.sheet_by_name(
                                            "PlottingSheet").cell(ik,
                                                                  PlotNameInsectHeader["object label " + str(i + 1)]).value))

                                        PlotDesciptiveDic["InsectColors" + str(i + 1)].append((workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNameInsectHeader["colors "+str(i+1)]).value))
                                        PlotDesciptiveDic["InsectMultColors" + str(i + 1)].append("")
                                        PlotDesciptiveDic["InsectSizes" + str(i + 1)].append((workbook2.sheet_by_name(
                                            "PlottingSheet").cell(ik,
                                                                  PlotNameInsectHeader["Size " + str(i + 1)]).value))
                                    PlotDesciptiveDic["OtherProjects"].append((workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNameInsectHeader["From another project"]).value))
                                    PlotDesciptiveDic["StartFrame"].append((workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNameInsectHeader["Start frame"]).value))
                                    PlotDesciptiveDic["EndFrame"].append((workbook2.sheet_by_name("PlottingSheet").cell(ik,PlotNameInsectHeader["End frame"]).value))


##########################################################################################
                            ##########################################################################################
                            ##########################################################################################
                            #color stuff
                            colorLetterDic={}
                            colorLetterDic["y"]=(1,1,0)
                            colorLetterDic["m"]=(1,0,1)
                            colorLetterDic["c"]=(0,1,1)
                            colorLetterDic["r"]=(1,0,0)
                            colorLetterDic["g"]=(0,1,0)
                            colorLetterDic["b"]=(0,0,1)
                            colorLetterDic["w"]=(1,1,1)
                            colorLetterDic["k"]=(0,0,0)



                            if PlotDesciptiveDic["ColorGradientDistinguishingMultiplePaths"]!="":
                                Multcolorarray = getattr(plt.cm, PlotDesciptiveDic["ColorGradientDistinguishingMultiplePaths"])\
                                    (numpy.linspace(0, 1, len(PlotDesciptiveDic["Insectnumbers"+str(i+1)])))


                            #getting the color formated correctly
                            for i in range(insectNumGroupNumber):
                                for ii in range(len(PlotDesciptiveDic["Insectnumbers"+str(i+1)])):
                                    ccc=PlotDesciptiveDic["InsectColors" + str(i + 1)][ii]
                                    if ccc=="":
                                        PlotDesciptiveDic["InsectColors" + str(i + 1)][ii]=(0,0,0)
                                    elif len(ccc)==1:
                                        PlotDesciptiveDic["InsectColors" + str(i + 1)][ii] =colorLetterDic[ccc]
                                    elif len(ccc.split(","))==3:
                                        PlotDesciptiveDic["InsectColors" + str(i + 1)][ii] = (float(ccc.split(",")[0]),float(ccc.split(",")[1]),float(ccc.split(",")[2]))
                                    elif len(ccc.split(";")) == 2:
                                        PlotDesciptiveDic["InsectColors" + str(i + 1)][ii] = getattr(plt.cm,ccc.split(";")[0])(float(ccc.split(";")[1]))
                                        #colorarray = plt.cm.jet(numpy.linspace(0, 1, len(Insectnumbers1)))
                                    if PlotDesciptiveDic["ColorGradientDistinguishingMultiplePaths"] != "":
                                        PlotDesciptiveDic["InsectMultColors" + str(i + 1)][ii]=Multcolorarray[ii]



                            #getting the other fwprojects

                            PlotDesciptiveDic["OtherProjectsPaths"]=[]
                            for i in range(len(PlotDesciptiveDic["OtherProjects"])):
                                #print i, PlotDesciptiveDic["OtherProjects"][i]
                                if PlotDesciptiveDic["OtherProjects"][i]=="":
                                    PlotDesciptiveDic["OtherProjectsPaths"].append("")
                                else:
                                    PlotDesciptiveDic["OtherProjectsPaths"].append(h5WritePath+"/"+str(workbook.sheet_by_name(projectWorkbook).cell(projName[PlotDesciptiveDic["OtherProjects"][i]],
                                                                                                  projHeader[
                                                                                                      "output_file"]).value))

                            #if plotKin == True:#setting up the reference vector
                            if 1:
                                KinPlotDesciptiveDic = {}
                                KinPlotSheetName = (workbook2.sheet_by_name("PlottingSheet").cell(
                                    PlotNamePositionHeader["Kinematics Graph Reference"], PlotNamePosition + 1).value)

                                worksheetKinPlot = workbook2.sheet_by_name(KinPlotSheetName)

                                KinPlotNamePositionHeader = dict(zip(worksheetKinPlot.col_values(0),
                                                                  range(len(worksheetKinPlot.col_values(0)))))

                                PlotDesciptiveDic["Reference vector"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Reference vector"], 1).value)
                                PlotDesciptiveDic["Smoothing Window"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Smoothing Window"], 1).value)
                                PlotDesciptiveDic["Smoothing Iteration"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Smoothing Iteration"], 1).value)

                                PlotDesciptiveDic["Use spline instead of average"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Use spline instead of average"], 1).value)
                                PlotDesciptiveDic["Spline order"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Spline order"], 1).value)
                                PlotDesciptiveDic["Spline smoothing factor"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Spline smoothing factor"], 1).value)


                                KinPlotDesciptiveDic["Panel number"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Panel number"], 1).value)
                                KinPlotDesciptiveDic["Panel Size x"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Panel Size x"], 1).value)
                                KinPlotDesciptiveDic["Panel Size y"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Panel Size y"], 1).value)
                                KinPlotDesciptiveDic["From relative start"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["From relative start"], 1).value)
                                KinPlotDesciptiveDic["Sync X axis"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Sync X axis"], 1).value)
                                KinPlotDesciptiveDic["Polar Plot"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Polar Plot"], 1).value)
                                KinPlotDesciptiveDic["Box Plot"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Box Plot"], 1).value)
                                KinPlotDesciptiveDic["Reference vector"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["Reference vector"], 1).value)
                                KinPlotDesciptiveDic["Smoothing Window"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["Smoothing Window"], 1).value)
                                KinPlotDesciptiveDic["Smoothing Iteration"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["Smoothing Iteration"], 1).value)
                                KinPlotDesciptiveDic["Show Error Bars"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["Show Error Bars"], 1).value)
                                KinPlotDesciptiveDic["Color gradient across plots"] = (
                                    workbook2.sheet_by_name(KinPlotSheetName).cell(
                                        KinPlotNamePositionHeader["Color gradient across plots"], 1).value)

                                KinPlotNameInsectHeader = dict(
                                    zip(worksheetKinPlot.row_values(KinPlotNamePositionHeader["Panel Header"]),
                                        range(len(worksheetKinPlot.row_values(
                                            KinPlotNamePositionHeader["Panel Header"])))))
                                #print KinPlotNameInsectHeader
                                j = 1
                                KinPlotDesciptiveDic["Parameters to graph (comma separated)" + str(j)] = []
                                KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(j)] = []
                                KinPlotDesciptiveDic["Y-axis label" + str(j)] = []
                                KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(j)] = []
                                KinPlotDesciptiveDic["X-axis range (comma separated)" + str(j)] = []
                                KinPlotDesciptiveDic["X-axis label" + str(j)] = []
                                KinPlotDesciptiveDic["Graph Title" + str(j)] = []
                                KinPlotDesciptiveDic["Path Labels"+ str(j)]=[]
                                KinPlotDesciptiveDic["Graph Right Offset"+ str(j)]=[]

                                for nop in range(int(KinPlotDesciptiveDic["Panel number"])):
                                    KinPlotDesciptiveDic["Parameters to graph (comma separated)" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Parameters to graph (comma separated)"]).value)

                                    KinPlotDesciptiveDic["Y-axis range (comma separated)" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Y-axis range (comma separated)"]).value)

                                    KinPlotDesciptiveDic["Y-axis label" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Y-axis label"]).value)

                                    KinPlotDesciptiveDic["X-Parameter to graph (single)" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["X-Parameter to graph (single)"]).value)

                                    KinPlotDesciptiveDic["X-axis label" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["X-axis label"]).value)

                                    KinPlotDesciptiveDic["X-axis range (comma separated)" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["X-axis range (comma separated)"]).value)

                                    KinPlotDesciptiveDic["Graph Title" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Graph Title"]).value)

                                    KinPlotDesciptiveDic["Path Labels" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Path Labels"]).value)

                                    KinPlotDesciptiveDic["Graph Right Offset" + str(j)].append(
                                        workbook2.sheet_by_name(KinPlotSheetName).cell(
                                            KinPlotNamePositionHeader["Panel " + str(nop + 1)],
                                            KinPlotNameInsectHeader["Graph Right Offset"]).value)

                            else:
                                PlotDesciptiveDic["Reference vector"] ="1,0,0"
                            try:
                                ResultRot = numpy.array(
                                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])
                                ResultRot=ResultRot.T

                                ResultTran = numpy.array(
                                    fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                            except:
                                ResultRot = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])#l;l

                                ResultTran = numpy.array(fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])



                            ########################################################################
                            ######################################################################
                            #ERROR CHECKS
                            if plotKin == True:
                                if KinPlotDesciptiveDic["Polar Plot"]=="y" and KinPlotDesciptiveDic["Box Plot"] =="y":
                                    print "You cannot have a Polar Plot and a Box Plot at the same time"
                                    NoPlottingError=False

                                if KinPlotDesciptiveDic["Polar Plot"] == "y" and KinPlotDesciptiveDic["From relative start"]=="y":
                                    print "You cannot select Polar Plot and From Relative Start at the same time"
                                    NoPlottingError = False

                            if NoPlottingError==True:



                                DictA = self.createPathDic2(PlotDesciptiveDic, ResultRot, ResultTran, fwData
                                                           )



                                KinPlotDesciptiveDic["Calculate distance to STL"] = (workbook2.sheet_by_name(KinPlotSheetName).cell(
                                    KinPlotNamePositionHeader["Calculate distance to STL"], 1).value)
                                if KinPlotDesciptiveDic["Calculate distance to STL"] =="y":
                                    DictA = self.createMoreSTLPathDic2(PlotDesciptiveDic, DictA,obbTree, polydata)

                                DictA,DictErr = self.createMorePathDic2(PlotDesciptiveDic, DictA,KinPlotDesciptiveDic)



                                    #plotting
                                #TypeOfPlotprint TypeOfPlot,"TypeOfPlot"

                                #if TypeOfPlot=="3D":
                                if plot3D==True:
                                    self.Plot3Dinsect2(PlotDesciptiveDic, DictA,DictErr,thePlatformNum)
                                #elif TypeOfPlot=="kinematics":

                                book = Workbook()

                                if plotKin == True:




                                    if OutputCVSType=="full":
                                        self.outputPathsinCSVMajor(PlotDesciptiveDic, KinPlotDesciptiveDic, DictA,book)
                                    #elif OutputCVSType=="y":
                                    #    self.outputPathsinCSVGraph(PlotDesciptiveDic, KinPlotDesciptiveDic, DictA,book)


                                    self.outputGraphs2(PlotDesciptiveDic,KinPlotDesciptiveDic, DictA,DictErr,book)
                                #plt.show()
                                #mngr = plt.get_current_fig_manager()
                                #mngr.frame.SetPosition((100, 20))

                                #making excel info file




                                sheet3=book.add_sheet('GeneralInfo')
                                sheet = book.add_sheet('PlottingSheet')
                                sheet2 = book.add_sheet('KinematicsGraph')
                                #s.write(xlrownum, 1,
                                #        str(workbook.sheet_by_name(projectWorkbook).cell(projName[projectName], 1).value))

                                #for ik in range(int(PlotNamePositionHeader["ToUse"]) + 1,
                                #                len(worksheetPlot.col_values(PlotNamePosition))):
                                for collit in range(PlotNamePositionStop-PlotNamePosition):
                                    for rowit in range(len(worksheetPlot.col_values(PlotNamePosition))):
                                        sheet.write(rowit, collit,
                                                    workbook2.sheet_by_name("PlottingSheet").cell(rowit,
                                                        collit+PlotNamePosition
                                                        ).value)


                                for collit in range(len(workbook2.sheet_by_name(KinPlotSheetName).row_values(0))):
                                    for rowit in range(len(workbook2.sheet_by_name(KinPlotSheetName).col_values(0))):
                                        sheet2.write(rowit, collit,
                                                    workbook2.sheet_by_name(KinPlotSheetName).cell(rowit,
                                                        collit
                                                        ).value)



                                book.save(PlotDesciptiveDic["Path2SaveImages"]+"/"+PlotDesciptiveDic["SaveFileName"]+"_INFO.xls")


                                plt.show()

                        else:
                            print "Plotting project not congruent with current project.   Please change the project name in the plotting page to match the current project."





                    #####################################################################
                    if c is 96: # 96 is `
                        print "`"
                        self.Plot3DSquid(fwData)

                    #############################################################

                    if c is 101:  # 121 is e

                        print "e was pressed"
                        #print frame


                        #KalmanOverlay = not KalmanOverlay
                        FillMapDic = not FillMapDic

                        print "FillMapDic is ",FillMapDic
                        #MapErrorDicMeauredPoint,MapErrorDicMapPoint=self.FillMapDictionaries(fwData,cameraNames,f1,f2,frame1num,h5SquidDelay,frame2num,dat2,
                        #                         cameraMatrix2, distCoeffs2,dat1, cameraMatrix1, distCoeffs1,MapErrorDicMeauredPoint,MapErrorDicMapPoint)



                    #############################################################
                    #This callibrates the camera
                    if c is 114:  # 114 is r

                        print "r was pressed"
                        self.CalibrateCameraOpenCVCheckerBoard(place2SaveImages)

                    if c is 116:  # 116 is t
                        #get fiducial from another
                        print "t was pressed"
                        if FiducialFromAnotherProject!="":
                            h5copy=h5WritePath+"/"+FiducialFromAnotherProject
                            fwfcopy = h5py.File(h5copy, 'r')
                            try:

                                ResultRot = numpy.array(fwfcopy["data"][cameraNames[2]]["fiducial0_T"]['Fiducial_Rotation'])#l;l

                                ResultTran = numpy.array(fwfcopy["data"][cameraNames[2]]["fiducial0_T"]['Fiducial_Translation'])

                                try:
                                    fwData[cameraNames[2]].create_group("fiducial0_T")
                                except:
                                    del fwData[cameraNames[2]]["fiducial0_T"]
                                    fwData[cameraNames[2]].create_group("fiducial0_T")

                                fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                                    'Fiducial_Rotation', data=ResultRot)

                                fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                                    'Fiducial_Translation', data=ResultTran)
                                print "Information transferred"
                                try:
                                    RotMatMinClas = numpy.array(
                                        fwfcopy["data"][cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                                    TranMatMinClas = numpy.array(
                                        fwfcopy["data"][cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])

                                    MinClassData = numpy.array(
                                        fwfcopy["data"][cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatParams'])
                                    fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                                        'Fiducial_fromMinClasicRotMat', data=RotMatMinClas)

                                    fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                                        'Fiducial_fromMinClasicRotMatTrans', data=TranMatMinClas)
                                    fwData[cameraNames[2]]["fiducial0_T"].create_dataset(
                                        'Fiducial_fromMinClasicRotMatParams', data=MinClassData)
                                except:
                                    print "No MinClas"


                            except:
                                print "didn't work"



                    #############################################################
                    #This measures the error of the entire path modified with a subframe delay.  This is where you
                    # find the optimum subframe delay for your system. It loops through an array of them to find the
                    # minimum.
                    if c is 121:  # 52 is y
                        print "y was pressed"



                        pp1 = numpy.zeros(2)
                        pp11 = numpy.zeros(2)
                        pp12=numpy.zeros(2)
                        pp2 = numpy.zeros(2)

                        #delayfraction=0.5

                        insectkey = fwData[cameraNames[0]][Tobject + str(int(insectnumit))].keys()
                        bugkey = fwData[cameraNames[1]][Tobject + str(int(insectnumit))].keys()
                        # changing it to a list of strings
                        insectkey = map(str, insectkey)
                        insectkey = set(insectkey)

                        # we are now linking the two cameras with the frame delay qwq is this correct?
                        bugkeyF = numpy.array(map(int, bugkey))
                        bugkeyF = bugkeyF + frameDelay  # + or - ????
                        # print frameDelay,"frameDelay"
                        bugkeyF = map(str, bugkeyF)
                        # print bugkeyF,"bugkeyF"
                        bugkeyF = set(bugkeyF)
                        bugkey = set(bugkey)

                        # finding all of the coincident points
                        inter = bugkeyF.intersection(insectkey)


                        # changing it to a list
                        inter = list(inter)
                        delaything=40
                        for ji in range(delaything):
                            delayfractionM=float(ji)/delaything
                            DifferenceNorm=[]
                            for i in inter:
                                try:

                                    pp11[0] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(i))][
                                        0]
                                    pp11[1] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(i))][
                                        1]
                                    pp12[0] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(i)+1)][
                                        0]
                                    pp12[1] = fwData[cameraNames[0]][Tobject + str(int(insectnumit))][str(int(i)+1)][
                                        1]


                                    pp1=pp11+delayfractionM*(pp12-pp11)

                                    pp2[0] = \
                                    fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(i) - frameDelay)][0]
                                    pp2[1] = \
                                    fwData[cameraNames[1]][Tobject + str(int(insectnumit))][str(int(i) - frameDelay)][1]
                                    datdat1 = numpy.array(
                                        f1['F' + str(int(i))]['CameraPos'][:])

                                    datdat2 = numpy.array(
                                        f2['F' + str(int(i) - frameDelay)]['CameraPos'][:])

                                    CamOrgn1, pointvect1 = self.theVectors(datdat1, pp1,
                                                                           cameraMatrix1,
                                                                           distCoeffs1)
                                    CamOrgn2, pointvect2 = self.theVectors(datdat2, pp2,
                                                                           cameraMatrix2,
                                                                           distCoeffs2)

                                    cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(
                                        CamOrgn1,
                                        pointvect1,
                                        CamOrgn2,
                                        pointvect2)

                                    DifferenceNorm.append(
                                        numpy.linalg.norm(cloPo1 - cloPo2))

                                except:
                                    None

                            DifferenceNorm = numpy.array(DifferenceNorm)
                            print DifferenceNorm.mean(),"DifferenceNorm.mean()",len(DifferenceNorm),delayfractionM




                    #############################################################
                    if c is 117:  # 117 is u
                        if 0:
                            print "u was pressed"
                            root = tk.Tk()
                            root.withdraw()
                            answer = int(tkSimpleDialog.askinteger("Input", "Mirror to which number?", parent=root))

                            while answer < 0 or answer > insectMaxNum or answer is None:
                                tkMessageBox.showerror("Error", "Input a number in range.")
                                answer = int(tkSimpleDialog.askinteger("Input", "Mirror to which number?", parent=root))
                            print answer
                            #add an array or dictionary??? no array [3,4] etc [mirror,actual]
                            #keep adding to it.
                        if 1:
                            #What does Mirror to mean???
                            answer = raw_input("Type the object number you wish Mirror to and press enter: ")
                            isnotandint = True

                            while isnotandint:
                                try:
                                    answer = int(answer)
                                    if answer < 0 or answer > insectMaxNum:
                                        answer = raw_input("please enter an integer within the range of 0 to " + str(
                                            insectMaxNum) + " for the object number you wish Mirror to and press enter: ")
                                    else:
                                        isnotandint = False
                                except:
                                    answer = raw_input("please enter an integer for the object number you wish Mirror to and press enter: ")

                        if moviewatch == 0 or moviewatch == 2:

                            try:
                                g23 = fwData[cameraNames[0]].create_dataset(
                                    "mirrorCamera1", data=[[int(insectnumit),int(answer)]])

                            except:
                                mirrorarray=fwData[cameraNames[0]]["mirrorCamera1"].value.tolist()
                                MAindicator=0
                                for MA in range(len(mirrorarray)):

                                    if mirrorarray[MA][0]==int(insectnumit):#### this is the object that is a reflection
                                        mirrorarray[MA][1]=int(answer)#### this is the insect number from which the obove insect number is a reflection of.
                                        MAindicator=1
                                        break
                                if MAindicator==0:
                                    mirrorarray.append([int(insectnumit),int(answer)])
                                del fwData[cameraNames[0]]["mirrorCamera1"]
                                fwData[cameraNames[0]].create_dataset(
                                    "mirrorCamera1", data=mirrorarray)
                                print mirrorarray, "camera1"

                        if moviewatch == 1 or moviewatch == 2:

                            try:
                                g23 = fwData[cameraNames[1]].create_dataset(
                                    "mirrorCamera2", data=[[int(insectnumit), int(answer)]])

                            except:
                                mirrorarray = fwData[cameraNames[1]]["mirrorCamera2"].value.tolist()
                                MAindicator = 0
                                for MA in range(len(mirrorarray)):
                                    print mirrorarray[MA][0],mirrorarray[MA][1]
                                    if mirrorarray[MA][0] == int(insectnumit):
                                        mirrorarray[MA][1] = int(answer)
                                        MAindicator = 1
                                        break
                                if MAindicator == 0:
                                    mirrorarray.append([int(insectnumit), int(answer)])
                                del fwData[cameraNames[1]]["mirrorCamera2"]
                                fwData[cameraNames[1]].create_dataset(
                                    "mirrorCamera2", data=mirrorarray)
                                print mirrorarray, "camera2"

                    #############################################################

                    if c is 105:  # 105 is i
                        print "i was pressed"

                        if 0: #getting solar vector
                            print "solar vector"

                            RotMatMinClas = numpy.array(
                                fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMat'])

                            TranMatMinClas = numpy.array(
                                fwData[cameraNames[2]]["fiducial0_T"]['Fiducial_fromMinClasicRotMatTrans'])
                            #top point
                            path0 = "insect" + "23_" + "insect" + "23"
                            inX=[]
                            inY = []
                            inZ = []
                            for keyt in fwData[cameraNames[2]][path0].keys():
                                if 1:

                                    try:
                                        InterPoint = numpy.array(
                                            fwData[cameraNames[2]][path0][keyt]['3Dpoint'])
                                        inX.append(InterPoint[0])
                                        inY.append(InterPoint[1])
                                        inZ.append(InterPoint[2])
                                        # print str(int(fwDataKeysmin + jj)),"str(int(fwDataKeysmin + jj))"
                                        fwDataKeysInt.remove(fwDataKeysmin + jj)
                                        jj += 1
                                        # if jj>5:
                                        #    IsthereaPoint = False

                                    except:
                                        IsthereaPoint = False
                                        # inX.append(numpy.NaN)
                                        # inY.append(numpy.NaN)
                                        # inZ.append(numpy.NaN)

                            inX = numpy.array(inX)
                            inY = numpy.array(inY)
                            inZ = numpy.array(inZ)
                            toppoint=numpy.array([numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)])
                            toppoint=numpy.matmul(RotMatMinClas,toppoint)+TranMatMinClas
                            # print [numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)]

                            # bottom point
                            path0 = "insect" + "24_" + "insect" + "24"
                            inX = []
                            inY = []
                            inZ = []
                            for keyt in fwData[cameraNames[2]][path0].keys():
                                if 1:

                                    try:
                                        InterPoint = numpy.array(
                                            fwData[cameraNames[2]][path0][keyt]['3Dpoint'])
                                        inX.append(InterPoint[0])
                                        inY.append(InterPoint[1])
                                        inZ.append(InterPoint[2])
                                        # print str(int(fwDataKeysmin + jj)),"str(int(fwDataKeysmin + jj))"
                                        fwDataKeysInt.remove(fwDataKeysmin + jj)
                                        jj += 1
                                        # if jj>5:
                                        #    IsthereaPoint = False

                                    except:
                                        IsthereaPoint = False
                                        # inX.append(numpy.NaN)
                                        # inY.append(numpy.NaN)
                                        # inZ.append(numpy.NaN)

                            inX = numpy.array(inX)
                            inY = numpy.array(inY)
                            inZ = numpy.array(inZ)
                            bottompoint = numpy.array([numpy.mean(inX), numpy.mean(inY), numpy.mean(inZ)])
                            bottompoint = numpy.matmul(RotMatMinClas, bottompoint) + TranMatMinClas
                            SolarSquidVect=(bottompoint-toppoint)/numpy.linalg.norm(bottompoint-toppoint)
                            print SolarSquidVect,"SolarSquidVect"

                        if 1: # extracting color value:
                            print "getting color"

                            frameSaveForLater = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                            # HowManyFrames=1277
                            HowManyFrames = 400  # just for one
                            frame2use = frameSaveForLater
                            print frame2use, "frame2use"
                            cap = self.capture1
                            iu=1
                            squareLen=2
                            HowManySquid=21
                            SquidStep=10
                            SquidColorChange=numpy.zeros((HowManySquid*SquidStep,HowManyFrames))
                            fig5 = plt.figure(figsize=(6, 6))
                            for iu in range(HowManySquid):
                                greenChan = []
                                greenChanX = []
                                for hh in range(HowManyFrames):



                                    if hh==0:
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame2use - 1)
                                    ret, frameOut = cap.read()
                                    framecurrent=cap.get(cv2.CAP_PROP_POS_FRAMES)
                                    as_array = numpy.asarray(frameOut[:, :])
                                    #print framecurrent

                                    try:
                                        point1fromH5[0] = fwData[cameraNames[0]][Tobject + str(int(iu+1))][str(int(framecurrent))][0]

                                        point1fromH5[1] = fwData[cameraNames[0]][Tobject + str(int(iu+1))][str(int(framecurrent))][1]
                                        #print point1fromH5
                                        #print point1fromH5[0],point1fromH5[1],int(point1fromH5[0])-squareLen,int(point1fromH5[1])-squareLen
                                        frameToRead1 = as_array[int(point1fromH5[1]) - squareLen:int(
                                            point1fromH5[1]) + squareLen, int(point1fromH5[0]) - squareLen:int(
                                            point1fromH5[0]) + squareLen]  # for 1080 images
                                        # print numpy.shape(frameToRead1)
                                        # print frameToRead1
                                        ColorSquid = numpy.mean(numpy.mean(frameToRead1, axis=1), axis=0)
                                        greenChan.append(ColorSquid[1])
                                        greenChanX.append(hh)

                                    except:
                                        None



                                    #print cap.get(cv2.CAP_PROP_POS_FRAMES)




                                cap.set(cv2.CAP_PROP_POS_FRAMES, frameSaveForLater)
                                xthing=range(HowManyFrames)
                                #print iu, len(greenChan)
                                try:
                                    Greensmoth = savgol_filter(greenChan, 41, 1)
                                    Greensmoth=numpy.array(Greensmoth)
                                    Greensmoth=Greensmoth-Greensmoth[0]
                                    #print "frame2use current", cap.get(cv2.CAP_PROP_POS_FRAMES)

                                    #plt.plot(greenChanX,greenChan, label="", linestyle="", marker=".")
                                    plt.plot(greenChanX,Greensmoth, label="", linestyle="", marker=".")
                                    for hu in range(len(greenChanX)):
                                        for yu in range(SquidStep):
                                            #SquidColorChange[iu*10:iu*10+9][greenChanX[hu]]=Greensmoth[hu]
                                            SquidColorChange[iu*SquidStep+yu][greenChanX[hu]] = Greensmoth[hu]
                                    print iu+1
                                    print list(Greensmoth)
                                    print greenChanX
                                except:
                                    None
                            #cv2.imshow("SquidColorChange",SquidColorChange)
                            fig6 = plt.figure(figsize=(6, 6))

                            #plt.imshow(SquidColorChange, cmap='jet', clim=(0, 40))
                            #plt.show()


                        if 0:  # developmental  UV map for blender from the video.
                            # new h5
                            h5filenamewriteUVmap= h5filenamewrite.split(".")[0]+"bkgSTL_UVmap.h5"
                            if h5filenamewriteUVmap != "" and h5filenamewriteUVmap.split(".")[1] == "h5" and os.path.isdir(h5WritePath):
                                print "yes the h5 write", h5filenamewriteUVmap, os.path.isfile(h5filenamewriteUVmap)

                                if os.path.isfile(h5filenamewriteUVmap):
                                    fUV = h5py.File(h5filenamewriteUVmap, 'a')
                                    fUVheader = fUV.get('header')
                                    fUVData = fUV.get('data')


                                else:
                                    fUV = h5py.File(h5filenamewriteUVmap, 'w')
                                    fUVheader = fUV.create_group('header')

                                    fUVData = fUV.create_group('data')
                            self.extractUVdata(fUV,dat1,cameraMatrix1,distCoeffs1,width,height,frame1,polydata,obbTree)

                #############################################################
                #calculates the moving average of all the paths. The length of the average mask can be inputted in the
                # subroutine of the keyboard key.
                if c is 104:  # 104 is h
                    windownum=8
                    self.movingAverage(fwData,windownum)




                #############################################################
                #This goes to various frame numbers along the path, corresponding the to the beginning, 1st quartile,
                # median, 3rd quartile, and the end.
                if c is 106:  # 106 is j --- for VSLAM  shifing around the path  --- for squid
                    if UseSquidExcel==False:
                        #if UseFullGoPro6Video == "a":
                        if 0:
                            print "j"
                            #framebase=49010
                            framebase = 51765
                            framebase=65835
                            #framebase = 22250

                            self.capture1.set(cv2.CAP_PROP_POS_FRAMES, int(
                                framebase- 1 + frameDelay))
                            ret, frame1 = self.capture1.read()

                            self.capture2.set(cv2.CAP_PROP_POS_FRAMES,
                                              framebase - 1)
                            ret, frame2 = self.capture2.read()

                        #else:
                        if 1:
                            if jnum < jnumMax:
                                jnum += 1
                            else:
                                jnum = 0

                            try:
                                fwDataKeys = fwData[cameraNames[moviewatch]][Tobject + str(int(insectnumit))].keys()
                                fwDataKeysInt = numpy.array(map(int, fwDataKeys))

                                fwDataKeysmin = int(numpy.percentile(fwDataKeysInt, 100 * float(jnum) / float(jnumMax)))
                                print fwDataKeysmin, "fwDataKeysmin"

                                if moviewatch == 0 or moviewatch == 2:
                                    self.rewind(frame1num - fwDataKeysmin, self.capture1)
                                    ret, frame1 = self.capture1.read()

                                    self.rewind(frame1num - fwDataKeysmin, self.capture2)
                                    ret, frame2 = self.capture2.read()

                                if moviewatch == 1 or moviewatch == 2:
                                    self.rewind(frame2num - fwDataKeysmin, self.capture1)
                                    ret, frame1 = self.capture1.read()

                                    self.rewind(frame2num - fwDataKeysmin, self.capture2)
                                    ret, frame2 = self.capture2.read()

                                print "FRAME:", self.capture1.get(cv2.CAP_PROP_POS_FRAMES), self.capture2.get(
                                    cv2.CAP_PROP_POS_FRAMES)
                            except:
                                None
                            self.isPaused = True



                #############################################################
                #
                if c is 9:  ## c is TAB
                    print keysImagesIndex
                    if keysImagesIndex<3:
                        keysImagesIndex += 1
                        Kkeys = cv2.imread(keysImages[keysImagesIndex])
                        cv2.imshow("Keys1", Kkeys)
                        cv2.moveWindow("Keys1", 1400, 0)


                    elif keysImagesIndex==3:
                        keysImagesIndex += 1
                        cv2.destroyWindow("Keys1")
                    elif keysImagesIndex>3:
                        keysImagesIndex=-1


                #############################################################
                if UseNumbersForError == True:
                    if c is 48:  # 48 is 0
                        ErrorCircle = 1
                        ClickedErrorInd = True
                    if c is 49:  # 49 is 1
                        ErrorCircle = 2
                        ClickedErrorInd = True
                    if c is 50:  # 50 is 2
                        ErrorCircle = 3
                        ClickedErrorInd = True
                    if c is 51:  # 51 is 3
                        ErrorCircle = 4
                        ClickedErrorInd = True
                    if c is 52:  # 52 is 4
                        ErrorCircle = 5
                        ClickedErrorInd = True
                    if c is 53:  # 53 is 5
                        ErrorCircle = 20
                        ClickedErrorInd = True
                    if c is 54:  # 54 is 6
                        ErrorCircle = 25
                        ClickedErrorInd = True
                    if c is 55:  # 55 is 7
                        ErrorCircle = 30
                        ClickedErrorInd = True
                    if c is 56:  # 56 is 8
                        ErrorCircle = 35
                        ClickedErrorInd = True
                    if c is 57:  # 57 is 9
                        ErrorCircle = 40
                        ClickedErrorInd = True
                    if ClickedErrorInd == True:
                        UseFWError = False
                        if moviewatch == 0:
                            frame1num = self.capture1.get(cv2.CAP_PROP_POS_FRAMES)
                            ClickedError[int(frame1num)] = ErrorCircle
                        if moviewatch == 1:
                            frame2num = self.capture2.get(cv2.CAP_PROP_POS_FRAMES)
                            ClickedError[int(frame2num)] = ErrorCircle
                        ClickedErrorInd = False


                #############################################################
                if UseNumbersForError == False:


                    if c is 53:  # 53 is 5   assigning error


                        for rr in fwData[cameraNames[0]][Tobject + str(int(insectnumit))].keys():
                            try:
                                fwData[cameraNames[0]][Tobject + str(int(insectnumit))][rr][2] = \
                                    ErrorCircle
                                fwData[cameraNames[0]][Tobject + str(int(insectnumit))][rr][3] = \
                                    ErrorCircle
                            except:
                                print "None found"

                        for rr in fwData[cameraNames[1]][Tobject + str(int(insectnumit))].keys():
                            try:
                                fwData[cameraNames[1]][Tobject + str(int(insectnumit))][rr][2] = \
                                    ErrorCircle
                                fwData[cameraNames[1]][Tobject + str(int(insectnumit))][rr][3] = \
                                    ErrorCircle
                            except:
                                print "None found"

                    #############################################################
                    if c is 54:  # 54 is 6
                        viewGuidingLines = not viewGuidingLines


                    #############################################################
                    if c is 55:  # 55 is 7  extracting aolp from polarization videos
                        if 1:  # getting readout of entire video
                            aolpBKArr = []
                            greenIntenArr = []
                            bkgrndindexArr = []
                            maxforAolpMap = int(numpy.floor(length1 / 20.0))
                            if 1:
                                for gg in range(maxforAolpMap):#what is gg
                                    aolpBK, greenInten = self.ReturnAngleAndIntenOfBkgnd(bitmap1)
                                    print aolpBK, greenInten, gg, maxforAolpMap
                                    aolpBKArr.append(aolpBK)
                                    greenIntenArr.append(greenInten)
                                    bkgrndindexArr.append(gg + 20)
                                    self.fastforward(20, self.capture1)
                                    ret, frame1 = self.capture1.read()
                                    bitmap1, self.KeepTrack1 = self.TransformFrame(frame1, self.point1, magZoom, windowsize)

                                fig5 = plt.figure(figsize=(6, 6))
                                plt.subplot(211)
                                plt.plot(bkgrndindexArr, aolpBKArr, label="", linestyle="", marker=".")
                                plt.subplot(212)
                                plt.plot(bkgrndindexArr, greenIntenArr, label="", linestyle="", marker=".")

                                # save things
                                saveFile = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/ImagesForSquidAnalysis/AolpIMUGraphs/"

                                numpy.save(saveFile + ActualFileName1[0] + "_aolpBKArr" + ".npy", aolpBKArr)
                                numpy.save(saveFile + ActualFileName1[0] + "_greenIntenArr" + ".npy", greenIntenArr)
                                numpy.save(saveFile + ActualFileName1[0] + "_bkgrndindexArr" + ".npy", bkgrndindexArr)

                            fig5.savefig(
                                saveFile + '/' + ActualFileName1[0] + '_AolpIMUGraph.png', bbox_inches='tight', dpi=600)

                            plt.show()




                    if c==48: #48 is 0:
                        pathWrite = "C:/Users/Parrish/Documents/aOTIC/put on a hard drive/maps/logsandsticks.txt"
                        PathWrite = open(pathWrite, "w")



                        for ii in range(len(PointCloudFromORBPoints)):
                            PathWrite.write(
                                str(PointCloudFromORBPoints[ii][0]) + " " + str(
                                    PointCloudFromORBPoints[ii][1]) + " " + str(
                                        PointCloudFromORBPoints[ii][2])+ "\n")

                    #############################################################

                    if c is 97:  # 97 is a    track from mousepoint


                        if moviewatch==0:
                            ThreeDlineDicB1[DicBInt1] = [self.clicked1[1], self.clicked1[2]]
                            DicBInt1 += 1
                        if moviewatch==1:
                            ThreeDlineDicB2[DicBInt2] = [self.clicked2[1], self.clicked2[2]]
                            DicBInt2 += 1





                    #############################################################


                    if c==57: # 57 is 9
                        print "9 was pressed"

                        if 1:
                            #ThreeDlineDicB1
                            #DicBInt1
                            kmin=5e-05
                            def pathlengthfromList(pLen,lineDic):
                                yeetLen=0
                                inthePath=False
                                for pl in range(len(lineDic.keys())):
                                    if pl>0:
                                        xo, yo = (lineDic[pl - 1][0]), (lineDic[pl - 1][1])
                                        xo2, yo2 = (lineDic[pl][0]), (lineDic[pl][1])
                                        forLen=numpy.sqrt((xo2-xo)**2+(yo2-yo)**2)
                                        yeetLen+=forLen
                                        if yeetLen>=pLen:
                                            trueLen=pLen-(yeetLen-forLen)
                                            yeetPos=[(trueLen/forLen)*(xo2-xo)+xo,(trueLen/forLen)*(yo2-yo)+yo]
                                            inthePath=True
                                            break
                                if inthePath==False:
                                    yeetPos=[numpy.nan,numpy.nan]
                                return yeetPos

                            yeetLen=0
                            for pl in range(len(ThreeDlineDicB1.keys())):
                                if pl > 0:
                                    xo, yo = (ThreeDlineDicB1[pl - 1][0]), (ThreeDlineDicB1[pl - 1][1])
                                    xo2, yo2 = (ThreeDlineDicB1[pl][0]), (ThreeDlineDicB1[pl][1])
                                    forLen = numpy.sqrt((xo2 - xo) ** 2 + (yo2 - yo) ** 2)
                                    yeetLen += forLen
                            totalpoints=100
                            stepLen=yeetLen/totalpoints
                            kp=[]
                            for pi in range(totalpoints):
                                kp.append(pathlengthfromList(stepLen*pi,ThreeDlineDicB1))
                            yeetLen=0
                            for pl in range(len(ThreeDlineDicB2.keys())):
                                if pl > 0:
                                    xo, yo = (ThreeDlineDicB2[pl - 1][0]), (ThreeDlineDicB2[pl - 1][1])
                                    xo2, yo2 = (ThreeDlineDicB2[pl][0]), (ThreeDlineDicB2[pl][1])
                                    forLen = numpy.sqrt((xo2 - xo) ** 2 + (yo2 - yo) ** 2)
                                    yeetLen += forLen
                            totalpoints2=int(numpy.floor(yeetLen/stepLen))
                            kp2=[]
                            for pi in range(totalpoints2):
                                kp2.append(pathlengthfromList(stepLen*pi,ThreeDlineDicB2))
                            #print len(kp),len(kp2)
                            #print kp
                            #print kp2
                            for kpi in range(len(kp)):

                                kpi2minarray = []
                                kpi2minarrayPt = []
                                for kp2i in range(len(kp2)):
                                    ImgPoint2 = kp2[kp2i]
                                    ImgPoint1 = kp[kpi]
                                    CamOrgn1, pointvect1 = self.theVectors(dat1, ImgPoint1, cameraMatrix1, distCoeffs1)
                                    CamOrgn2, pointvect2 = self.theVectors(dat2, ImgPoint2, cameraMatrix2, distCoeffs2)
                                    # print pointvect1,pointvect2
                                    cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1,
                                                                                              CamOrgn2,
                                                                                              pointvect2)

                                    cloPoAve = [(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                                (cloPo1[2] + cloPo2[2]) / 2]
                                    kpi2minarrayPt.append(cloPoAve)
                                    cloPoError = numpy.linalg.norm(cloPo1 - cloPo2) / 2
                                    # print ImgPoint1, ImgPoint2, cloPoError
                                    kpi2minarray.append(cloPoError)
                                print kpi, min(kpi2minarray),kpi2minarray.index(min(kpi2minarray)),kpi2minarrayPt[kpi2minarray.index(min(kpi2minarray))]
                                if min(kpi2minarray) < kmin:
                                   PointCloudFromORBPoints.append(
                                        kpi2minarrayPt[kpi2minarray.index(min(kpi2minarray))])


                        if 0:# orb points
                            windowsize = [WindowWidth * 2, int(WindowWidth * frameratio * 2)]
                            Bbitmap1, self.KeepTrack1 = self.TransformFrame(frame1, self.point1, magZoom, windowsize)
                            Bbitmap2, self.KeepTrack2 = self.TransformFrame(frame2, self.point2, magZoom, windowsize)
                            kmin=0.05

                            orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
                            kp = orb.detect(Bbitmap1, None)
                            kp, des = orb.compute(Bbitmap1, kp)
                            print kp[0].pt
                            #print des
                            kp2 = orb.detect(Bbitmap2, None)
                            kp2, des2 = orb.compute(Bbitmap2, kp2)
                            print kp2[1].pt


                            if 0:
                                for kpi in kp:

                                    kpi2minarray=[]
                                    kpi2minarrayPt=[]
                                    for kp2i in kp2:
                                        ImgPoint2 = kp2i.pt
                                        ImgPoint1 = kpi.pt
                                        CamOrgn1, pointvect1 = self.theVectors(dat1, ImgPoint1, cameraMatrix1, distCoeffs1)
                                        CamOrgn2, pointvect2 = self.theVectors(dat2, ImgPoint2, cameraMatrix2, distCoeffs2)
                                        # print pointvect1,pointvect2
                                        cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1,
                                                                                                  CamOrgn2,
                                                                                                  pointvect2)

                                        cloPoAve = [(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                                    (cloPo1[2] + cloPo2[2]) / 2]
                                        kpi2minarrayPt.append(cloPoAve)
                                        cloPoError = numpy.linalg.norm(cloPo1 - cloPo2) / 2
                                        #print ImgPoint1, ImgPoint2, cloPoError
                                        kpi2minarray.append(cloPoError)
                                    #print kpi.pt, min(kpi2minarray),kpi2minarray.index(min(kpi2minarray)),kpi2minarrayPt[kpi2minarray.index(min(kpi2minarray))]
                                    if min(kpi2minarray)<kmin:
                                        PointCloudFromORBPoints.append(kpi2minarrayPt[kpi2minarray.index(min(kpi2minarray))])


                            if 1:
                                # create BFMatcher object
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                                # Match descriptors.
                                matches = bf.match(des, des2)

                                # Sort them in the order of their distance.
                                matches = sorted(matches, key=lambda x: x.distance)

                                print len(matches)
                                for matchthings in matches:
                                    img1_idx = matchthings.queryIdx
                                    img2_idx = matchthings.trainIdx
                                    #print img1_idx,img2_idx
                                    ImgPoint1=kp[img1_idx].pt
                                    ImgPoint2=kp2[img2_idx].pt
                                    #print ImgPoint1,ImgPoint2
                                    #for kp2i in kp2:
                                    if 1:
                                        #ImgPoint2=kp2i.pt
                                        CamOrgn1, pointvect1 = self.theVectors(dat1, ImgPoint1, cameraMatrix1, distCoeffs1)
                                        CamOrgn2, pointvect2 = self.theVectors(dat2,ImgPoint2 , cameraMatrix2, distCoeffs2)
                                        #print pointvect1,pointvect2
                                        cloPo1, cloPo2 = self.findClosestPointsOfnonParallelLines(CamOrgn1, pointvect1, CamOrgn2,
                                                                                                  pointvect2)

                                        cloPoAve = [(cloPo1[0] + cloPo2[0]) / 2, (cloPo1[1] + cloPo2[1]) / 2,
                                                    (cloPo1[2] + cloPo2[2]) / 2]
                                        cloPoError = numpy.linalg.norm(cloPo1 - cloPo2) / 2
                                        #print ImgPoint1, ImgPoint2,cloPoError
                                        if cloPoError < kmin:
                                            PointCloudFromORBPoints.append(cloPoAve)


                            if 0:

                                #cv2.circle(bitmap1, (int(p1[0, 0, 0]), int(p1[0, 0, 1])), 2, (0, 244, 23), 3)
                                bitmap1 = cv2.drawKeypoints(bitmap1, kp, bitmap1, color=(255, 0, 0), flags=0)








                    #############################################################




            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################
            #######################################################################
                            #######################################################################


            ####   This is the section only used for SQUID analysis



            if UseSquidExcel == True:

                #############################################################
                # This is a selection functionality.   In the Squid mode this is used for making the outline of the ROI.
                # check the 9 button for more information
                if c is 97:  # 97 is a    track from mousepoint

                    ClickedDicB[DicBInt] = [self.clicked1[1], self.clicked1[2]]
                    DicBInt += 1

                #############################################################
                #This sets the video from which the image was taken to run so that one can better determine if the
                # fish is swimming toward the camera or away.
                if c is 32:  # is SPACE BAR
                    setFrame = 1

                #############################################################
                if c is 112:  # 112 is p  # what is this?  Setting the roll to zero???  FOr squid   zxz
                    datOrdAltAlt=datOrdAlt
                    resx0 = 0
                    # print "datOrdAlt=", datOrdAlt
                    # print "datOrdAltAlt=",datOrdAltAlt
                    RotAroundY=numpy.array([[1,0,0],[0,1.0,0.0],[0,0,1.0]])
                    cameraOrginW = numpy.array([datOrdAltAlt[9], datOrdAltAlt[10], datOrdAltAlt[11]])
                    Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
                    cameraZ = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], datOrdAltAlt[8]])
                    cameraZX = numpy.array([datOrdAltAlt[6], datOrdAltAlt[7], 0])
                    cameraPitch = (180 / 3.1415926) * numpy.arccos(
                        numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))

                    cameraPitch8 = (180 / 3.1415926) * numpy.arctan(datOrdAltAlt[8] / numpy.linalg.norm(cameraZX))

                    print "squid pitch", resx0
                    print "camera pitch", cameraPitch8
                    print "Asimuth", Asimuth
                    print "Distance", numpy.linalg.norm(cameraOrginW)

                    cameraZ = numpy.array([-datOrdAltAlt[9], -datOrdAltAlt[10], -datOrdAltAlt[11]])
                    cameraZX = numpy.array([-datOrdAltAlt[9], -datOrdAltAlt[10], 0])

                    zvect1 = numpy.zeros(3)

                    xvect1 = numpy.zeros(3)

                    xvect1[0] = datOrdAltAlt[0]
                    xvect1[1] = datOrdAltAlt[1]
                    xvect1[2] = datOrdAltAlt[2]

                    yvect1[0] = datOrdAltAlt[3]
                    yvect1[1] = datOrdAltAlt[4]
                    yvect1[2] = datOrdAltAlt[5]

                    zvect1[0] = datOrdAltAlt[6]
                    zvect1[1] = datOrdAltAlt[7]
                    zvect1[2] = datOrdAltAlt[8]

                    zhat = numpy.array([0, 0, 1.0])
                    # print zhat,zvect1
                    camHrizontal = numpy.cross(zvect1, zhat)
                    # print camHrizontal
                    camHrizontal = numpy.array(camHrizontal)
                    camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
                    CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
                        numpy.linalg.norm(xvect1) * numpy.linalg.norm(
                            camHrizontal))) * 180.0 / 3.1415926

                    print "camera Roll", CameraRoll
                    cameraPitch = (180 / 3.1415926) * numpy.arctan(
                        -datOrdAltAlt[11] / numpy.linalg.norm(cameraZX))
                    print "camera pitch from camorig", cameraPitch
                    print "camera pitch differece", cameraPitch - cameraPitch8

                    #########################################################################################################
                    # getting original numbers here    from the unfitted PnP fit.

                    cameraOrginW = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])
                    Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
                    cameraZ = numpy.array([datOrdAlt[6], datOrdAlt[7], datOrdAlt[8]])
                    cameraZX = numpy.array([datOrdAlt[6], datOrdAlt[7], 0])
                    cameraPitch = (180 / 3.1415926) * numpy.arccos(
                        numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))
                    cameraPitch8 = (180 / 3.1415926) * numpy.arctan(datOrdAlt[8] / numpy.linalg.norm(cameraZX))
                    zvect1 = numpy.zeros(3)

                    xvect1 = numpy.zeros(3)

                    xvect1[0] = datOrdAlt[0]
                    xvect1[1] = datOrdAlt[1]
                    xvect1[2] = datOrdAlt[2]

                    yvect1[0] = datOrdAlt[3]
                    yvect1[1] = datOrdAlt[4]
                    yvect1[2] = datOrdAlt[5]

                    zvect1[0] = datOrdAlt[6]
                    zvect1[1] = datOrdAlt[7]
                    zvect1[2] = datOrdAlt[8]

                    zhat = numpy.array([0, 0, 1.0])
                    # print zhat,zvect1
                    camHrizontal = numpy.cross(zvect1, zhat)
                    # print camHrizontal
                    camHrizontal = numpy.array(camHrizontal)
                    camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
                    CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
                        numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926
                    # print CameraRoll, "CameraRoll"




                    print "squid pitch", 0
                    print "camera pitch orig", cameraPitch8
                    print "Asimuth orig", Asimuth

                    print "camera Roll", CameraRoll

                    cameraZ = numpy.array([-datOrdAlt[9], -datOrdAlt[10], -datOrdAlt[11]])
                    cameraZX = numpy.array([-datOrdAlt[9], -datOrdAlt[10], 0])

                    cameraPitch = (180 / 3.1415926) * numpy.arccos(
                        numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))

                    cameraPitch = (180 / 3.1415926) * numpy.arctan(
                        -datOrdAlt[11] / numpy.linalg.norm(cameraZX))
                    print "camera pitch from camorig", cameraPitch
                    print "camera pitch differece", cameraPitch - cameraPitch8


                #############################################################
                #moves through whether or not the fish is toward the camera, or away from the camera, or parallel to
                # the camera.  The q button sequentially moves through these options.
                if c is 113:  # 113 is q
                    if UseSquidExcel==True:
                        try :
                            fwData["direction"][str(frame1num)]["TorA"]
                        except:
                            fwData["direction"][str(frame1num)].create_dataset("TorA", data=["?"])
                    if towardOrAwayFromSun=="?":
                        del fwData["direction"][str(frame1num)]["TorA"]
                        fwData["direction"][str(frame1num)].create_dataset("TorA",data=["T"])
                    elif towardOrAwayFromSun=="T":
                        del fwData["direction"][str(frame1num)]["TorA"]
                        fwData["direction"][str(frame1num)].create_dataset("TorA", data=["A"])
                    elif towardOrAwayFromSun=="A":
                        del fwData["direction"][str(frame1num)]["TorA"]
                        fwData["direction"][str(frame1num)].create_dataset("TorA", data=["P"])#changed this for fish
                    elif towardOrAwayFromSun=="P":
                        del fwData["direction"][str(frame1num)]["TorA"]
                        fwData["direction"][str(frame1num)].create_dataset("TorA", data=["T"])





                ##############################333
                #############################################################
                #############################################################

                #try rotation

                RotMat[0][0] = datOrdAlt[0]
                RotMat[1][0] = datOrdAlt[1]
                RotMat[2][0] = datOrdAlt[2]
                RotMat[0][1] = datOrdAlt[3]
                RotMat[1][1] = datOrdAlt[4]
                RotMat[2][1] = datOrdAlt[5]
                RotMat[0][2] = datOrdAlt[6]
                RotMat[1][2] = datOrdAlt[7]
                RotMat[2][2] = datOrdAlt[8]

                # camera origin
                TranMat[0] = datOrdAlt[9]
                TranMat[1] = datOrdAlt[10]
                TranMat[2] = datOrdAlt[11]

                #RotMat = numpy.transpose(RotMat)
                #TranMat = (-1) * numpy.matmul(RotMat, TranMat)


                #############################################################
                if c is 121:  # 52 is y
                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta), -numpy.sin(theta)], [0, numpy.sin(theta), numpy.cos(theta)]])

                    RotMat = numpy.matmul(rotx, RotMat)
                    TranMat=numpy.matmul(rotx,TranMat)
                    rottag = 1

                #############################################################
                if c is 117:  # 117 is u
                    rotx = numpy.array(
                        [[1, 0, 0], [0, numpy.cos(theta), numpy.sin(theta)], [0, -numpy.sin(theta), numpy.cos(theta)]])
                    #rotx = self.rotation_matrix(xvect1, (-1 * theta))
                    RotMat = numpy.matmul(rotx, RotMat)
                    TranMat=numpy.matmul(rotx,TranMat)
                    rottag = 1

                #############################################################
                if c is 116:  # 116 is t
                    roty = numpy.array(
                        [[numpy.cos(theta), 0, numpy.sin(theta)], [0, 1, 0], [-numpy.sin(theta), 0, numpy.cos(theta)]])
                    #roty = self.rotation_matrix(yvect1, theta)
                    RotMat = numpy.matmul(roty, RotMat) #what was
                    TranMat=numpy.matmul(roty,TranMat)
                    #RotMat = numpy.matmul(RotMat,roty)
                    #RotMat = numpy.matmul(numpy.transpose(roty),RotMat)
                    rottag = 1

                #############################################################
                if c is 111:  # 111 is o
                    roty = numpy.array(
                        [[numpy.cos(-fishRoll*3.14159/180), 0, -numpy.sin(-fishRoll*3.14159/180)], [0, 1, 0], [numpy.sin(-fishRoll*3.14159/180), 0, numpy.cos(-fishRoll*3.14159/180)]])

                    RotMat = numpy.matmul(roty, RotMat)
                    TranMat=numpy.matmul(roty,TranMat)
                    rottag = 1

                #############################################################
                if c is 114:  # 114 is r
                    roty = numpy.array(
                        [[numpy.cos(theta), 0, -numpy.sin(theta)], [0, 1, 0], [numpy.sin(theta), 0, numpy.cos(theta)]])

                    RotMat = numpy.matmul(roty, RotMat)
                    TranMat=numpy.matmul(roty,TranMat)
                    rottag = 1

                #############################################################
                if c is 119:  # 54 is w
                    rotz = numpy.array(
                        [[numpy.cos(theta), -numpy.sin(theta), 0], [numpy.sin(theta), numpy.cos(theta), 0], [0, 0, 1]])
                    RotMat = numpy.matmul(rotz, RotMat)
                    TranMat=numpy.matmul(rotz,TranMat)
                    rottag=1
                if c is 101:  # 121 is e
                    rotz = numpy.array(
                        [[numpy.cos(theta), numpy.sin(theta), 0], [-numpy.sin(theta), numpy.cos(theta), 0], [0, 0, 1]])
                    RotMat = numpy.matmul(rotz, RotMat)
                    TranMat = numpy.matmul(rotz, TranMat)
                    rottag = 1



                if c is 98:  # 49 is b
                    TranMat=TranMat+TransStep*TranMat
                    rottag = 1
                if c is 104:  # 113 is h
                    TranMat = TranMat - TransStep * TranMat
                    rottag = 1

                if c is 106:  # 50 is j
                    TranMat[1] += TransStep
                    rottag = 1
                if c is 107:  # 119 is k
                    TranMat[1] -= TransStep
                    rottag = 1


                if c is 118:  # 51 is v
                    TranMat[2] += TransStep
                    rottag = 1
                if c is 100:  # 101 is d
                    TranMat[2] -= TransStep
                    rottag = 1


                #RotMat = numpy.transpose(RotMat)
                #TranMat = (-1) * numpy.matmul(RotMat, TranMat)

                datOrdAlt[0] = RotMat[0][0]
                datOrdAlt[1] = RotMat[1][0]
                datOrdAlt[2] = RotMat[2][0]
                datOrdAlt[3] = RotMat[0][1]
                datOrdAlt[4] = RotMat[1][1]
                datOrdAlt[5] = RotMat[2][1]
                datOrdAlt[6] = RotMat[0][2]
                datOrdAlt[7] = RotMat[1][2]
                datOrdAlt[8] = RotMat[2][2]

                # camera origin
                datOrdAlt[9] = TranMat[0]
                datOrdAlt[10] = TranMat[1]
                datOrdAlt[11] = TranMat[2]

                if rottag==1:
                    CircleArray, Outline, twoAve = self.getMeshOutline(TwoDMarkedPoints2, datOrdAlt, cameraMatrix1,
                                                                       distCoeffs1, obbTree)
                    RotMatCopy=RotMat
                    if 1:  # working with IMU   ### for f  zxz
                        try:
                            RealHeading = fwData["direction"][str(frame1num)]["Real IMU Solar Heading"].value
                            # the Roll and the pitch are switched
                            RealRoll = fwData["direction"][str(frame1num)]["Real IMU camera pitch"].value
                            RealPitch = fwData["direction"][str(frame1num)]["Real IMU camera roll"].value
                        except:
                            print "no IMU"

                        RealPitch = RealPitch - 10
                        RealRoll = RealRoll
                        print RealHeading, "RealHeading"
                        print RealPitch, "RealPitch"
                        print RealRoll, "RealRoll"

                        # now to make vectors out of this

                        cz = numpy.array(
                            [-numpy.cos(RealPitch * 3.14159 / 180) * numpy.cos(RealHeading * 3.14159 / 180),
                             numpy.cos(RealPitch * 3.14159 / 180) * numpy.sin(RealHeading * 3.14159 / 180),
                             numpy.sin(RealPitch * 3.14159 / 180)])
                        # cz=-cz
                        cx = numpy.array([-numpy.sin(RealHeading * 3.14159 / 180),
                                          -numpy.cos(RealHeading * 3.14159 / 180),
                                          0.0])
                        cx = -cx
                        cy = numpy.array([numpy.sin(RealPitch * 3.14159 / 180) * numpy.cos(RealHeading * 3.14159 / 180),
                                          -numpy.sin(RealPitch * 3.14159 / 180) * numpy.sin(
                                              RealHeading * 3.14159 / 180),
                                          numpy.cos(RealPitch * 3.14159 / 180)])
                        cy = -cy

                        ###### what???
                        RollRotMat = self.rotation_matrix(cz, RealRoll * 3.14159 / 180)
                        cx = numpy.matmul(RollRotMat, cx)
                        cy = numpy.matmul(RollRotMat, cy)

                        # Getting the rotation matrix for the IMU vectors
                        RotMatIMU = numpy.zeros((3, 3))

                        RotMatIMU[0, 0] = cx[0]
                        RotMatIMU[1, 0] = cx[1]
                        RotMatIMU[2, 0] = cx[2]
                        RotMatIMU[0, 1] = cy[0]
                        RotMatIMU[1, 1] = cy[1]
                        RotMatIMU[2, 1] = cy[2]
                        RotMatIMU[0, 2] = cz[0]
                        RotMatIMU[1, 2] = cz[1]
                        RotMatIMU[2, 2] = cz[2]

                        zhatTrans = numpy.zeros(3)
                        yhatTrans = numpy.zeros(3)
                        xhatTrans = numpy.zeros(3)
                        xhat = numpy.array([1.0, 0, 0])
                        yhat = numpy.array([0, -1.0, 0])
                        zhat = numpy.array([0, 0, 1.0])
                        #rot3 = numpy.array([RotMatCopy[0, 2], RotMatCopy[1, 2], RotMatCopy[2, 2]])
                        if 1:
                            zhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), zhat))
                            yhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), yhat))
                            xhatTrans = numpy.matmul(RotMatIMU, numpy.matmul(numpy.linalg.inv(RotMatCopy), xhat))
                            # print rot3,"rot3"

                        # print "fish heading", numpy.arctan2(yhatTrans[1],-yhatTrans[0])*180/3.14159
                        fhead = numpy.arctan2(yhatTrans[1], -yhatTrans[0]) * 180 / 3.14159
                        if fhead < 0:
                            fhead = fhead + 360
                        print "fish heading", fhead
                        yhatTransnonZ = numpy.array([yhatTrans[0], yhatTrans[1], 0])
                        # print "fish pitch", numpy.arccos(numpy.dot(yhatTransnonZ,zhatTrans)/(numpy.linalg.norm(yhatTransnonZ)*numpy.linalg.norm(zhatTrans)))*180/3.14159
                        print "fish pitch", numpy.arctan(
                            yhatTrans[2] / numpy.linalg.norm(yhatTransnonZ)) * 180 / 3.14159
                        print zhatTrans, "zhatTrans"
                        yzcross = numpy.cross(numpy.cross(yhatTrans, zhat), yhatTrans)
                        yzcross = yzcross / numpy.linalg.norm(yzcross)
                        fishRoll = numpy.arccos(numpy.dot(yzcross, zhatTrans) / (
                            numpy.linalg.norm(yzcross) * numpy.linalg.norm(zhatTrans))) * 180 / 3.14159
                        ####????

                        # the sign...
                        fishRoll = fishRoll * numpy.dot(
                            numpy.cross(zhatTrans, yzcross) / numpy.linalg.norm(numpy.cross(zhatTrans, yzcross)),
                            yhatTrans / numpy.linalg.norm(yhatTrans))
                        print "coplanar", numpy.dot(
                            numpy.cross(zhatTrans, yzcross) / numpy.linalg.norm(numpy.cross(zhatTrans, yzcross)),
                            yhatTrans / numpy.linalg.norm(yhatTrans))

                        print "coplanar", numpy.dot(
                            numpy.cross(zhatTrans, yzcross),
                            yhatTrans)

                        print "fish Roll", fishRoll



                        # finding fishRelAzimuth with the yvector and the camera vector.
                        fishRelAzimuth = numpy.arccos(numpy.dot(cz, yhatTrans) / (
                            numpy.linalg.norm(cz) * numpy.linalg.norm(yhatTrans))) * 180 / 3.14159
                        ####????

                        # the sign...
                        azisign = numpy.dot(
                            numpy.cross(cz, yhatTrans),
                            zhatTrans)
                        if azisign < 0:
                            fishRelAzimuth = fishRelAzimuth * -1.0

                        print "Fish relative azimuth", fishRelAzimuth
                        print fwData["direction"][str(frame1num)]["TorA"][0]



                    rottag=0


                #####################################################################
                #This does a PnP fit between the SLT mesh and the specific marked points on the image. This returns
                # the camera positions for the fit. This can also use a randomization in the inputs to determine an
                # error for the PnP fit.
                if c is 102:  # 102 is f
                    UseRandom = False
                    #for ic in range(100):
                    if 1:
                        newProjMesh, datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt,ThreeDMarkedPoints2,TwoDMarkedPoints2,Outline,fishRoll =self.PnPtoFacePoints(ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num, cameraMatrix1,
                                            distCoeffs1, pcMat,Tobject,obbTree,UseRandom,insectnumit)

                    # trying to figure out the displacement from the camera:
                    CamOrgn1, pointvect1 = self.theVectors(dat1, [640, 360], cameraMatrix1, distCoeffs1)
                    pointvect1 = numpy.array(pointvect1)
                    zvector = numpy.array([0, 0, 1])
                    xvector = numpy.array([1, 0, 0])
                    yvector = numpy.array([0, 1, 0])

                    print "x angle", numpy.arctan(xvector.dot(pointvect1) / zvector.dot(pointvect1)) * 180 / 3.14159
                    print "y angle", numpy.arctan(-yvector.dot(pointvect1) / zvector.dot(pointvect1)) * 180 / 3.14159
                    #print datOrdAlt
                    #del fwSTL["463"]  Just doing this
                    #del fwSTL["462"]
                    #del fwSTL["467"]



                #####################################################################
                #This does a PnP fit between the SLT mesh and the specific marked points on the image AND the outline
                # of the object selected by the user.  This returns the camera positions for the fit.  The points are
                # selected from 20 to 39.
                if c is 105:  # 105 is i
                    datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt,Outline,corEdgepnts,resx0,fishRoll=self.PnPtoOutlinePoints(TwoDMarkedPoints2, datOrdAlt, cameraMatrix1, distCoeffs1, obbTree,
                                       ThreeDMarkedPoints2, Tobject, fwData, cameraNames, moviewatch, frame1num,Ptarlen,width,height,makeInt)


                #####################################################################
                #This is an earlier way to determine the error for the Pnp processes.   I didn't use it for the last one.
                if c is 99: # 99 is c  #what is this zxz
                    UseRandom=True
                    cameraPitchArr = []
                    AsimuthArr = []
                    CameraRollArr = []
                    cameraPitchPosArr = []
                    for i in range(10):
                        newProjMesh, datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt, ThreeDMarkedPoints2, TwoDMarkedPoints2, Outline = self.PnPtoFacePoints(
                            ThreeDMarkedPoints, fwData, cameraNames, moviewatch, frame1num, cameraMatrix1,
                            distCoeffs1, pcMat, Tobject, obbTree,UseRandom,insectnumit)
                        if 1:
                            datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt, Outline, corEdgepnts,resx0 = self.PnPtoOutlinePoints(
                                TwoDMarkedPoints2, datOrdAlt, cameraMatrix1, distCoeffs1, obbTree,
                                ThreeDMarkedPoints2, Tobject, fwData, cameraNames, moviewatch, frame1num, Ptarlen, width,height, makeInt)
                            datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt, Outline, corEdgepnts,resx0 = self.PnPtoOutlinePoints(
                                TwoDMarkedPoints2, datOrdAlt, cameraMatrix1, distCoeffs1, obbTree,
                                ThreeDMarkedPoints2, Tobject, fwData, cameraNames, moviewatch, frame1num, Ptarlen, width, height,
                                makeInt)
                            datOrdAlt, CameraRoll, RotAroundY, datOrdAltAlt, Outline, corEdgepnts,resx0 = self.PnPtoOutlinePoints(
                                TwoDMarkedPoints2, datOrdAlt, cameraMatrix1, distCoeffs1, obbTree,
                                ThreeDMarkedPoints2, Tobject, fwData, cameraNames, moviewatch, frame1num, Ptarlen, width, height,
                                makeInt)

                            cameraOrginW = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])
                            Asimuth = numpy.arctan2(cameraOrginW[1], cameraOrginW[0]) * 180 / 3.14159
                            cameraZ = numpy.array([datOrdAlt[6], datOrdAlt[7], datOrdAlt[8]])
                            cameraZX = numpy.array([datOrdAlt[6], datOrdAlt[7], 0])
                            cameraPitch = (180 / 3.1415926) * numpy.arccos(
                                numpy.dot(cameraZ, cameraZX) / (numpy.linalg.norm(cameraZ) * numpy.linalg.norm(cameraZX)))
                            cameraPitch8 = (180 / 3.1415926) * numpy.arctan(datOrdAlt[8] / numpy.linalg.norm(cameraZX))

                            zvect1 = numpy.zeros(3)
                            yvect1 = numpy.zeros(3)

                            xvect1 = numpy.zeros(3)

                            xvect1[0] = datOrdAlt[0]
                            xvect1[1] = datOrdAlt[1]
                            xvect1[2] = datOrdAlt[2]

                            yvect1[0] = datOrdAlt[3]
                            yvect1[1] = datOrdAlt[4]
                            yvect1[2] = datOrdAlt[5]

                            zvect1[0] = datOrdAlt[6]
                            zvect1[1] = datOrdAlt[7]
                            zvect1[2] = datOrdAlt[8]

                            zhat = numpy.array([0, 0, 1.0])
                            # print zhat,zvect1
                            camHrizontal = numpy.cross(zvect1, zhat)
                            # print camHrizontal
                            camHrizontal = numpy.array(camHrizontal)
                            camHrizontal = camHrizontal / numpy.linalg.norm(camHrizontal)
                            CameraRoll = numpy.arccos(numpy.dot(xvect1, camHrizontal) / (
                                numpy.linalg.norm(xvect1) * numpy.linalg.norm(camHrizontal))) * 180.0 / 3.1415926
                            # print CameraRoll, "CameraRoll"




                            print "camera pitch orig", cameraPitch
                            cameraPitchArr.append(cameraPitch)
                            print "Asimuth orig", Asimuth
                            AsimuthArr.append(Asimuth)

                            print "camera Roll", CameraRoll
                            CameraRollArr.append(CameraRoll)
                            cameraZX = numpy.array([-datOrdAlt[9], -datOrdAlt[10], 0])


                            cameraPitch = (180 / 3.1415926) * numpy.arctan(
                                -datOrdAlt[11] / numpy.linalg.norm(cameraZX))
                            print "camera pitch from camorig", cameraPitch
                            cameraPitchPosArr.append(cameraPitch)
                    cameraPitchArr =numpy.array(cameraPitchArr)
                    AsimuthArr = numpy.array(AsimuthArr)
                    CameraRollArr = numpy.array(CameraRollArr)
                    cameraPitchPosArr = numpy.array(cameraPitchPosArr)
                    print numpy.mean(cameraPitchArr),numpy.std(cameraPitchArr),"cameraPitchArr"
                    print numpy.mean(AsimuthArr),numpy.std(AsimuthArr),"AsimuthArr"
                    print numpy.mean(CameraRollArr),numpy.std(CameraRollArr),"CameraRollArr"
                    print numpy.mean(cameraPitchPosArr),numpy.std(cameraPitchPosArr),"cameraPitchposArr"


                if UseNumbersForError == False:

                    #####################################################################

                    if c is 54:  # 54 is 6
                        #qwq
                        for hh in range(len(insectPointArray)):
                            frame1num = SquidIndex
                            try:
                                fwData[cameraNames[0]][insectPointArray[hh]].create_dataset(
                                    str(int(frame1num)), data=[projPoints[hh][0], projPoints[hh][1], 4, 4])

                            except:
                                fwData[cameraNames[0]][insectPointArray[hh]][str(int(frame1num))][
                                    0] = \
                                    projPoints[hh][0]
                                fwData[cameraNames[0]][insectPointArray[hh]][str(int(frame1num))][
                                    1] = \
                                    projPoints[hh][1]

                    #####################################################################
                    if c is 49:  # 49 is 1
                        viewSquidMeshStuff = not viewSquidMeshStuff
                        print viewSquidMeshStuff

                    #####################################################################
                    if c is 50:  # 50 is 2
                        insectnumit=1

                    #####################################################################
                    if c is 51:  # 51 is 3
                        insectnumit = 20

                    #####################################################################
                    if c is 52:  # 52 is 4
                        insectnumit = 40

                    #####################################################################
                    if c is 55:  # 55 is 7  extracting aolp from missael videos

                        if 1:# looking at polarization IMU
                            print self.ReturnAngleArrayOfBkgnd(bitmap1,dat1, cameraMatrix1, distCoeffs1, frame1num)

                    #####################################################################
                    #this is used only with the squid portion of this program.  The ROI is imaged
                    #on the frame with a blue color.  This switches the RO's incramentally/
                    if c is 48:  ## 48 is 0  changing the roi number to view and to make.
                        if RoiStringIndex< len(RoiStringArray)-1:
                            RoiStringIndex+=1
                        else:
                            RoiStringIndex=0

                        RoiString=RoiStringArray[RoiStringIndex]
                        ClickedDicB = {}
                        DicBInt=0
                        print RoiString



                    #####################################################################
                    #This creates an ROI group from the outline of ClickedDicB that is selected using A.
                    #you can select to mirror an ROI accross the mirror axis of the STL.
                    #To create the ROI you use the a button when the mouse is in the area desired.  the path will
                    #be a closed polygon.  The ends of the polygon will automatically be connected.
                    if c is 57:  # c is 9
                        #creating the ROI group
                        try:
                            fwROI = fw.create_group('ROI')
                            print "made ROI"


                        except:
                            fwROI = fw['ROI']


                        #initiallizing stuff
                        ClickFill=[]
                        ClickFillFill=1000
                        CellIdROI=[]
                        uselineorPoly=2


                        #this is the line that we were using first
                        if uselineorPoly==0:
                            #DicBInt is how many points are in the line
                            # we are using a dctionary
                            for ii in range(DicBInt):
                                if ii >= 1:
                                    x1 = numpy.array([ClickedDicB[ii - 1][0], ClickedDicB[ii - 1][1]])
                                    x2 = numpy.array([ClickedDicB[ii][0],ClickedDicB[ii][1]])
                                    #filling up the points.
                                    for ij in range(ClickFillFill+1):
                                        #print x1*((float(ClickFillFill)-float(ij))/float(ClickFillFill))+(x2)*(float(ij)/float(ClickFillFill))
                                        ClickFill.append(x1*((float(ClickFillFill)-float(ij))/float(ClickFillFill))+(x2)*(float(ij)/float(ClickFillFill)))


                            print len(ClickFill)
                            print datOrdAlt,"datOrdAlt"
                            for ii in range(len(ClickFill)):

                                ClickPoint=ClickFill[ii]

                                CamOrgn23, pointvect23 = self.theVectors(datOrdAlt, ClickPoint, cameraMatrix1,
                                                                         distCoeffs1)

                                pSource = CamOrgn23
                                pTarget = CamOrgn23 + Ptarlen * pointvect23


                                pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)
                                if not pointsIntersection:
                                    None
                                else:
                                    # print three
                                    print ii, cellIdsInter[0]
                                    if len(CellIdROI)==0:
                                        CellIdROI.append(cellIdsInter[0])
                                    #elif CellIdROI[len(CellIdROI)-1]!=cellIdsInter[0]:### need to fix this
                                    elif cellIdsInter[0] not in CellIdROI:  ### need to fix this

                                        CellIdROI.append(cellIdsInter[0])

                        #using enclosed loop
                        if uselineorPoly==1:
                            # squidVert
                            print polydata.GetNumberOfPoints(), "number of point"
                            print polydata.GetNumberOfCells(), "number of point"

                            print polydata.GetCellData(), "cells"
                            print polydata.GetCell(0).GetPoints().GetPoint(0)
                            print polydata.GetCell(0).GetPoints().GetPoint(1)
                            print polydata.GetCell(0).GetPoints().GetPoint(2)

                            RotMat0 = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
                            TranMat0 = numpy.zeros(3)

                            pSource = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])

                            # going through all the cells
                            for ig in range(polydata.GetNumberOfCells()):
                                STLcount = 0
                                CellMesh = []
                                # looking at all the verticies in a triangle
                                for vv in range(3):
                                    pTarget = numpy.array(polydata.GetCell(ig).GetPoints().GetPoint(vv))
                                    pointsIntersection, cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)
                                    # looking to see if it intersects the mesh only once.  If it does then it is facing the camera
                                    if len(pointsIntersection) != 1:
                                        STLcount += 1
                                    else:
                                        CellMesh.append(pointsIntersection[0])
                                if STLcount == 0:
                                    makeInt = False
                                    CellMesh = numpy.array(CellMesh)
                                    CellMesh = numpy.transpose(CellMesh)

                                    MeshProjCell = self.ReturnMeshProjection(CellMesh, RotMat0, TranMat0, datOrdAlt,
                                                                             cameraMatrix1,
                                                                             distCoeffs1,
                                                                             width, height, makeInt)

                                    # print MeshProjCell
                                    xbound = numpy.array([MeshProjCell[0][0], MeshProjCell[1][0], MeshProjCell[2][0]])
                                    ybound = numpy.array([MeshProjCell[0][1], MeshProjCell[1][1], MeshProjCell[2][1]])
                                    # print xbound,ybound,"bound"


                                    # here is the list of pixels that are occluded by the triangle
                                    PixelArray = []

                                    # and how much they are occluded.
                                    PixelWeight = []

                                    # here is fancy geometry using ogr

                                    # looking at the intersection of a pixil square and the Mesh triangle

                                    subjectPolygon=ClickedDicB

                                    # this is the triangle.  There are two here for some reason
                                    clipPOlygon = [[MeshProjCell[0][0], MeshProjCell[0][1]],
                                                   [MeshProjCell[1][0], MeshProjCell[1][1]],
                                                   [MeshProjCell[2][0], MeshProjCell[2][1]]]
                                    clipPOlygon = [[MeshProjCell[2][0], MeshProjCell[2][1]],
                                                   [MeshProjCell[1][0], MeshProjCell[1][1]],
                                                   [MeshProjCell[0][0], MeshProjCell[0][1]]]


                                    if 0:#delete when you can
                                        # here we are putting the polygons in the right format
                                        # pixel
                                        wkt1 = "POLYGON (("
                                        wkt1it = 0
                                        for ii in range(DicBInt):
                                            if wkt1it == 0:
                                                wkt1 += str(subjectPolygon[ii][0]) + " " + str(subjectPolygon[ii][1])
                                                wkt1it += 1
                                            else:
                                                wkt1 += " , " + str(subjectPolygon[ii][0]) + " " + str(subjectPolygon[ii][1])
                                        wkt1 += " , " + str(subjectPolygon[0][0]) + " " + str(subjectPolygon[0][1])
                                        wkt1 += "))"

                                        # triangle
                                        wkt2 = "POLYGON (("
                                        wkt1it = 0
                                        for sP in clipPOlygon:
                                            if wkt1it == 0:
                                                wkt2 += str(sP[0]) + " " + str(sP[1])
                                                wkt1it += 1
                                            else:
                                                wkt2 += " , " + str(sP[0]) + " " + str(sP[1])
                                        wkt2 += " , " + str(clipPOlygon[0][0]) + " " + str(clipPOlygon[0][1])
                                        wkt2 += "))"


                                        poly1 = ogr.CreateGeometryFromWkt(wkt1)
                                        poly2 = ogr.CreateGeometryFromWkt(wkt2)

                                        # getting the intersection
                                        intersection = poly1.Intersection(poly2)
                                        areaTriangle = poly2.GetArea()
                                        area = intersection.GetArea()


                                    poly1 = Polygon(subjectPolygon)

                                    poly2 = Polygon(clipPOlygon)
                                    areaTriangle = poly2.area

                                    area = poly1.intersection(poly2).area






                                    # here we choose the pixels that were intersected as well as how much of the pixel was interesected
                                    if area > 0.0000001:
                                        CellIdROI.append(ig)



                        #mirroring
                        if uselineorPoly == 2:# can this be done for both linear and area?
                            roinum=int(RoiString.split("I")[1])
                            roiMirror="ROI"+str(roinum-1)
                            print roiMirror,RoiString
                            for ik in fwROI[roiMirror]["ROI cell list"]:
                                if 1:

                                    CellMeshAve=numpy.zeros(3)
                                    for pixarr in fwSTL[str(int(frame1num))][filenameSTL][str(int(ik))]["CellMesh"]:
                                        CellMeshAve+=numpy.array(pixarr)
                                    CellMeshAve=CellMeshAve/3.0
                                    print CellMeshAve,"inital"
                                    pTarget=numpy.array([-10000,CellMeshAve[1],CellMeshAve[2]])
                                    pSource=numpy.array([0,CellMeshAve[1],CellMeshAve[2]])

                                    pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)
                                    if not pointsIntersection:
                                        None
                                    else:
                                        print pointsIntersection,"pointsintersection"
                                        if len(CellIdROI)==0:
                                            CellIdROI.append(cellIdsInter[0])
                                        elif cellIdsInter[0] not in CellIdROI:  ### need to fix this

                                            CellIdROI.append(cellIdsInter[0])


                        print CellIdROI,"CellIdROI"

                        try:

                            fwROI.create_group(RoiString)
                            fwROI[RoiString].create_dataset("ROI cell list", data=CellIdROI)
                            fwROI[RoiString].create_dataset("CameraInfo", data=datOrdAlt)
                            fwROI[RoiString].create_dataset("LineOut", data=ClickFill)
                        except:

                            del fwROI[RoiString]
                            fwROI.create_group(RoiString)
                            fwROI[RoiString].create_dataset("ROI cell list", data=CellIdROI)
                            fwROI[RoiString].create_dataset("CameraInfo", data=datOrdAlt)
                            fwROI[RoiString].create_dataset("LineOut", data=ClickFill)
                            print "didn't"



                            #####################################################################
                    # saving squid info
                    #####################################################################
                    if c is 56:  # c is 8
                        print "you pressed 8"

                        try:
                            fwSTL = fw.create_group('STL')
                            print "made STL"


                        except:
                            fwSTL = fw['STL']



                        # this is the data in the squid excell file.
                        print SquidExcelDetails
                        if 1:
                            try:
                                fwSTL.create_group(str(int(frame1num)))

                                fwSTL[str(int(frame1num))].create_group(filenameSTL)

                                fwSTL[str(int(frame1num))].create_dataset("SquidExcelDetails", data=SquidExcelDetails)

                                # this is the camera data, the position and attidue given by Pnp and the new algorithm, by pressing f and i
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("datOrdAlt", data=datOrdAlt)
                                # this is the pitch of the camera from the fit above assuming that the squid does not roll
                                # comes from res = minimize(self.AdjustPitchtoMinimizeRoll, 0, method='nelder-mead',options = {'xtol': 1e-8, 'disp': True})
                                # this is a minimization program that adjusts the pitch of the squid frame to minimize the camera roll.

                                #this is the resulting pitch angle
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("PitchTheta", data=resx0)


                                # this is the minimized camera roll, probably close to 0
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("CameraRoll", data=CameraRoll)

                                # this is the squid rotating matrix about the y axis for pitch
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("RotAroundY", data=RotAroundY)

                                #this is the resulting camera pararmaters.
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("datOrdAltAlt", data=datOrdAltAlt)

                                # comes from CameraRoll, RotAroundY, datOrdAltAlt=self.AdjustPitchtoMinimizeRollOut(res.x[0])



                            except:

                                del fwSTL[str(int(frame1num))]
                                fwSTL.create_group(str(int(frame1num)))


                                fwSTL[str(int(frame1num))].create_group(filenameSTL)

                                fwSTL[str(int(frame1num))].create_dataset("SquidExcelDetails", data=SquidExcelDetails)

                                # this is the camera data, the position and attidue given by Pnp and the new algorithm, by pressing f and i
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("datOrdAlt", data=datOrdAlt)
                                # this is the pitch of the camera from the fit above assuming that the squid does not roll
                                # comes from res = minimize(self.AdjustPitchtoMinimizeRoll, 0, method='nelder-mead',options = {'xtol': 1e-8, 'disp': True})
                                # this is a minimization program that adjusts the pitch of the squid frame to minimize the camera roll.

                                # this is the resulting pitch angle
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("PitchTheta", data=resx0)

                                # this is the minimized camera roll, probably close to 0
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("CameraRoll", data=CameraRoll)

                                # this is the squid rotating matrix about the y axis for pitch
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("RotAroundY", data=RotAroundY)

                                # this is the resulting camera pararmaters.
                                fwSTL[str(int(frame1num))][filenameSTL].create_dataset("datOrdAltAlt", data=datOrdAltAlt)

                                # comes from CameraRoll, RotAroundY, datOrdAltAlt=self.AdjustPitchtoMinimizeRollOut(res.x[0])

                                #fwSTL[str(int(frame1num))]["SquidExcelDetails"] = SquidExcelDetails
                                #fwSTL[str(int(frame1num))][filenameSTL]["datOrdAlt"] = datOrdAlt
                                #fwSTL[str(int(frame1num))][filenameSTL]["PitchTheta"] = res.x[0]
                                #fwSTL[str(int(frame1num))][filenameSTL]["CameraRoll"] = CameraRoll
                                #fwSTL[str(int(frame1num))][filenameSTL]["RotAroundY"] = RotAroundY
                                #fwSTL[str(int(frame1num))][filenameSTL]["datOrdAltAlt"] = datOrdAltAlt

                        # squidVert
                        print polydata.GetNumberOfPoints(), "number of point"
                        print polydata.GetNumberOfCells(), "number of point"

                        print polydata.GetCellData(), "cells"
                        print polydata.GetCell(0).GetPoints().GetPoint(0)
                        print polydata.GetCell(0).GetPoints().GetPoint(1)
                        print polydata.GetCell(0).GetPoints().GetPoint(2)

                        RotMat0 = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
                        TranMat0 = numpy.zeros(3)

                        pSource = numpy.array([datOrdAlt[9], datOrdAlt[10], datOrdAlt[11]])



                        #going through all the cells
                        for ig in range(polydata.GetNumberOfCells()):
                            STLcount = 0
                            CellMesh = []
                            #looking at all the verticies in a triangle
                            for vv in range(3):
                                pTarget = numpy.array(polydata.GetCell(ig).GetPoints().GetPoint(vv))
                                pointsIntersection,cellIdsInter = self.VTKcaster(obbTree, pSource, pTarget)
                                #looking to see if it intersects the mesh only once.  If it does then it is facing the camera
                                if len(pointsIntersection) != 1:
                                    STLcount += 1
                                else:
                                    CellMesh.append(pointsIntersection[0])
                            if STLcount == 0:
                                makeInt = False
                                CellMesh = numpy.array(CellMesh)
                                CellMeshT = numpy.transpose(CellMesh)
                                #print CellMesh,"CellMesh"
                                MeshProjCell = self.ReturnMeshProjection(CellMeshT, RotMat0, TranMat0, datOrdAlt,
                                                                         cameraMatrix1,
                                                                         distCoeffs1,
                                                                         width, height, makeInt)

                                # print MeshProjCell
                                xbound = numpy.array([MeshProjCell[0][0], MeshProjCell[1][0], MeshProjCell[2][0]])
                                ybound = numpy.array([MeshProjCell[0][1], MeshProjCell[1][1], MeshProjCell[2][1]])
                                # print xbound,ybound,"bound"


                                # here is the list of pixels that are occluded by the triangle
                                PixelArray = []

                                # and how much they are occluded.
                                PixelWeight = []

                                # here is fancy geometry using ogr

                                # looking at the intersection of a pixil square and the Mesh triangle

                                for ix in range(numpy.floor(xbound.min()), numpy.ceil(xbound.max())):
                                    for iy in range(numpy.floor(ybound.min()), numpy.ceil(ybound.max())):
                                        # this is a square pixel
                                        subjectPolygon = [[float(ix), float(iy)], [float(ix + 1), float(iy)],
                                                          [float(ix + 1), float(iy + 1)], [float(ix), float(iy + 1)]]

                                        # this is the triangle.  There are two here for some reason
                                        clipPOlygon = [[MeshProjCell[0][0], MeshProjCell[0][1]],
                                                       [MeshProjCell[1][0], MeshProjCell[1][1]],
                                                       [MeshProjCell[2][0], MeshProjCell[2][1]]]
                                        clipPOlygon = [[MeshProjCell[2][0], MeshProjCell[2][1]],
                                                       [MeshProjCell[1][0], MeshProjCell[1][1]],
                                                       [MeshProjCell[0][0], MeshProjCell[0][1]]]

                                        if 0: #delete when you can
                                            # here we are putting the polygons in the right format
                                            # pixel
                                            wkt1 = "POLYGON (("
                                            wkt1it = 0
                                            for sP in subjectPolygon:
                                                if wkt1it == 0:
                                                    wkt1 += str(sP[0]) + " " + str(sP[1])
                                                    wkt1it += 1
                                                else:
                                                    wkt1 += " , " + str(sP[0]) + " " + str(sP[1])
                                            wkt1 += " , " + str(subjectPolygon[0][0]) + " " + str(subjectPolygon[0][1])
                                            wkt1 += "))"

                                            # print wkt1
                                            # triangle
                                            wkt2 = "POLYGON (("
                                            wkt1it = 0
                                            for sP in clipPOlygon:
                                                if wkt1it == 0:
                                                    wkt2 += str(sP[0]) + " " + str(sP[1])
                                                    wkt1it += 1
                                                else:
                                                    wkt2 += " , " + str(sP[0]) + " " + str(sP[1])
                                            wkt2 += " , " + str(clipPOlygon[0][0]) + " " + str(clipPOlygon[0][1])
                                            wkt2 += "))"

                                            # print wkt2

                                            poly1 = ogr.CreateGeometryFromWkt(wkt1)
                                            poly2 = ogr.CreateGeometryFromWkt(wkt2)

                                            # getting the intersection
                                            intersection = poly1.Intersection(poly2)
                                            areaTriangle = poly2.GetArea()



                                        poly1 = Polygon(subjectPolygon)

                                        poly2 = Polygon(clipPOlygon)
                                        areaTriangle = poly2.area








                                        try:
                                            # area = intersection.GetArea()
                                            area = poly1.intersection(poly2).area
                                        except:
                                            area=0
                                        # here we choose the pixels that were intersected as well as how much of the pixel was interesected
                                        if area > 0.0000001:
                                            PixelArray.append([ix, iy])
                                            PixelWeight.append(area)

                                # now we are recording it.
                                if 1:
                                    try:
                                        fwSTL[str(int(frame1num))][filenameSTL].create_group(str(int(ig)))

                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))].create_dataset("PixelWeight",
                                                                                                             data=PixelWeight)
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))].create_dataset("PixelArray",
                                                                                                             data=PixelArray)
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))].create_dataset("areaTriangle",
                                                                                                             data=areaTriangle)
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))].create_dataset("clipPOlygon",
                                                                                                             data=clipPOlygon)
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))].create_dataset("CellMesh",
                                                                                                             data=CellMesh)

                                    except:
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))]["PixelWeight"] = PixelWeight
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))]["PixelArray"] = PixelArray
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))]["areaTriangle"] = areaTriangle
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))]["clipPOlygon"] = clipPOlygon
                                        fwSTL[str(int(frame1num))][filenameSTL][str(int(ig))]["CellMesh"] = CellMesh



                        print "Done"
                        RoiStringIndex = 0



if __name__ == "__main__":
    app = animateFromVideo()
    app.watchVidandPicFrames()
