

import numpy as np
import matplotlib.pyplot as plt
import h5py


####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

#  This program is taking the ORB txt data and putting it into an H5.

# if you want the full information use Moredata=True
# this stores it with the name _H

#if you want the camera information only use use Moredata=True
#this stores it with the name _HCO

#Next if you want to convert this _H to being listed from the map points perspective go to OrganizeandInvestigateOrbMap


####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################


#filename = 'F:/belize_nov_trip/maps/example_video171109152339.txt'
#filewrite = 'F:/belize_nov_trip/maps/example_video171109152339_H.h5'

filenameinsert="metadata_tom_351_190712"
filename = '/home/parrish/ORB_SLAM2/Data/'+filenameinsert+'.txt'
filewrite = '/home/parrish/ORB_SLAM2/Data/'+filenameinsert+'_HCO.h5'
#filewrite = '/home/parrish/ORB_SLAM2/Data/'+filenameinsert+'_H.h5'

Moredata=False
#filename = '/media/parrish/Blue/Australia/maps/example_video171109152339.txt'
#filewrite = '/media/parrish/Blue/Australia/maps/example_video171109152339_H.h5'
#filename = "/media/parrish/Blue/jayStuff/mybox-selected/example_Cam_1_T3b.txt"
#filewrite = "/media/parrish/Blue/jayStuff/mybox-selected/example_Cam_1_T3b_H.h5"

#filename = '/media/parrish/Blue/belize_nov_trip/maps/example_tom_31.txt'
#filewrite = '/media/parrish/Blue/belize_nov_trip/maps/example_tom_31_H.h5'

#filename = '/media/parrish/Blue/belize_nov_trip/maps/example_v17110909_1.txt'
#filewrite = '/media/parrish/Blue/belize_nov_trip/maps/example_v17110909_1_H1.h5'




print "PC = open(filename)"
PC = open(filename)

Header=PC.readline()
print Header
CameraCalibration=PC.readline()
print CameraCalibration
line=CameraCalibration

hf = h5py.File(filewrite, 'w')

mapdataheader=18# was 13
data=np.zeros(12)
mapdata=np.zeros((1,mapdataheader))
mapdataInsert=np.zeros((1,mapdataheader))
kIt=0
kStart=0

g1 = hf.create_group('header')



datas=[""]*12



for i in range(len(Header.split("|"))):
    if i > mapdataheader:
        datas[i - (mapdataheader+1)] = Header.split("|")[i]


g1.create_dataset('camera header',data=datas)
print datas

datas=[""]*mapdataheader




for i in range(len(Header.split("|"))):
    if i > 0 and i < mapdataheader+1:
        datas[i - 1] = Header.split("|")[i]



g1.create_dataset('map header',data=datas)
print datas


datas=[""]*9

for i in range(len(CameraCalibration.split("|"))-1):
    print np.floor(i/2), float(i)/2
    if float(i)/2==np.floor(i/2):
        datas[i/2] = CameraCalibration.split("|")[i]



g1.create_dataset('CameraCalibrationHeader',data=datas)
print datas



data=np.zeros(9)

for i in range(len(CameraCalibration.split("|"))):
    if float(i)/2!=np.floor(i/2):
        data[i/2] = float(CameraCalibration.split("|")[i])



g1.create_dataset('CameraCalibrationValue',data=data)
print data








data=np.zeros(12)




line=PC.readline()
lineFormer=line
while line:

    #if k>=kStart:
    if 1:

        #print line
        if line.split("|")[1]=="":
            print 'F'+line.split("|")[0]
            g1 = hf.create_group('F'+line.split("|")[0])
            for i in range(len(Header.split("|"))+1):
                if i>mapdataheader+1:
                    data[i-(mapdataheader+2)]=float(line.split("|")[i])

            #g2 = g1.create_group("CameraPos")
            g1.create_dataset("CameraPos", data=data)


            #print data

            data=np.zeros(12)


        if Moredata:
            #We
            if lineFormer.split("|")[1]!="" and line.split("|")[1]!="":
            #if PPlines[k].split("|")[1]!="" and PPlines[k+1].split("|")[1]!="":
                for i in range(len(Header.split("|"))):
                    if i>0 and i<mapdataheader+1:

                        mapdataInsert[0][i-1]=float(lineFormer.split("|")[i])


                if kIt==0:
                    mapdata[0][:] = mapdataInsert[0][:]

                else:

                    mapdata=np.append(mapdata,mapdataInsert,axis=0)

                kIt+=1



            #We  looking for the end.
            if lineFormer.split("|")[1]!="" and line.split("|")[1]=="" :

                #print PPlines[k]
                if 1:
                    for i in range(len(Header.split("|"))):
                        if i > 0 and i < mapdataheader+1:
                            mapdataInsert[0][i - 1] = float(lineFormer.split("|")[i])

                    if kIt == 0:
                        mapdata[0][:] = mapdataInsert[0][:]

                    else:

                        mapdata=np.append(mapdata,mapdataInsert,axis=0)


                    #print "made it here"
                    #print mapdata
                    #g2 = g1.create_group("MapData")
                    g1.create_dataset("MapData", data=mapdata) # non compressed
        #            g1.create_dataset("MapData", data=mapdata,compression='gzip',compression_opts=9) #compressed

                    kIt=0
                    mapdata = np.zeros((1, mapdataheader))
                    mapdataInsert = np.zeros((1, mapdataheader))

        lineFormer=line
        line=PC.readline()

