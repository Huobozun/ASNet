import aedat
import numpy as np
import cv2
import h5py
import skimage.morphology as morpho
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import os

import random
import numpy as np
from PIL import Image
import torch


from datasets.utils import mvsecLoadRectificationMaps, mvsecRectifyEvents
from datasets.indices import *
from datasets.zdata_framedeal import CatchV, EventFrameIterator


def dataprepare(dataset,datapath):
    print("==========================\npreparing\n==========================")
    
    if(dataset =='mvsec'):
        
    


        data = h5py.File(datapath+'/indoor_flying1_data.hdf5')
        print(data['davis']['left'].keys())
        eve0=np.array(data['davis']['left']['events'])
        eve1=np.array(data['davis']['right']['events'])

        Lx_path = datapath+'/indoor_flying_left_x_map.txt'
        Ly_path = datapath+'/indoor_flying_left_y_map.txt'
        Rx_path = datapath+'/indoor_flying_right_x_map.txt'
        Ry_path = datapath+'/indoor_flying_right_y_map.txt'
        Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
        rect_Levents = np.array(mvsecRectifyEvents(eve0, Lx_map, Ly_map)) 
        rect_Revents = np.array(mvsecRectifyEvents(eve1, Rx_map, Ry_map))
        pointsv = 20
        timeunit = 1
        timeslot = 0.1
        timefre = 0.05
        shape = [260,346]
        eventshape = [2,0,1,3]
        Nsmall = 16
        Nbig = 64


        #[pointsv:] is to ensure counting starts from the 20th and increases by 20 each time.
        Vmax,Vmin,Vmean,Vmedian,sensitivity=CatchV(rect_Levents[pointsv:],timeslot,timefre,eventshape[0],timeunit,pointsv)
        print(Vmax,Vmin,Vmean,Vmedian,sensitivity)


        Vmax2,Vmin2,Vmean2,Vmedian2,sensitivity2=CatchV(rect_Revents[pointsv:],timeslot,timefre,eventshape[0],timeunit,pointsv)
        print(Vmax2,Vmin2,Vmean2,Vmedian2,sensitivity2)


        with open('./datasets/prepare.txt','w') as file:
            file.write('left: '+str(sensitivity)+' right: '+str(sensitivity2))
    else:


        decoder=aedat.Decoder(datapath)
        #initial time
        for packet in decoder:
            if(packet['stream_id']==0):  #left events
                if'events'in packet:
                    
                    timestart=list(packet['events'])[0][0]+1000000
                    break
        #Obtain sampled speed
        Levents = np.array([])
        Revents = np.array([])
        ii1=0
        ii2=0
        i0=0
        for packet in decoder:
            if i0>=0:
                if(packet['stream_id']==0):  #left events
                    if'events'in packet:
                        if(ii1==0):
                            Levents=packet['events']
                            ii1=1
                        else:

                            
                            Levents = np.append(Levents,packet['events'])

                if(packet['stream_id']==1):  #right events
                    if'events'in packet:
                        if(ii2==0):
                            Revents=packet['events']
                            ii2=1
                        else:
                            
                            Revents = np.append(Revents,packet['events'])
            i0+=1
            if i0==100:
                break
        #Set initial parameters
        timeunit=0.000001#Unit of time relative to seconds
        timestart=timestart*timeunit#The conversion unit is unified as seconds
        pointsv = 20#Sampling speed of data speed measurement
        timeslot = 0.05#The time slot involved in the stacks
        timefre = 0.05#Generate frame time interval
        timestamp=timestart#The time stamp when the frame is generated for the first time
        shape = [480,640]#h*w
        eventshape = [0,1,2,3]#Event Data Format
        Nsmall = 16
        Nbig = 64



        #Data speed measurement
        Vmax,Vmin,Vmean,Vmedian,sensitivity=CatchV(list(Levents)[pointsv:],timeslot,timefre,eventshape[0],timeunit, pointsv)
        print(Vmax,Vmin,Vmean,Vmedian,sensitivity)


        Vmax2,Vmin2,Vmean2,Vmedian2,sensitivity2=CatchV(list(Revents)[pointsv:],timeslot,timefre,eventshape[0],timeunit, pointsv)
        print(Vmax2,Vmin2,Vmean2,Vmedian2,sensitivity2)


        with open('./datasets/prepare4.txt','w') as file:
            file.write('left: '+str(sensitivity)+' right: '+str(sensitivity2))