import aedat
import numpy as np
import cv2
import h5py
import skimage.morphology as morpho
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd





def CatchV(events,timeslot,timefre,tbit,timeunit, pointsv):
	Vmax = 0
	Vmin = 100000000
	Vmean = 0
	Vmi = 0
	Vs = np.array([])
	timeslot = timeslot/timeunit

	eventcount=np.array([0])

	timeleft = [events[0][tbit]]
	
	ievent = pointsv
	while ievent<=len(events):
		

		if(events[ievent][tbit] >= timeleft[0]+timeslot):#Fragments Exceeding the Minimum Recording Time
			v=eventcount[0]/(timeslot*timeunit)
			
			Vs = np.append(Vs,v)
			
			Vmi+=1
			Vmean=(Vmean*(Vmi-1)+v)/Vmi

			timeleft=timeleft[1:]
			eventcount=eventcount[1:]        
		if(events[ievent][tbit] >= timeleft[-1]+timefre):#A Time Gap That Needs a New Record
			timeleft.append(events[ievent][tbit])
			eventcount=np.append(eventcount,0)


		eventcount=eventcount+np.ones(len(eventcount))*pointsv
		ievent+=pointsv
	sensitivity = Vmean#1s points

	return np.max(Vs), np.min(Vs), Vmean, np.median(Vs),sensitivity

class EventFrameIterator:
	def __init__(self, timestamp, timeslot, timefre, timeunit, shape, Vmedian, event_sensitivity, eventshape, Nsmall, Nbig, dirr):
		self.eventstack = [[]]
		self.eventstackcount = np.array([0])
		self.eventstacktimestamp = np.array([])
		self.timeunit = timeunit
		self.timeslot = timeslot/self.timeunit
		self.timefre = timefre/self.timeunit
		self.timestamp = timestamp/self.timeunit#Calculate the timestamp when the frame should be generated for the first time, and so on
		
		
		self.shape = shape
		self.tbit = eventshape[0]
		self.xbit = eventshape[1]
		self.ybit = eventshape[2]
		self.pbit = eventshape[3]
		self.dir = dirr
		self.iframe=0 #Record how many frames have been generated
		self.Vmedian = Vmedian
		self.Nsmall = Nsmall
		self.Nbig = Nbig
		self.event_sensitivity = int(event_sensitivity/((self.shape[0]*self.shape[1])/(260*346))) #Accounting for the actual sensitivity change in image size
		
		self.maxpoints = self.Vmedian*(self.timeslot*self.timeunit)/(Nsmall)
		self.Maxmaxpoints = self.maxpoints*5
		self.Minmaxpoints = self.maxpoints/5

		self.frames = []
		self.framelist = np.array([])
    


	def _addevent_(self, fetch_size, newevents):


		
		self.eventstackcount[-1]+=fetch_size
		
		self.eventstack[-1]+=newevents


		if len(self.eventstack)>=self.Nsmall and self._timedetect_():
			self._formframe_()

		self._detect_()
	
		fetch_size=self.maxpoints/100#The speed of reading into points every time

		if(self.event_sensitivity > 1):
			fetch_slot = fetch_size*(self.event_sensitivity-1)
		else:
			fetch_slot = 0

		self.eventstackcount[-1]+=fetch_slot



		return fetch_size, fetch_slot
    

	def _detect_(self):
	

		if len(self.eventstack[-1])>=int(self.maxpoints):
			self.eventstacktimestamp=np.append(self.eventstacktimestamp,self.eventstack[-1][-1][self.tbit])
			self.eventstackcount=np.append(self.eventstackcount,0)
			self.eventstack.append([])
			
			if(len(self.eventstackcount)>=self.Nbig):
				#Adjust the size of maxpoints
				Vlast=np.sum(self.eventstackcount[-1*int(self.Nsmall/2)-1:])/(self.eventstacktimestamp[-1]-self.eventstacktimestamp[-1*int(self.Nsmall/2)-1])
				Vbig=np.sum(self.eventstackcount)/(self.eventstacktimestamp[-1]-self.eventstacktimestamp[0])



				if(Vlast<self.Vmedian/5):#The upper and lower five times as the limit, far less than the median 
					self.maxpoints=self.maxpoints*(1.0+0.5*np.log10(self.Vmedian/Vlast))
		
				elif(Vlast>self.Vmedian*5):#The upper and lower five times as the limit, far greater than the median
					self.maxpoints=self.maxpoints*(1.0-0.5*np.log10(Vlast/self.Vmedian))
		

				elif(Vlast>Vbig):#Accelerate in the range around the median
					#Determine whether it is caused by texture problems
					if(self._is_texture_complex(self.eventstack[-2],self.eventstack[-3])):
						self.maxpoints=self.maxpoints*(1.0-0.1*np.log10(Vlast/Vbig))
					else:
						pass
			

				elif(Vlast<Vbig):#Decelerate in the range around the median
					#Determine whether it is caused by texture problems
					if(self._is_texture_complex(self.eventstack[-3],self.eventstack[-2])):
						self.maxpoints=self.maxpoints*(1.0+0.1*np.log10(Vbig/Vlast))
					else:
						pass
						
				
				else:pass


				if(self.maxpoints>self.Maxmaxpoints):
					self.maxpoints = self.Maxmaxpoints
				elif(self.maxpoints<self.Minmaxpoints):
					self.maxpoints = self.Minmaxpoints
				else:
					pass


		
	

				

			if len(self.eventstack)>=self.Nsmall+1:
				self.eventstack = self.eventstack[1:]

			if len(self.eventstackcount)>=self.Nbig+1:
				self.eventstackcount = self.eventstackcount[1:] 
				self.eventstacktimestamp = self.eventstacktimestamp[1:]
    

	def _timedetect_(self):

		if self.eventstack[-1][-1][self.tbit]>=self.timestamp:#Check whether the specified timestamp is reached, and if so, generate a frame
			self.timestamp=self.timestamp+self.timefre
			return True
		else:
			return False



	def _formframe_(self):
		#Multiple stacks forming three layers of frame
		frame1 = np.ones(self.shape)*128
		frame2 = np.ones(self.shape)*128
		frame3 = np.ones(self.shape)*128
		for istack in range(-1,self.Nsmall*(-1)-1,-1):
			weight = int(self.Nsmall+1+istack)
			for ievent in self.eventstack[istack]:

				frame1[int(ievent[self.ybit]), int(ievent[self.xbit])] += 1*int(weight) if ievent[self.pbit]==1.0 else -1*int(weight)
				frame2[int(ievent[self.ybit]), int(ievent[self.xbit])] += 1*int(weight) if ievent[self.pbit]==1.0 else  0*int(weight)
				frame3[int(ievent[self.ybit]), int(ievent[self.xbit])] += 0*int(weight) if ievent[self.pbit]==1.0 else -1*int(weight)


		#The three layers are merged and deposited
		

		frameall=np.stack((frame1,frame2,frame3),axis=2)

		

		self.frames.append(frameall)
		self.framelist = np.append(self.framelist, '{}.jpg'.format(str(self.iframe)))

		self.iframe+=1
	
	def _is_texture_complex(self,events1,events2):
		frame1 = np.zeros(self.shape)
		frame2 = np.zeros(self.shape)
	
		for ievent in events1:
			frame1[int(ievent[self.ybit]), int(ievent[self.xbit])] += 1 if ievent[self.pbit]==1.0 else -1
		for ievent in events2:
			frame2[int(ievent[self.ybit]), int(ievent[self.xbit])] += 1 if ievent[self.pbit]==1.0 else -1

		

		if(np.sum(np.abs(frame1))>np.sum(np.abs(frame2))):#The average range becomes larger, indicating that it is not a texture problem, but a speed problem
			return True
		else:
			return False
 		

