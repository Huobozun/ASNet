# import cv2
# import numpy as np

# names = ['/home/tlab4/zjg/data/zdatasets/frameindoor_mvsec3_video/groundtruth.mp4',
# '/home/tlab4/zjg/data/zdatasets/frameindoor_mvsec3_video/event.mp4',
# '/home/tlab4/zjg/data/zdatasets/frameindoor_mvsec3_video/image.mp4',
# '/home/tlab4/zjg/data/zdatasets/frameindoor_mvsec3_video/prediction.mp4']

# window_titles = ['camera_1', 'camera_2', 'lidar_1', 'fusion']


# cap = [cv2.VideoCapture(i) for i in names]

# frames = [None] * len(names)
# ret = [None] * len(names)
# width = 480
# height = 270
# while True:

#     for i,c in enumerate(cap):
#         if c is not None:
#             ret[i], frames[i] = c.read()


#     frameLeftUp = cv2.resize(frames[0], (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
#     frameRightUp = cv2.resize(frames[1], (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
#     frameLeftDown = cv2.resize(frames[2], (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

#     frameRightDown= cv2.resize(frames[3], (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
 
#     frameUp = np.hstack((frameLeftUp, frameRightUp))
#     frameDown = np.hstack((frameLeftDown, frameRightDown))
#     frame = np.vstack((frameUp, frameDown))
#     cv2.imshow("Sensor fusion", frame)



#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break


# for c in cap:
#     if c is not None:
#         c.release()

# cv2.destroyAllWindows()


from  moviepy.editor import *

clips = [VideoFileClip('./image.mp4'),VideoFileClip('./event.mp4')]
clips1 = [VideoFileClip('./groundtruth.mp4'),VideoFileClip('./prediction.mp4')]
video = clips_array([clips,clips1])
video.write_videofile('./all.mp4')

