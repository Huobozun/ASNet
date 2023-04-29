import numpy as np
import cv2

size = (346,260)
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
videowrite0 = cv2.VideoWriter('groundtruth.mp4',fourcc,20,size)
img_array=[]
for filename in ['./depthshow/{}.png'.format(i) for i in range(0,1874)]:
    img = cv2.imread(filename)
    img_array.append(img)


for i in range(0,1874):
    text = 'groundtruth'
    loc = (150,240)
    font=cv2.FONT_HERSHEY_COMPLEX
    font_size =0.5
    font_color= (100,200,200)
    font_bold = 1

    frame_texted = img_array[i].copy()
    cv2.putText(frame_texted,text,loc,font,font_size,font_color,font_bold)
    videowrite0.write(frame_texted)

videowrite0.release()



videowrite1 = cv2.VideoWriter('image.mp4',fourcc,31,size)
img_array=[]
for filename in ['./image/{}.png'.format(i) for i in range(0,2944)]:
    img = cv2.imread(filename)
    img_array.append(img)


for i in range(0,2944):
    text = 'image'
    loc = (150,240)
    font=cv2.FONT_HERSHEY_COMPLEX
    font_size =0.5
    font_color= (100,200,200)
    font_bold = 1

    frame_texted = img_array[i].copy()
    cv2.putText(frame_texted,text,loc,font,font_size,font_color,font_bold)
    videowrite1.write(frame_texted)

videowrite1.release()



videowrite2 = cv2.VideoWriter('prediction.mp4',fourcc,20,size)
img_array=[]
for filename in ['./predictionshow/{}.png'.format(i) for i in range(0,1874)]:
    img = cv2.imread(filename)
    img_array.append(img)


for i in range(0,1874):
    text = 'prediction'
    loc = (150,240)
    font=cv2.FONT_HERSHEY_COMPLEX
    font_size =0.5
    font_color= (100,200,200)
    font_bold = 1

    frame_texted = img_array[i].copy()
    cv2.putText(frame_texted,text,loc,font,font_size,font_color,font_bold)
    videowrite2.write(frame_texted)


videowrite2.release()


videowrite3 = cv2.VideoWriter('event.mp4',fourcc,20,size)
img_array=[]
for filename in ['./frameleft/{}.png'.format(i) for i in range(0,1874)]:
    img = cv2.imread(filename)
    img_array.append(img)


for i in range(0,1874):
    text = 'event frame'
    loc = (150,240)
    font=cv2.FONT_HERSHEY_COMPLEX
    font_size =0.5
    font_color= (100,200,200)
    font_bold = 1

    frame_texted = img_array[i].copy()
    cv2.putText(frame_texted,text,loc,font,font_size,font_color,font_bold)
    videowrite3.write(frame_texted)

videowrite3.release()

print('end')




