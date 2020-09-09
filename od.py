#!/usr/bin/python
# -*- coding: utf-8 -*-
from imutils.video import VideoStream
import numpy as np
import sys
import time
import subprocess
import datetime
import imutils
import time
import cv2
import socket




CLASSES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    ]
COLORS = np.random.uniform(0, 0xFF, size=(len(CLASSES), 3))
resultClasses=[]
# Model Files

prototxt = './ssd_MobileNet.txt'
model = './ssd_model.caffemodel'

# Confidence score
interface=sys.argv[2].rstrip('\n')
conf = 0.8
fileToBeSent=sys.argv[3].rstrip('\n')
# Loading model from file

print ('* Loading Model.......')
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print ('* Starting Stream......')

# vs = VideoStream(src=0).start()

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Grabbing Videostream

while True:
    try:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

            # Passing into the NN

        net.setInput(blob)
        detections = net.forward()

            # loop over the detections

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

                # Filter out the low confidence prediction

            if confidence > conf:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                         # Bounding Box
                resultClasses.append(CLASSES[idx])
                label = '{}: {:.2f}%'.format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = (startY - 15 if startY - 15 > 15 else startY + 15)
                cv2.putText(
                                     frame,
                                     label,
                                     (startX, y),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5,
                                     COLORS[idx],
                                     2,
                                )

                 # Output Bounding Box
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        time.sleep(20)
    
                 #End Capturing When Ctrl-C is pressed

       
    except KeyboardInterrupt:
            print ("Bye")
            print ("Caught ^C \n")
            timenow = datetime.datetime.now()
		
            device_ip=subprocess.check_output("ifconfig " + interface + " | grep 'inet'| cut -d':' -f2", shell = True).strip()    
            broadcast_msg="%s-Detected on %s !!! at %s "%(resultClasses,device_ip,timenow)
            print(broadcast_msg)
            f=open(fileToBeSent,"w+")
            f.write(broadcast_msg)
            f.close()
            remote_machine=sys.argv[1]
            remote_machine.rstrip('\n')
            print (remote_machine)
            remotePath="/home/pi/Desktop"
            remote_machine_path="%s:%s"%(remote_machine,remotePath)
            command=["scp",fileToBeSent, remote_machine_path]
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            print (p.communicate())
            sys.exit (0) 



   
cv2.destroyAllWindows()
vs.stop()

