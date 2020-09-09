from imutils.video import VideoStream
import numpy as np
import subprocess
import datetime
import imutils
import sys
import time
import cv2

remote_machine=sys.argv[1]
interface=sys.argv[2].rstrip('\n')
msgFile=sys.argv[3].rstrip('\n')
detectList=[]

def sendNotification(interface,remote_machine,msg):
	timenow = datetime.datetime.now()
	device_ip=subprocess.check_output("ifconfig " + interface + " | grep 'inet'| cut -d':' -f2", shell = True).strip()    
	broadcast_msg="%s-Person Detected on %s !!! at %s "%(msg,device_ip,timenow)
	print(broadcast_msg)
	f=open(msgFile,"w+")
	f.write(broadcast_msg)
	f.close()	
	remote_machine.rstrip('\n')
	print (remote_machine)
	remotePath="/home/pi/Desktop"
	remote_machine_path="%s:%s"%(remote_machine,remotePath)
	command=["scp","msgFile.txt", remote_machine_path]
	p = subprocess.Popen(command, stdout=subprocess.PIPE)
	print (p.communicate())
	sys.exit (0) 

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err





CLASSES = ["person"]


#Model Files
prototxt='./ssd_MobileNet.txt'
model='./ssd_model.caffemodel'

#Confidence score
conf=0.8


# Loading model from file
print("* Loading Model.......")
net = cv2.dnn.readNetFromCaffe(prototxt, model)



print("* Starting Stream......")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
prev_frames=[]
diff=0

# Grabbing Videostream
while True:
	frame = vs.read()
	

	frame = imutils.resize(frame, width=400)
	prev_frames.append(frame)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	
	#Passing into the NN
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		#Filter out the low confidence prediction
		if confidence > conf:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Bounding Box and Display Percenteage Detected 
			label = "{:.2f}%".format(confidence * 100)

			
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# Output Bounding Box
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if(len(prev_frames)>2):
		diff=mse(prev_frames[-1],prev_frames[-2])

	if(len(prev_frames)>100):

		prev_frames=[]

	if(diff>2000):
		print (label) #print the label to terminal
		detectList.append(label)
		if len(detectList) > 15:
			sendNotification(interface,remote_machine,detectList)
			
	# Quit if 'q' pressed
	if key == ord("q"):
		break




cv2.destroyAllWindows()
vs.stop()
