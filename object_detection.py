from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


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


# Grabbing Videostream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
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

			# Bounding Box
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


	# Output Bounding Box
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
        print(CLASSES[idx])
	# Quit if 'q' pressed
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
