# object-and-person-detection
detect object and person using raspberry pi or similar device using raspbian, ubuntu or debian OS based system

Person Detection Using Raspberry Pi

#Command to run the code and launch picamera frame

#usage sudo python3 person_detection.py <picamera username@ipaddress>  <interface>  <filename> 

#this file will be sent to other Raspberry Pi at /home/pi/Desktop (hard coded) location 

sudo python3 person_detection.py pi@192.168.1.2 wlan0 sendNotification.txt 
