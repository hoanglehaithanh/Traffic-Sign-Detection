# import the necessary packages
import numpy as np
import argparse
import cv2
import time

cap = cv2.VideoCapture('./Videos/MVI_1049.avi') # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices

#Set Width and Height 
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.
### Preprocess image
def constrastLimit(image):
	img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	channels = cv2.split(img_hist_equalized)
	channels[0] = cv2.equalizeHist(channels[0])
	img_hist_equalized = cv2.merge(channels)
	img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
	return img_hist_equalized

def LaplacianOfGaussian(image):
	LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
	gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
	LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
	LoG_image = cv2.convertScaleAbs(LoG_image)
	return LoG_image
	
def binarization(image):
	thresh = cv2.threshold(image,96,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	return thresh

def preprocess_image(image):
	image = constrastLimit(image)
	image = LaplacianOfGaussian(image)
	image = binarization(image)
	return image

# Find Signs
def removeSmallComponents(image, threshold):
	#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
	sizes = stats[1:, -1]; nb_components = nb_components - 1

	img2 = np.zeros((output.shape),dtype = np.uint8)
	#for every component in the image, you keep it only if it's above threshold
	for i in range(0, nb_components):
	    if sizes[i] >= threshold:
	        img2[output == i + 1] = 255
	return img2

count =  1
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if not ret:
		print('Done')
		break
	# load the image, clone it for output, and then convert it to grayscale
			
	output = frame.copy()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
	#gray = cv2.GaussianBlur(gray,(3,3),0);
	#gray = cv2.medianBlur(gray,3)
	
	# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
	#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,3.5)
	
	#kernel = np.ones((3,3),np.uint8)
	#gray = cv2.erode(gray,kernel,iterations = 1)
	# gray = erosion
	
	#gray = cv2.dilate(gray,kernel,iterations = 1)
	# gray = dilation

	# get the size of the final image
	# img_size = gray.shape
	# print img_size
	
	# detect circles in the image
	frame_1 = preprocess_image(frame)
	image = removeSmallComponents(frame_1, 300)
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=45, minRadius=0, maxRadius=0)
	# print circles
	print(count)
	count += 1
	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle in the image
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), 3)
			#time.sleep(0.5)
			#print("Column Number: ")
			#print(x)
			#print("Row Number: ")
			#print(y)
			#print("Radius is: ")
			#print(r)
	# Display the resulting frame
	cv2.imshow('gray',image)
	cv2.imshow('frame',output)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


