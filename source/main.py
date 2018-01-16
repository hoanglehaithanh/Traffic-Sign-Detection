import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils




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

def findContour(image):
	#find contours in the thresholded image
	cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE	)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	return cnts

def contourIsSign(perimeter, centroid, threshold):
	#  perimeter, centroid, threshold
	# # Compute signature of contour
	result=[]
	for p in perimeter:
		p = p[0]
		distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
		result.append(distance)
	max_value = max(result)
	signature = [float(dist) / max_value for dist in result ]
	# Check signature of contour.
	temp = sum((1 - s) for s in signature)
	temp = temp / len(signature)
	if temp < threshold: # is  the sign
		return True, max_value
	else: 				# is not the sign
		return False, max_value

#crop sign 
def cropContour(image, center, max_distance):
	width = image.shape[1]
	height = image.shape[0]
	top = max([int(center[0] - max_distance), 0])
	bottom = min([int(center[0] + max_distance + 1), width-1])
	left = max([int(center[1] - max_distance), 0])
	right = min([int(center[1] + max_distance+1), height-1])
	return image[left:right, top:bottom]

def findSigns(image, contours, threshold):
	signs = []
	coordinates = []
	for c in contours:
		# compute the center of the contour
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
		if is_sign:
			sign = cropContour(image, [cX, cY], max_distance)
			signs.append(sign)
			coordinates.append([[cX, cY], max_distance])
        
	return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle):
	binary_image = preprocess_image(image)
	#cv2.imwrite('A.png', binary_image)
	binary_image = removeSmallComponents(binary_image, min_size_components)
	#cv2.imwrite('B.png', binary_image)
	cv2.imshow('BINARY IMAGE', binary_image)
	contours = findContour(binary_image)
	signs, coordinates = findSigns(image, contours, similitary_contour_with_circle)

	for obj in coordinates:
		center, max_distance = obj
		width = image.shape[1]
		height = image.shape[0]
		top = max([int(center[0] - max_distance), 0])
		bottom = min([int(center[0] + max_distance + 1), width-1])
		left = max([int(center[1] - max_distance), 0])
		right = min([int(center[1] + max_distance+1), height-1])		
		cv2.rectangle(image, (top, left), (bottom, right), (0, 255, 0), 2)
	cv2.imshow('Result', image)
	return signs, image

def showsigns(signs, count):
	c = 0
	for s in signs:
		cv2.imwrite(str(count)+'_'+str(c)+'.png', s)
		c = c + 1

def main(file_name):
	vidcap = cv2.VideoCapture(file_name)

	fps = vidcap.get(cv2.CAP_PROP_FPS)
	width = vidcap.get(3)  # float
	height = vidcap.get(4) # float

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

	count = 0
	success = True
	min_size_components = 350	          # parameter
	similitary_contour_with_circle = 0.55   # parameter
	count = 0
	while True:
		success,frame = vidcap.read()
		print(count)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		if not success:
			print("FINISHED")
			break
		count = count + 1
		signs, image = localization(frame, min_size_components, similitary_contour_with_circle)
		out.write(image)
		#showsigns(signs, count)
		#if count == 2000:                    
			# signs = localization(image, min_size_components, similitary_contour_with_circle)
			# showsigns([image], count)
			# showsigns(signs, count)
			#detectSignWithColor(image,'BLUE')
			#break
		#stop = input('Continue? {}'.format(count))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	print("Finish {} frames".format(count))
	return 

main('./Videos/MVI_1049.avi')