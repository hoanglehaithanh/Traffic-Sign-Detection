from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# define the dictionary of signs segments so we can identify
# each signs on the image
SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'turnRight', # turnRight
        (0, 0, 1, 1): 'turnLeft', # turnLeft
        (0, 1, 0, 0): 'moveStraight', # moveStraight
        (1, 0, 1, 1): 'turnBack', # turnBack
}

camera = cv2.VideoCapture(0)

def defineTrafficSign(image):

        # pre-process the image by resizing it, converting it to
        # graycale, blurring it, and computing an edge map
        image = imutils.resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)

        # find contours in the edge map, then sort them by their
        # size in descending order
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        # The is_cv2() and is_cv3() are simple functions that can be used to 
        # automatically determine the OpenCV version of the current environment
        # cnts[0] or cnts[1] hold contours
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None

        # loop over the contours
        for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if the contour has four vertices, then we have found
                # the thermostat display
                if len(approx) == 4:
                        displayCnt = approx
                        break

        # extract the sign borders, apply a perspective transform
        # to it
        # A common task in computer vision and image processing is to perform 
        # a 4-point perspective transform of a ROI in an image and obtain a top-down, "birds eye view" of the ROI
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        output = four_point_transform(image, displayCnt.reshape(4, 2))

        # draw a red square on the image
        cv2.drawContours(image, [displayCnt], -1, (0, 0, 255), 5)

        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        # cv2.THRESH_OTSU. it automatically calculates a threshold value from image histogram 
        # for a bimodal image
        thresh = cv2.threshold(warped, 0, 255, 
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


        # (roiH, roiW) = roi.shape
        #subHeight = thresh.shape[0]/10
        #subWidth = thresh.shape[1]/10
        (subHeight, subWidth) = np.divide(thresh.shape, 10)
        subHeight = int(subHeight)
        subWidth = int(subWidth)

        # mark the ROIs borders on the image
        cv2.rectangle(output, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
        cv2.rectangle(output, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
        cv2.rectangle(output, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
        cv2.rectangle(output, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

        # substract 4 ROI of the sign thresh image
        leftBlock = thresh[4*subHeight:9*subHeight, subWidth:3*subWidth]
        centerBlock = thresh[4*subHeight:9*subHeight, 4*subWidth:6*subWidth]
        rightBlock = thresh[4*subHeight:9*subHeight, 7*subWidth:9*subWidth]
        topBlock = thresh[2*subHeight:4*subHeight, 3*subWidth:7*subWidth]

        # we now track the fraction of each ROI. (sum of active pixels)/(total number of pixels)
        leftFraction = np.sum(leftBlock)/(leftBlock.shape[0]*leftBlock.shape[1])
        centerFraction = np.sum(centerBlock)/(centerBlock.shape[0]*centerBlock.shape[1])
        rightFraction = np.sum(rightBlock)/(rightBlock.shape[0]*rightBlock.shape[1])
        topFraction = np.sum(topBlock)/(topBlock.shape[0]*topBlock.shape[1])

        segments = (leftFraction, centerFraction, rightFraction, topFraction)
        segments = tuple(1 if segment > 230 else 0 for segment in segments)

        

        if segments in SIGNS_LOOKUP:
            # show original image
            cv2.imshow("output", output)
            return SIGNS_LOOKUP[segments]
        else:
            return None


        
while True:
        (grabbed, frame) = camera.read()
        defineTrafficSign(frame)
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            print("Stop programm and close all windows")
            break