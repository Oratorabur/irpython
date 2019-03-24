import sys
import os
import time
import colorsys
import numpy as np
import cv2
import datetime

from PIL import Image

sys.path.insert(0, "./build/lib.linux-armv7l-3.5")

import MLX90640 as mlx

img = Image.new( 'L', (24,32), "black")

def irCounter():        
        mlx.setup(8) #set frame rate of MLX90640
        
        f = mlx.get_frame()

        mlx.cleanup()

        # get max and min temps from sensor
        v_min = min(f)
        v_max = max(f)

        # Console output for testing
        textTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # get timestamp
        print(textTime)
        print(min(f))
        print(max(f))
        print("")
    
        for x in range(24):
            row = []
            for y in range(32):
                val = f[32 * (23-x) + y]
                row.append(val)
                img.putpixel((x, y), (int(val)))
 
        # convert raw temp data to numpy array
        imgIR = np.array(img)

        ## Threshold the -40C to 300 C temps to a more human range
        # Sensor seems to read a bit cold, calibrate in final setting
        rangeMin = 6 # low threshold temp in C
        rangeMax = 20 # high threshold temp in C


        # Apply thresholds based on min and max ranges
        depth_scale_factor = 255.0 / (rangeMax-rangeMin)
        depth_scale_beta_factor = -rangeMin*255.0/(rangeMax-rangeMin)

        depth_uint8 = imgIR*depth_scale_factor+depth_scale_beta_factor
        depth_uint8[depth_uint8>255] = 255
        depth_uint8[depth_uint8<0] = 0
        depth_uint8 = depth_uint8.astype('uint8')

        # increase the 24x32 px image to 240x320px for ease of seeing
        bigIR = cv2.resize(depth_uint8, dsize=(240,320), interpolation=cv2.INTER_CUBIC)

        # Normalize the image
        normIR = cv2.normalize(bigIR, bigIR, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply a color heat map
        colorIR = cv2.applyColorMap(normIR, cv2.COLORMAP_JET)

        # Use a bilateral filter to blur while hopefully retaining edges
        brightBlurIR = cv2.bilateralFilter(normIR,9,150,150)

        # Threshold the image to black and white 
        retval, threshIR = cv2.threshold(brightBlurIR, 210, 255, cv2.THRESH_BINARY)

        # Define kernal for erosion and dilation and closing operations
        kernel = np.ones((5,5),np.uint8)

        erosionIR = cv2.erode(threshIR,kernel,iterations = 1)

        dilationIR = cv2.dilate(erosionIR,kernel,iterations = 1)

        closingIR = cv2.morphologyEx(dilationIR, cv2.MORPH_CLOSE, kernel)

        # Detect edges with Canny detection, currently only for visual testing not counting
        edgesIR = cv2.Canny(closingIR,50,70, L2gradient=True)

        # Detect countours
        contours, hierarchy = cv2.findContours(closingIR, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Get the number of contours ( contours count when touching edge of image while blobs don't)
        ncontours = str(len(contours))

        # Invert the image
        invertIR = cv2.bitwise_not(closingIR)

        ## Begin Blob Detection ##
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 255;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 7000
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
         
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.01
         
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs.
        keypoints = detector.detect(invertIR)
        ## End Blob Detection ##
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        IR_with_keypoints = cv2.drawKeypoints(invertIR, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        nblobs = str(len(keypoints))

        # Test wether Blobs or Contours provide a more accurate count
        # Put text number of Contour Keypoints on Screen in Blue
        cv2.putText(IR_with_keypoints, nblobs, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),3)

        # Put text number of Contour Keypoints on Screen in Blue
        cv2.putText(IR_with_keypoints, ncontours, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),3)

        #make all arrays same color space befor concatenating
        RGBnormIR = cv2.cvtColor(normIR, cv2.COLOR_GRAY2RGB)
        brightBlurIR = cv2.cvtColor(brightBlurIR, cv2.COLOR_GRAY2RGB)
        edgesIR = cv2.cvtColor(edgesIR, cv2.COLOR_GRAY2RGB)
        
        # stack 2 sets of images side by side for testing analysis
        imstack1 = np.concatenate((edgesIR,colorIR), axis=1)   #1 : horz, 0 : Vert.
        imstack2 = np.concatenate((brightBlurIR,IR_with_keypoints), axis=1)   #1 : horz, 0 : Vert.

        # Then stack those 2 verticaly
        imstack = np.concatenate((imstack1,imstack2), axis=0)   #1 : horz, 0 : Vert.

        # Show images in window during testing
        cv2.imshow("Combined", imstack)

        # Save timestamped PNGs of images and CSV of temperatures during tsting
        imagepath = '/home/pi/irpython/images'
        tempspath = '/home/pi/irpython/temps'
        #np.savetxt(os.path.join(tempspath , textTime + " temps.csv"), f, delimiter=",") # save csv file of temps 
        #cv2.imwrite(os.path.join(imagepath , textTime + " frames.png"), imstack ) # save png of frames
        
        cv2.waitKey(1)



try:
    while True:
        irCounter()



except KeyboardInterrupt:
    print('interrupted!')

