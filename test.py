import sys
import time
import colorsys
import numpy as np
import cv2
import datetime

from PIL import Image

sys.path.insert(0, "./build/lib.linux-armv7l-3.5")

import MLX90640 as mlx

img = Image.new( 'L', (24,32), "black")

#function to false color image from code example
#def temp_to_col(val):
    #hue = (180 - (val * 6)) / 360.0
    #return tuple([int(c*255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)])

#function to false color image from code example
#def temp_to_col(val):
#    return tuple([int(c*255) for c in colorsys.hsv_to_rgb(1.0, 0.0, val/50.0)])

try:
    while True:

        textTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        mlx.setup(2) #set frame rate of MLX90640
        
        f = mlx.get_frame()

        mlx.cleanup()

        v_min = min(f)
        v_max = max(f)

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
 
        imgIR = np.array(img)

        bigIR = cv2.resize(imgIR, dsize=(240,320), interpolation=cv2.INTER_CUBIC)
    
        normIR = cv2.normalize(bigIR, bigIR, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        colorIR = cv2.applyColorMap(normIR, cv2.COLORMAP_JET)

        brightBlurIR = cv2.bilateralFilter(normIR,9,150,150)

        retval, threshIR = cv2.threshold(brightBlurIR, 200, 255, cv2.THRESH_BINARY)
        edgesIR = cv2.Canny(threshIR,50,70, L2gradient=True)

        contours, hierarchy = cv2.findContours(threshIR, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        ncontours = str(len(contours))

        

        

        invertIR = cv2.bitwise_not(threshIR)


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
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        IR_with_keypoints = cv2.drawKeypoints(invertIR, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        nblobs = str(len(keypoints))
        
        cv2.putText(IR_with_keypoints, nblobs, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),3)
        cv2.putText(IR_with_keypoints, ncontours, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),3)

        #make all arrays same color space befor concatenating
        RGBnormIR = cv2.cvtColor(normIR, cv2.COLOR_GRAY2RGB)
        brightBlurIR = cv2.cvtColor(brightBlurIR, cv2.COLOR_GRAY2RGB)
        edgesIR = cv2.cvtColor(edgesIR, cv2.COLOR_GRAY2RGB)
        
        imstack1 = np.concatenate((edgesIR,colorIR), axis=1)   #1 : horz, 0 : Vert.
        imstack2 = np.concatenate((brightBlurIR,IR_with_keypoints), axis=1)   #1 : horz, 0 : Vert.
        

        imstack = np.concatenate((imstack1,imstack2), axis=0)   #1 : horz, 0 : Vert.
        # Show keypoints
        cv2.imshow("Keypoints", imstack)


        


        #np.savetxt(textTime + " temps.csv", f, delimiter=",") # save csv file of temps
        #img.save(textTime + " frames.png") # save png of frames
        
        
         
        #time.sleep(.1)
        cv2.waitKey(1)
        

except KeyboardInterrupt:
    print('interrupted!')

