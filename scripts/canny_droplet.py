import cv2 as cv
import numpy as np
import os
import sys
import random

def calculate_contour_distance(contour1, contour2): 
    x1, y1, w1, h1 = cv.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)

# merge neighboring contours into one
def agglomerative_cluster(contours, threshold_distance=40.0):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else: 
            break
 
    return current_contours


def main(argv):
    filename = argv[0]
    root, ext = os.path.splitext(filename)

    srcs = []

    # Load images from video
    if (ext == '.avi'):
        cap = cv.VideoCapture(filename)
        if not cap.isOpened:
            print("Error opening video file!")
            return -1
        
        while True:
            # this is probably a bad idea...
            success, frame = cap.read()

            if not success: break # break loop if no more frames

            srcs.append(frame)
    else:
        srcs.append(cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR))
   
    # Check if image(s) is loaded fine
    if len(srcs) == 0:
        print ('Error opening image(s)!')
        return -1
    
    # l oop through every frame of the video and caluclate an average droplet size
    count = 0
    total_size = 0
    for idx, src in enumerate(srcs):
        # src = src.astype('float32')
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        edges = cv.Canny(gray, 40, 120)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found!")
            continue
        
        # filter out contours that are too big, small, or touch the border
        height, width = gray.shape
        image_area = height * width
        min_area = 0.005 * image_area
        max_area = 0.2 * image_area

        def touches_border(cnt, margin=2):
            x, y, w, h = cv.boundingRect(cnt)
            return (
                x <= margin or y <= margin or
                x + w >= width - margin or  + h >= height - margin
            )
        
        filtered_contours = [
            cnt for cnt in contours
            if min_area < cv.contourArea(cnt) < max_area and not touches_border(cnt)
        ]

        filtered_contours = agglomerative_cluster(filtered_contours)

        contour_image = cv.cvtColor(np.zeros_like(gray), cv.COLOR_GRAY2BGR) 
        font = cv.FONT_HERSHEY_SIMPLEX

        for i, cnt in enumerate(contours):
            cv.drawContours(contour_image, contours, i, (255, 255, 255), 2) # 2 is the thickness

        for i, cnt in enumerate(filtered_contours):
            # Generate a random BGR color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw the individual contour
            cv.drawContours(contour_image, filtered_contours, i, color, 2) # 2 is the thickness
            # if len(cnt) >= 5:
            #     ellipse = cv.fitEllipse(cnt) 
            #     cv.ellipse(contour_image, ellipse, color, 2)
            #     (center, axes, angle) = ellipse
            #     major, minor = axes
            #     circularity = min(major, minor) / max(major, minor)
            #     cv.putText(contour_image, str(circularity), cv.boundingRect(cnt)[:2], font, 1, color, 2, cv.LINE_AA)

        if not filtered_contours:
            print("No valid contours found after filtering.")
            continue
        
        # best ellipse is determined by the most circular one (not explicitly neccessary)
        best_ellipse = None
        best_score = 0
        circularity_thresh = 0.95
        for cnt in filtered_contours:
            if len(cnt) >= 5:
                ellipse = cv.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                major, minor = axes
                circularity = min(major, minor) / max(major, minor)
                # print(circularity)
                if circularity < circularity_thresh: continue
                if circularity > best_score:
                    best_score = circularity
                    best_ellipse = ellipse

        if best_ellipse is None:
            print("No valid ellipses could be fitted.")
            continue
    
        # calculate the volume of the ellipse rotated about the vertical axis
        (xc, xy), (major_axis, minor_axis), angle = best_ellipse

        a = major_axis / 2
        b = minor_axis / 2

        volume = (4/3) * np.pi * a * b * b

        # calculate the diameter of a sphere with the same volume
        equiv_diameter = ((6 * volume) / np.pi) ** (1/3)
        total_size += equiv_diameter
        count += 1

        # preview selected ellipse to confirm everything is function as desired
        print(f"Image {idx}: {equiv_diameter} px")
        result_img = gray.copy()
        cv.ellipse(result_img, best_ellipse, (0, 255, 0), 3)
        
        grey_3_channel = cv.cvtColor(result_img, cv.COLOR_GRAY2BGR)
        a = cv.resize(contour_image, (0, 0), None, .5, .5)
        b = cv.resize(grey_3_channel, (0, 0), None, .5, .5)

        numpy_horizontal = np.hstack((a, b))

        cv.imshow('Detected Ellipse', numpy_horizontal)
        cv.waitKey(0)

    print()
    print(f"Count: {count}")
    print(f"Average diameter {(total_size/count)} px")
    print(f"Average diameter {(total_size/count)*scale} mm")
    print()

# mm to pixel scaling value
scale = 6 / 895.09 # mm/pix

if __name__ == "__main__":
    main([sys.argv[1]])