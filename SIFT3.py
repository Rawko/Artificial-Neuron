import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Initialization
sift = cv2.SIFT_create()  
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Load Images
img1 = cv2.imread('/Users/rachelonassis/Desktop/EMC/room1.JPG')
img2 = cv2.imread('/Users/rachelonassis/Desktop/EMC/room2.JPG')

# Crop Images 
x, y, width, height = 1250, 1650, 800, 450
img1 = img1[y:y+height, x:x+width]
img2 = img2[y:y+height, x:x+width]

# Compute Keypoints and Descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# Feature Matching
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:10]



# Split View Visualization
img1_keypoints = cv2.drawKeypoints(img1, keypoints_1, None)
img2_keypoints = cv2.drawKeypoints(img2, keypoints_2, None)
split_view = np.hstack((img1_keypoints, img2_keypoints))
for match in good_matches:
    pt1 = tuple(map(int, keypoints_1[match.queryIdx].pt))
    pt2 = tuple(map(int, keypoints_2[match.trainIdx].pt))
    pt2_shifted = (pt2[0] + img1.shape[1], pt2[1])
    cv2.line(split_view, pt1, pt2_shifted, (0, 255, 0), 1)
    disparity_value = pt2[0] - pt1[0]
    label_pos = (int((pt1[0]+pt2_shifted[0])/2), int((pt1[1]+pt2_shifted[1])/2))
    cv2.putText(split_view, str(disparity_value), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow('Split View Visualization', split_view)
cv2.waitKey(0)


######Another Process^^^

# Initialize empty lists for src and dst points



###############


# Overlay Visualization
alpha = 0.5
overlay = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
#cv2.imshow('Overlay', overlay)

# Compute Disparity
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)
disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
disparity = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)

#
# Anaglyph Creation
red_channel = img1[:, :, 2]
green_channel = img2[:, :, 1]
blue_channel = img2[:, :, 0]
red_image = cv2.merge([np.zeros_like(red_channel), np.zeros_like(red_channel), red_channel])
cyan_image = cv2.merge([blue_channel, green_channel, np.zeros_like(blue_channel)])
anaglyph = cv2.addWeighted(red_image, 0.5, cyan_image, 0.5, 0)
#cv2.imshow('Anaglyph', anaglyph)
#cv2.waitKey(0)

# Define a function to compute the average displacement based on SIFT matches:
def compute_average_displacement(img1, img2):
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]
    
    displacements = []
    for match in good_matches:
        pt1 = keypoints_1[match.queryIdx].pt
        pt2 = keypoints_2[match.trainIdx].pt
        displacements.append(pt2[0] - pt1[0])
    
    return sum(displacements) / len(displacements)

# Global variable to keep track of the shift
shift = 0

def compute_anaglyph_with_homography(shift, img1, img2):
    """ Compute the anaglyph image with a given shift using homography """
    height, width, channels = img1.shape

    # Get keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Feature Matching
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]

    # Retrieve the good keypoints
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Add the shift to the x-coordinates of the destination points
    dst_pts[:,0,0] += shift

    # Find homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp the first image
    warped_img1 = cv2.warpPerspective(img1, M, (width, height))
    
    # Create the anaglyph using the red channel from warped_img1 and the green and blue channels from img2
    red_channel = warped_img1[:, :, 2]
    green_channel = img2[:, :, 1]
    blue_channel = img2[:, :, 0]
    red_image = cv2.merge([np.zeros_like(red_channel), np.zeros_like(red_channel), red_channel])
    cyan_image = cv2.merge([blue_channel, green_channel, np.zeros_like(blue_channel)])
    anaglyph = cv2.addWeighted(red_image, 0.5, cyan_image, 0.5, 0)
    
    # Compute net displacement based on SIFT matches
    net_disp = compute_average_displacement(warped_img1, img2)
    
    # Overlay the net displacement value
    cv2.putText(anaglyph, f"Net Displacement: {net_disp:.2f}px", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return anaglyph



refPt = []

def on_mouse(event, x, y, flags, param):
    global shift, refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x, y]
    elif event == cv2.EVENT_LBUTTONUP and refPt:
        shift += x - refPt[0]  # update the shift based on mouse drag distance
        anaglyph = compute_anaglyph_with_homography(shift, img1, img2)
        cv2.imshow("Anaglyph", anaglyph)

cv2.namedWindow("Anaglyph")
cv2.setMouseCallback("Anaglyph", on_mouse)

# Initially display the anaglyph without any shift
cv2.imshow("Anaglyph", compute_anaglyph_with_homography(shift, img1, img2))
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()


