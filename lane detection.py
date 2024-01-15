import numpy as np
import cv2

"""
Sample amount, would be better to have exact measurements,
but it's useful enough as is to measure out how close to the center the car is
by detecting how far the lines are from the center of the screen.

Of course, this doesn't necessarily work the best mainly due to differing camera
lenses and resolutions, but it helps by being some sort of guideline.
"""
CAR_WIDTH = 100

def lineDist(x1, y1, x2, y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)

video = cv2.VideoCapture("./car1.mp4")
vid, frame = video.read()
frame = cv2.resize(frame, (640, 480))

h, w, channels = frame.shape
hh, hw = h//2, w//2

lower = np.array([0, 0, 160])
upper = np.array([255, 40, 255])

tl, tr, bl, br = (hw-30, hh+50), (hw+70, hh+50), (hw-160, h), (hw+280, h)

while vid:
    vid, frame = video.read()
    frame = cv2.resize(frame, (640, 480)) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    p1 = np.float32([tl, bl, tr, br])
    p2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

    matrix = cv2.getPerspectiveTransform(p1, p2)

    mask = cv2.inRange(hsv, lower, upper)

    # These below are positional points for the narrow and wide areas
    # of a lane. They're rough, but they were used to determine where
    # the perspective transform should be made.    

    # cv2.circle(frame, tl, 10, (0, 0, 255), -1)
    # cv2.circle(frame, tr, 10, (0, 0, 255), -1)
    # cv2.circle(frame, bl, 10, (255, 0, 255), -1)
    # cv2.circle(frame, br, 10, (255, 0, 255), -1)

    perp_base = cv2.warpPerspective(frame, matrix, (w, h))
    
    perp_frame = cv2.warpPerspective(mask, matrix, (w, h))
    perp_frame = cv2.GaussianBlur(perp_frame, (5,5), 0)

    edges = cv2.Canny(perp_frame, 0, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 0, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(perp_base, (x1, y1), (x2, y2), (190, 152, 0), 5)
            if y1 <= hh/2:
                print("Distance to center:", lineDist(x1, y1, hw, h))
                if lineDist(x1, y1, hw, h) <= CAR_WIDTH:
                    # If this were a full car system, this warning would be implemented differently,
                    # inlcuding a check for turn signals. Since it's not, it can stay this way.
                    print("DANGER!!! CAR TOO CLOSE TO SIDE OF LANE")
                    
    cv2.imshow("Base frame", frame)
    cv2.imshow("Thresholded", mask)
    cv2.imshow("Warped perspective + thresholding", perp_frame)
    cv2.imshow("Warped perspective + lines", perp_base)
    cv2.imshow("Edges", edges)

    key = cv2.waitKey(0)
    if key == 27:
        break;

video.release()
cv2.destroyAllWindows()

