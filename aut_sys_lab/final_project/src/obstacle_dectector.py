#!/usr/bin/env python3


import rospy
from std_msgs.msg import Header

from sensor_msgs.msg import Image,NavSatFix,Imu

import argparse
import cv2
import numpy as np


from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values


class Node():
    def __init__(self):
        rospy.loginfo("Init camera node!!")
        self.rate = rospy.Rate(5) # ROS Rate at 5Hz
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()
        self.cv_image = None

    def camera_callback(self, data):
        try:
            # We select bgr8 because its the OpneCV encoding by default
            self.cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        raw_height, raw_width, channels = self.cv_image.shape

        # set the ratio of resized image
        k = 5
        height = int(raw_height / k)
        width = int((raw_width) / k)
        # resize the image by resize() function of openCV library
        scaled = cv2.resize(self.cv_image, (width, height), interpolation=cv2.INTER_AREA)
        hsv_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])  # mb_marker_buoy_white
        upper_white = np.array([51, 0, 255])  # world1 = sunny
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
        res_white = cv2.bitwise_and(scaled, scaled, mask=mask_white)
        lower_black = np.array([0, 0, 50])  # mb_marker_buoy_black   and  mb_round_buoy_black
        upper_black = np.array([0, 0, 150])
        mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
        res_black = cv2.bitwise_and(scaled, scaled, mask=mask_black)
        cv2.imshow("Black", res_black)
        cv2.imshow("White", res_white)
        cv2.waitKey(1)



if __name__ == '__main__':
    rospy.init_node('lidar_localization_with_rgb')
    node = Node()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    '''
    range_filter = "HSV"
    setup_trackbars(range_filter)

    # construct the argument parser to parse input static image
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to the input image")
    args = vars(ap.parse_args())
    print(args)

    while not rospy.is_shutdown():

        raw_height, raw_width, channels = node.cv_image.shape

        # set the ratio of resized image
        k = 5
        height = int(raw_height/ k)
        width = int((raw_width) / k)
        # resize the image by resize() function of openCV library
        scaled = cv2.resize(node.cv_image, (width, height), interpolation=cv2.INTER_AREA)
        hsv_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)



        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(hsv_img, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        preview = cv2.bitwise_and(scaled, scaled, mask=thresh)
        # preview = cv.cvSet(preview, (0, 255, 0))
        # preview[np.where(preview == [1])] = (0, 255, 0)
        # preview[np.where(preview == [0])] = [255]   # for detect black

        cv2.imshow("Preview", preview)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

    rospy.logwarn("Shutting down")

    cv2.destroyAllWindows()
    '''