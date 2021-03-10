#! /usr/bin/env python

# MIT License

# Copyright (c) 2017-2018 Yongyang Nie

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Run a YOLO_v2 style detection model test images.
This ROS node uses the object detector class to run detection.
"""

#from object_detector import ObjectDetector
#from PIL import Image

#ros
import time 
import cv2
#import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
#from object_detection.msg import DetectionResult
#from object_detection.msg import DetectionResults


class ObjectDetectionNode:

    current_frame = None
    current_depth_frame = None
    prev_depth_frame = None

    info = None
    prev_info = None

    original_data = None
    prev_original_data = None


    def image_update_callback(self, data):
        try:
	    self.original_data = data
	    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	    #rospy.loginfo("Rgb callback")
            '''
	    self.original_data = data

	    self.visualize = 1
            self.model_path = "/home/pritammane105/catkin_ws/src/object_detection/scripts/model_data/yolo-tiny.h5"
            self.anchors_path = "/home/pritammane105/catkin_ws/src/object_detection/scripts/model_data/tiny_yolo_anchors.txt"
            self.classes_path = "/home/pritammane105/catkin_ws/src/object_detection/scripts/model_data/coco_classes.txt"
            self.iou_threshold = 0.5
            self.score_threshold = 0.5
	    self.input_size = (416, 416)

            self.detector = ObjectDetector(model_path=self.model_path,
                                       classes_path=self.classes_path,
                                       anchors_path=self.anchors_path,
                                       score_threshold=self.score_threshold,
                                       iou_threshold=self.iou_threshold,
                                       size=self.input_size)
	    
	    detection_image_pub = rospy.Publisher('/vedant', Image, queue_size=5)
            depth_image_pub = rospy.Publisher('/vedant/depth', Image, queue_size=5)
            info_image_pub = rospy.Publisher('/vedant/info', CameraInfo, queue_size=5)

	    
	    if cv_image is not None:
	        image, out_boxes, out_scores, out_classes = \
	            self.detector.detect_object(cv_image, visualize=self.visualize)
		
	        #img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
		#rospy.loginfo(img_msg)

		if 0 in out_classes :
		     rospy.loginfo("Person Detected")
		
	       
		#rospy.loginfo(type(self.original_data))
		#pub_image.header() = data.header()
		#pub_image.
		else : 		
		    detection_image_pub.publish(self.original_data)
		    rospy.loginfo("No Person Detected")
		    depth_image_pub.publish(self.current_depth_frame)
		    info_image_pub.publish(self.info)		
		
	        #msg = self.convert_results_to_message(out_boxes, out_scores, out_classes)
	        #detection_results_pub.publish(msg)
	    	'''


        except CvBridgeError as e:
            raise e

        self.current_frame = cv_image

    def depth_update_callback(self, data):
        try:
	    #rospy.loginfo("Depth callback")
            self.current_depth_frame = data
        except CvBridgeError as e:
            raise e

    
    def info_update_callback(self, data):
	try:
	    #rospy.loginfo("Info callback")
            self.info = data 	
        except CvBridgeError as e:
            raise e
    


    def __init__(self):
	
	
        rospy.init_node('faceDetect')
	
	rospy.Subscriber('/camera/rgb/image_rect_color', Image, self.image_update_callback, queue_size=5)
	rospy.Subscriber('/camera/depth_registered/image_raw', Image, self.depth_update_callback, queue_size=5)
	rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.info_update_callback, queue_size=5)	
	
	

        self.bridge = CvBridge()

        rospy.loginfo("Object Detection Initializing")
        
	
        detection_image_pub = rospy.Publisher('/vedant', Image, queue_size=5)
        depth_image_pub = rospy.Publisher('/vedant/depth', Image, queue_size=5)
        info_image_pub = rospy.Publisher('/vedant/info', CameraInfo, queue_size=5)
	

	'''
	rate = rospy.Rate(30)
	
	while not rospy.is_shutdown() : 
		
		rospy.spin()
		rate.sleep()
	'''
	       
	rate = rospy.Rate(30)
	
        while not rospy.is_shutdown():

	   # rospy.loginfo("Entered While")
            if self.current_frame is not None:
		start = time.time()

                face_cascade = cv2.CascadeClassifier('/home/pritammane105/catkin_ws/src/object_detection/scripts/haarcascade_frontalface_default.xml')
		gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 4)

		end = time.time()
		rospy.loginfo(end - start)

		if len(faces) != 0 :
		    rospy.loginfo("Person Detected")
		    detection_image_pub.publish(self.prev_original_data)
		    depth_image_pub.publish(self.prev_depth_frame)
		    info_image_pub.publish(self.prev_info)

		    continue 
               
		
		else :
		    detection_image_pub.publish(self.original_data)
		    self.prev_original_data = self.original_data
 
		    rospy.loginfo("No Person Detected")
		    depth_image_pub.publish(self.current_depth_frame)
		    self.prev_depth_frame = self.current_depth_frame 

		    info_image_pub.publish(self.info)	
		    self.prev_info = self.info 

#		rospy.loginfo(type(img_msg))
		#rospy.loginfo("Published")
                #rospy.loginfo(img_msg)

                #msg = self.convert_results_to_message(out_boxes, out_scores, out_classes)
                #detection_results_pub.publish(msg)
            rate.sleep()
	      	


'''
    @staticmethod
    def convert_results_to_message(out_boxes, out_scores, out_classes):

        msgs = DetectionResults()
        for i in range(len(out_scores)):
            msg = DetectionResult()
            msg.out_class = out_classes[i]
            msg.out_score = out_scores[i]
            msg.location = out_boxes[i, :]
            msgs.results.append(msg)

        return msgs
'''

if __name__ == "__main__":

    try:
        ObjectDetectionNode()
    except rospy.ROSInterruptException:
        pass
