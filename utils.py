from cv_bridge import CvBridge, CvBridgeError
from enum import Enum
import json

cv_bridge = CvBridge()

class Task(Enum):
    OBJECT_DETECTION = 1
    HUMAN_RECOGNITION = 2
    ACTIVITY_RECOGNITION = 3
    GESTURE_RECOGNITION = 4
    TASK_ORIENTED_GRASPING = 5
    HANDOVER_OBJECT = 6
    RECEIVE_OBJECT = 7

def get_cv_frame(msg):
    try:
        cv_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except CvBridgeError as e:
        print(e)
        return None
    return cv_img

def get_compressed_cv_frame(msg):
    try:
        cv_img = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except CvBridgeError as e:
        print(e)
        exit(0)
    return cv_img

def get_compressed_cv_frame_depth(msg):
    try:
        cv_img = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError as e:
        print(e)
        exit(0)
    return cv_img

def get_task_type(jfile):
    if 'object_detection' in jfile:
        return Task.OBJECT_DETECTION
    if 'human_recognition' in jfile:
        return Task.HUMAN_RECOGNITION
    elif 'gesture_recognition' in jfile:
        return Task.GESTURE_RECOGNITION
    elif 'activity_recognition' in jfile:
        return Task.ACTIVITY_RECOGNITION
    elif 'handover_object' in jfile:
        return Task.HANDOVER_OBJECT
    elif 'receive_object' in jfile:
        return Task.RECEIVE_OBJECT

def get_bagfile_name(jfile):
    with open(jfile, 'r') as json_file:
        data = json.load(json_file)
        bagfile_name = data['bagfile']
    return bagfile_name
