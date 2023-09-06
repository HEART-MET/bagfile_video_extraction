import numpy as np
import rosbag
import os
import glob
import pdb
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import shutil
import pandas as pd
import argparse

from utils import *


def extract_video(bagfile, data_folder, camera_topic):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    head_cam_name = os.path.join(data_folder, 'camera.mp4')
    frame_rate = 20.0 # FPS
    width = 640
    height = 480
    head_cam_writer = cv2.VideoWriter(head_cam_name, fourcc, frame_rate, (width,height))
    head_cam_timestamps = []

    bag = rosbag.Bag(bagfile)
    # Camera topic
    topics_to_read = [camera_topic]

    act_end_topic = '/metrics_refbox_client/activity_recognition_result'
    gest_end_topic = '/metrics_refbox_client/gesture_recognition_result'

    topics_to_read.append(act_end_topic)
    topics_to_read.append(gest_end_topic)

    task = 'activity_recognition'
    topic_exists = False
    for topic, msg, t in bag.read_messages(topics=topics_to_read):
        if topic == topics_to_read[0]: # camera topic
            topic_exists = True
            head_cam_writer.write(get_cv_frame(msg))
            head_cam_timestamps.append(t.to_sec())
        elif topic == act_end_topic or topic == gest_end_topic:
            if topic == act_end_topic:
                task = 'activity_recognition'
            elif topic == gest_end_topic:
                task = 'gesture_recognition'
            break
    head_cam_writer.release()
    if not topic_exists:
        print("Topic %s does not exist in %s! Could not extract data" % (camera_topic, bagfile))
        return

    head_cam_timestamps = np.array(head_cam_timestamps)
    np.save(os.path.join(data_folder, 'camera_ts.npy'), head_cam_timestamps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bagfile_path')
    parser.add_argument('json_path')
    parser.add_argument('data_path')
    parser.add_argument('-c', '--camera_topic', default='/usb_cam/image_raw')
    args = parser.parse_args()
    path_to_bagfiles = args.bagfile_path
    path_to_json_files = args.json_path
    path_to_store_extracted_data = args.data_path

    json_files = sorted(glob.glob(path_to_json_files + '/*.json'))
    for jfile in json_files:
        with open(jfile, 'r') as fp:
            data = json.load(fp)
        bagfile_name = data['bagfile']
        bagfile_path = os.path.join(path_to_bagfiles, bagfile_name)
        if not os.path.exists(bagfile_path):
            print('Bagfile %s does not exist' % bagfile_path)
            continue
        print('Extracting data from %s' % bagfile_path)
        basename = os.path.basename(bagfile_name)[:-4]
        data_folder = os.path.join(path_to_store_extracted_data, 'data', basename)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        extract_video(bagfile_path, data_folder, args.camera_topic)


if __name__ == '__main__':
    main()
