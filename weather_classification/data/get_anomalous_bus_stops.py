import os
import argparse

import glob

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geopy.distance import geodesic

import random



class BagExtractor():
    def __init__(self, cameras) -> None:        
        self.cameras_dict = {
            "back-right"  : '/camera1/image_raw/compressed',
            "back-left"   : '/camera2/image_raw/compressed',
            "front-center" : '/camera3/image_raw/compressed',
            "front-left"    : '/camera4/image_raw/compressed',
            "front-right"   : '/camera5/image_raw/compressed',
        }
        self.cameras_dict_rev = {
            v : k for k, v in self.cameras_dict.items()
        }
        self.cameras = cameras

    def extract_frames(self, bag_file, only_locs=False):
        if not only_locs:
            topics = [ self.cameras_dict[x] for x in self.cameras ]
            topics.append("/fix")
        else:
            topics = [ '/fix' ]
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()
        count = 0
        cv_imgs = {
            k : [] for k in self.cameras
        }
        locs = []
        for topic, msg, t in bag.read_messages(topics=topics):
            if topic == "/fix":
                locs.append([msg.latitude, msg.longitude, msg.altitude])
            else:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg)
                curr_camera = self.cameras_dict_rev[topic]
                cv_imgs[curr_camera].append(cv_img)
            count += 1
        bag.close()
        if not only_locs:
            return locs, cv_imgs
        else:
            return locs

months_code = {
    "01" : "January",
    "02" : "February",
    "03" : "March",
    "04" : "April",
    "05" : "May",
    "06" : "June",
    "07" : "July",
    "08" : "August",
    "09" : "September",
    "10" : "October",
    "11" : "November",
    "12" : "December",
}

import csv
def get_bus_stop_info(path="busstops.txt"):
    bus_stops = {}
    with open(path) as f:
        csvf = csv.reader(f)
        for i, row in enumerate(csvf):
            if i == 0:
                continue            
            stop_name = row[2].replace("@", "").replace(" ", "-").replace("(", "").replace(")", "").replace(".", "").replace("--", "-").replace("--", "-").rstrip('-')
            lat, lon = float(row[4]), float(row[5])
            bus_stops[stop_name] = [lat, lon]
    return bus_stops


bus_stops = get_bus_stop_info()


for code, month in months_code.items():
    for stop_name in bus_stops:
       os.makedirs("BusStopData/{}/{}".format(month, stop_name), exist_ok=True)

if __name__ == '__main__':

    import time
    months = [ "January", "February", "March", "April", "May", "November", "December" ]
    max_per_month = 200
    cameras = [ "front-center", "front-right", "front-left" ]

    base_folder = "/mnt/Data-Kolossus/busedge/RAW_DATA/"
    out_base_folder = "/mnt/Data/BusStopDataAllAngles/"
    folders = glob.glob(base_folder + "*")
    random.shuffle(folders)

    counter = {
        month : 0 for month in months
    }
    for folder in folders:
        
        try:
            year, month, date, hour, minute = os.path.split(folder)[1].split("_")
        except:
            continue
        if not months_code[month] in months:
            continue
        
        if int(month) <=6 and year == "2021":
            continue

        if int(hour) < 9 or int(hour) > 17:
            night = True
        else:
            night = False

        # if not (int(hour) > 17):
        #     continue

        bag_files = glob.glob(folder + "/*.bag")
        if bag_files == []:
            continue
        
        counter[months_code[month]] += 1
        if counter[months_code[month]] > max_per_month:
            continue
        
        # print(year, month, date, hour, minute)
        # continue
        # bag_files = bag_files[5:8]
        be = BagExtractor(
            cameras=cameras
        )

        for bag_file in bag_files:
            print("Reading: ", bag_file)
            dt = os.path.split(bag_file)[1].split("_")[1]
            locs = be.extract_frames(bag_file, only_locs=True)
            save_ims_idxs = []
            count = 0
            for i, loc in enumerate(locs):
                lat, lon, _ = loc
                for stop_name in bus_stops:
                    lat_st, lon_st = bus_stops[stop_name]
                    if geodesic((lat_st, lon_st), (lat, lon)).m < 60:
                        save_ims_idxs.append([stop_name, i, count])
                        count += 1
            
            if save_ims_idxs != []:
                print("Reading Images: ", bag_file)
                locs, cv_imgs = be.extract_frames(bag_file)
                for stop_name, i, count in save_ims_idxs:
                    for camera in cameras:
                        out_folder = os.path.join(out_base_folder, "{}/{}/".format(months_code[month], stop_name))
                        if night == True:
                            out_folder = os.path.join(out_folder, "night/")
                        os.makedirs(out_folder, exist_ok=True)
                        pth = os.path.join(out_folder, "frame_{}_{}_{}.jpg".format(camera, dt, count))
                        try:
                            cv2.imwrite(pth, cv_imgs[camera][int(i*5)])
                        except:
                            try:
                                cv2.imwrite(pth, cv_imgs[camera][int(i)])
                            except:
                                pass

