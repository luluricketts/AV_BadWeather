import os
import glob
import numpy as np
import csv
import shutil
import time
# from get_anomalous_bus_stops import BagExtractor
from query_noaa import query_noaa_bad_weather

def get_badweather_files():

    # parses through each directory, gets gps location and query nws 

    cameras = [ "front-center", "front-right", "front-left" ]
    base_folder = "/mnt/Data-Kolossus/busedge/RAW_DATA/" # TODO UPLOAD DATA AND CHANGE
    folders = glob.glob(base_folder + "*")
    # random.shuffle(folders)

    for folder in folders:
        
        try:
            year, month, day, hour, minute = os.path.split(folder)[1].split("_")
            date = f'{year}-{month}-{day}'
        except:
            continue
        if int(hour) < 7 or int(hour) > 19:
            continue # nighttime

        bag_files = glob.glob(folder + "/*.bag")
        if bag_files == []:
            continue
        
        be = BagExtractor(
            cameras=cameras
        )

        with open('badweather_bags.txt', 'w') as file:
            for bag_file in bag_files:
                print("Reading: ", bag_file)
                bag_write = False
                locs = be.extract_frames(bag_file, only_locs=True)
                for i, loc in enumerate(locs): 
                    lat, lon, _ = loc
                    latlong = np.array([lat, long])
                    if query_noaa_bad_weather(date, latlong):
                        if bag_write:
                            file.write(',' + i)
                        else:
                            file.write(bag_file + '\t' + i )
                            bag_write = True
                file.write('\n')
                        


def get_badweather_data():

    # parse through bag files, download data
    with open('badweather_bags.txt', 'r') as file:
        lines = file.readlines()

    cameras = [ "front-center", "front-right", "front-left" ]
    be = BagExtractor(cameras=cameras)

    for bag in lines:
        bag_file = bag.strip().split('\t')[0]
        loc_i = bag.strip().split('\t')[1].split(',')

        # TODO edit --
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
