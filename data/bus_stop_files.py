import glob
import os
import time
import shutil
import csv

import numpy as np

from query_noaa import query_noaa_bad_weather


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



def get_badweather_busstops():

    bus_stops = get_bus_stop_info()
    base_dir = '/media/kolossus_data3/busedge/BusStopData/'
    out_dir = '/home/lulu/datasets/bus_edge'
    folders = glob.glob(base_dir + '*')

    max_per_stop = 10
    for folder in folders:
        print(f'Processing Month: {os.path.split(folder)[1]}')
        street_dirs = glob.glob(folder + '/*')
        if street_dirs == []:
            continue

        for street in street_dirs:
            print(f'Processing Stop: {os.path.split(street)[1]}')
            try:
                latlong = np.array(bus_stops[os.path.split(street)[1]])
            except:
                continue

            
            imgs = glob.glob(street + '/*.jpg')

            n_imgs = 0
            for img in imgs:
                try:
                    year, month, day, _, _, _ = os.path.split(img)[1].split("_")[2].split('-')
                except:
                    try:
                        year, month, day, _, _ = os.path.split(img)[1].split("_")[2].split("-")
                    except:
                        continue
                date = f'{year}-{month}-{day}'
                time.sleep(0.5) # api limit 5 per sec
                success, label, val = query_noaa_bad_weather(date, latlong)
                if not success:
                    continue

                pth = os.path.join(out_dir, label, os.path.split(img)[1])
                im = os.path.split(img)[1].split('.')[0]
                txt_pth = os.path.join(out_dir, label, im + '.txt')
                with open(txt_pth, 'w+') as f:
                    f.write(str(val))
                shutil.copyfile(img, pth)

                n_imgs += 1
                if n_imgs == max_per_stop:
                    break


# TODO
def get_badweather_pairs():
    ...



if __name__ == "__main__":
    
    get_badweather_busstops()