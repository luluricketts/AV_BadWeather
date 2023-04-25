################
# run on local #
################

import argparse
import json
import paramiko

import cv2
import numpy as np


def main(json_file, args, client):
    sftp = client.open_sftp()
    for k,v in json_file.items():
    
        if f'{args.label_type}_label' in v.keys():
            continue
        if args.label_type == 'rc' and v['source'] == 'MWI':
            json_file[k][f'{args.label_type}_label'] = 3 # no label

        key = None
        while key not in range(len(l_names[args.label_type])):

            if v['source'] == 'DENSE':
                img_path = '/home/lulu/datasets/cam_stereo_left_lut/' + v['img_path'].split('/')[-1]
            else:
                img_path = v['img_path']

            try:
                with sftp.open(img_path) as f:
                    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)            

                cv2.imshow('image', img)
                key = chr(cv2.waitKey(0))
            except:
                json_file[k][f'{args.label_type}_label'] = -1 # invalid shapes
                break

            if key == 'q':
                print(k)
                return json_file
            try:
                key = int(key)
            except ValueError:
                print('Incorrect label range, try again')
            if key not in range(len(l_names[args.label_type])):
                print('Incorrect label range, try again')
        
        json_file[k][f'{args.label_type}_label'] = key

    return json_file


parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, required=True, help='json file to edit')
parser.add_argument('--label_type', type=str, required=True, choices=['tod', 'rc'])
if __name__ == "__main__":
    args = parser.parse_args()
    l_names = {'tod': ['day', 'night', 'dawn/dusk'], 'rc': ['dry', 'wet', 'snow', 'no_road']}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("gpuserver6.perception.cs.cmu.edu",username="user",password='password',timeout=4)

    with open(args.json, 'r') as file:
        json_file = json.load(file)

    print(f'Label type: {args.label_type}, enter keys:')
    print(dict(zip(range(len(l_names[args.label_type])), l_names[args.label_type])))
    print('Press \'q\' to quit')

    new_json = main(json_file, args, client)


    with open(args.json, 'w') as file:
        json.dump(new_json, file)

    client.close()