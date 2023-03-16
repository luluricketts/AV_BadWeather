import argparse
import json

import cv2


def get_label(path):
    img = cv2.imread(path)
    cv2.imshow('image', img)
    key = cv2.waitkey(0)
    return key


def main(json_file, args):

    for k,v in json_file.items():
    
        if f'{args.label_type}_label' in v.keys():
            continue

        label = None
        while label not in range(len(l_names[args.label_type])):
            label = get_label(v['img_path'])
            if label == 'q':
                return json_file
            if label not in range(len(l_names[args.label_type])):
                print('Incorrect label range, try again')
        
        json_file[k]['tod_label'] = label

        break

    return json_file


parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, required=True, help='json file to edit')
parser.add_argument('--label_type', type=str, required=True, choices=['tod', 'rc'])
if __name__ == "__main__":
    args = parser.parse_args()
    l_names = {'tod': ['day', 'night', 'dawn/dusk'], 'rc': ['dry', 'wet']}

    with open(args.json, 'r') as file:
        json_file = json.load(file)

    print(f'Label type: {args.label_type}, enter keys:')
    print(dict(zip(range(len(l_names[args.label_type])), l_names[args.label_type])))
    print('Press \'q\' to quit')

    new_json = main(json_file, args)


    with open(args.json, 'w') as file:
        json.dump(new_json)

