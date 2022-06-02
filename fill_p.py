import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob
import os
 
_json = glob.glob('./*.json')
 
name_dict = {
                "__background__": "0",
                "jb": "1",
                "ak": "2",
                "bb": "3",
                "zw": "4",
                "hh": "5",
                "dbh": "6",
                "fcb": "7",
                "lf": "8",
                "yw": "9",
                "xk": "10",
                "fbz": "11",
                "akd": "12",
                "zlb": "13",
                "ywq": "14",
                "gh": "15",
                "jy": "16",
                "qj": "17",
                "sr": "18",
                "yq": "19",
                "wyb": "20",
                "qb": "21",
                "yy": "22",
                "ms": "23",
                "th": "24",
                "mx": "25",
                "sz": "26",
                "lw": "27",
                "jh": "28"
            }
for j in _json:
    with open(j, 'r') as f:
        r = f.read()
    r = json.loads(r)
    obj = r['shapes']
 
    img = np.zeros((2048,2048), dtype="uint8")
    
    for i in obj:
        name = i['label']
        po = i['points']
        n = name_dict[name]
        print(name, po, n)

        img = cv2.fillPoly(img, [np.array(po, dtype=np.int32)], int(n), 0, 0)

    path = '../../ade_01/ADEChallengeData2016/annotations/training/'
    cv2.imwrite(os.path.join(path , j[:-4]+'png'), img)