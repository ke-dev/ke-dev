from ast import Delete
from cProfile import label
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import numpy as np
import time
import scipy.io as io
from json_01 import load_json_file, save_json_file
from labelme import gen_labelme_json
import cv2
import os
import shapely
from shapely.ops import unary_union
import re
from glob import glob

start = time.time()

def Cal_area_2poly(data1,data2):
    """
    任意两个图形的相交面积的计算
    """

    poly1 = Polygon(data1).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两图形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area



def Cal_iou_area(bbox1, bbox2):
    """
    计算重合面积
    """
    is_inter = Cal_area_2poly(bbox1, bbox2)
    if (is_inter != 0):
        S1 = bbox1.area       #area
        S2 = bbox2.area
        #union_area = Cal_area_2poly(bbox1,bbox2)
        iou1 = is_inter / S1
        iou2 = is_inter / S2
        if(((1 > iou1 >= 0.5) or (1 > iou2 >= 0.5))):       #0.5 ~ 1
            #if score1 > score2:
            return True
        elif (iou1 >= 1.0 and (S1 / S2) >= 0.5) or (iou2 >= 1.0 and (S2 / S1) >= 0.5):    # > 1 need to del
            return True
        else:
            return False

    else:
        return False



video_path = "./source_data/before/"
gen_path = "./source_data/after/"
#video_path = "./source_data/raw_data/" # 1000 
#gen_path = "./source_data/new_data/"
frames = glob(os.path.join(video_path, '*.json'))

for i, frame in enumerate(frames):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(frame)
    data_now = data[0]
    Newdir = os.path.join(video_path, str(data_now) + '.json')

    Gendir = os.path.join(gen_path, str(data_now) + '.json')

    name = str(data_now) + ".jpg"

    data_json = load_json_file(Newdir)

    list_01 = []
    score_01 = []
    label_01 = []

    need_del_ele = []
    merge_label = []

    version_01 = data_json['version']
    image_path = data_json['imagePath']      #1515599_0.jpg
    image_height = data_json['imageHeight']
    image_width = data_json['imageWidth']

    data_dict = data_json['shapes']
    for i in data_dict:
        list_01.append(i['points'])
        score_01.append(i['score'])
        label_01.append(i['label'])


    length_list = len(list_01)
    #print(list_01)
    length_after = length_list

    for i in range(0, length_list):
        for j in range(i+1, length_list):
            if (Cal_iou_area(Polygon(list_01[i]), Polygon(list_01[j]))):
                a = label_01[i].split('_')
                b = label_01[j].split('_')
                if a[0] != b[0]:             # 有重合且不是同一类型需要删除
                    if score_01[i] > score_01[j]:
                        print("need to del %d, %s" % (j, label_01[j]))
                        need_del_ele.append(label_01[j])
                    else:
                        print("need to del %d, %s" % (i, label_01[i]))
                        need_del_ele.append(label_01[i])

                else:                        # 需要合并
                   # print("need to merge%s, %s" % ((label_01[i]), label_01[j]))
                    merge_label.append(label_01[i])
                    merge_label.append(label_01[j])
                    print("need to merge %s " % merge_label)
                    
                    #merge_point = list_01[label_01.index(label_01[i])]+list_01[label_01.index(label_01[j])]
                    #list_01[label_01.index(label_01[i])] = shapely.ops.unary_union(merge_point)
                    #del list_01[label_01.index(label_01[j])]    #point
                    #label_01.remove(label_01[j])                #label


    set_c = set(label_01) & set(need_del_ele)
    list_c = list(set_c)

    for m in list_c:
        #print(m)
        #print(tuple(list_01[label_01.index(m)]))
        
        del list_01[label_01.index(m)]    #point
        label_01.remove(m)                #label
    
    #print(list_01)
    if merge_label:                      #删除标签合并点
        length_merge = len(merge_label)
        for i in range(0, length_merge):
            if i != 0:                  #保留第一个，删除其他
                #print("sdfsdfsdf", list_01[label_01.index(merge_label[0])])
                merge_point = list_01[label_01.index(merge_label[0])]+list_01[label_01.index(merge_label[i])]
                #list_01[label_01.index(merge_label[0])] = shapely.ops.unary_union(Polygon(merge_point))  # 此处维度出错
                #list_01[label_01.index(merge_label[0])] = shapely.ops.unary_union(Polygon(merge_point))
                list_01[label_01.index(merge_label[0])] = list(zip(*shapely.ops.unary_union(Polygon(merge_point)).exterior.coords.xy))
                #points_01 = list_01

                del list_01[label_01.index(merge_label[i])]    #point
                label_01.remove(merge_label[i])                #label
                print("need ot merge %s" % merge_label[i])
                #print(list_01)
    #print(len(label_01), len(list_01))
    #print(list_01)
    new_json_data = gen_labelme_json(name, image_height, image_width, label_01, list_01)
    save_json_file(new_json_data, Gendir)

#print(need_del_ele)
#使用最大得分过滤，并保存内部缺陷, 合并重合得分相近的缺陷
end = time.time()
print("time(s): ",(end-start))