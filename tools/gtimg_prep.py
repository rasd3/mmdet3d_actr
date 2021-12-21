#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle as pkl


def dbpkl_to_dbimgpkl():

    def arange_img_list(db_img_list):
        arr_list = [[] for _ in range(7481)]
        for img_dir in db_img_list:
            img_idx, _ = img_dir.split('_')
            arr_list[int(img_idx)].append(img_dir)
        return arr_list

    def euc_dis(coor1, coor2):
        coor1, coor2 = coor1.astype(np.float32), coor2.astype(np.float32)
        return np.sqrt(np.square(coor1 - coor2).sum())

    db_infos = pkl.load(open('./data/kitti/kitti_dbinfos_train.pkl', 'rb'))

    db_img_list = os.listdir('./data/kitti/gt_database_img/img/')
    db_img_list.sort()
    img_list = arange_img_list(db_img_list)

    img_path = 'data/kitti/gt_database_img/img/'
    txt_path = 'data/kitti/gt_database_img/txt/'

    for cur_class in db_infos.keys():
        connect_cnt = 0
        all_cnt = 0
        print(cur_class)
        for idx in range(len(db_infos[cur_class])):
            all_cnt += 1
            c_db = db_infos[cur_class][idx]
            c_idx = int(c_db['image_idx'])
            if 'bbox' not in c_db:
                lines = open(
                    './data/kitti/training/label_2/%06d.txt' %
                    c_db['image_idx'], 'r').readlines()
                for line in lines:
                    line = line.split(' ')
                    b_size = np.array([float(line[i]) for i in [9, 10, 8]],
                                      dtype=np.float32)
                    if (b_size == c_db['box3d_lidar'][3:6]).sum() == 3:
                        c_db['bbox'] = np.array(
                            [float(line[i]) for i in [4, 5, 6, 7]],
                            dtype=np.float32)
                        break
            c_bbox = np.array(
                [c_db['bbox'][::2].sum() / 2, c_db['bbox'][1::2].sum() / 2])

            if len(img_list[c_idx]) == 0:
                continue
            bet_min, bet_dir, mi_bbox = 30., -1, np.array([-1., -1.])
            for img_dir in img_list[c_idx]:
                i_img = cv2.imread(img_path + img_dir)
                i_H, i_W, _ = i_img.shape
                i_txt = open(txt_path + img_dir[:-4] + '.txt',
                             'r').readline().split(' ')
                i_bbox = np.array([int(a) for a in i_txt])
                i_bbox_ul = i_bbox.copy()
                i_bbox[0], i_bbox[1] = i_bbox[0] + i_W / 2, i_bbox[1] + i_H / 2

                bet_dis = euc_dis(i_bbox[:2], c_bbox)
                if bet_min > bet_dis:
                    bet_min = bet_dis
                    bet_dir = img_dir
                    mi_bbox = i_bbox_ul

            if bet_dir == -1:
                continue
            c_db['img_path'] = img_path + bet_dir
            c_db['img_bbox_coor'] = mi_bbox
            connect_cnt += 1

        print('%d / %d' % (connect_cnt, all_cnt))

    pkl.dump(db_infos, open('data/kitti/kitti_dbinfos_img_train.pkl', 'wb'))


def make_scene_list():

    def match_btw_calib(num1, num2):
        cal1 = open('./data/kitti/training/calib/%06d.txt' % num1).readlines()
        cal2 = open('./data/kitti/training/calib/%06d.txt' % num2).readlines()
        for line1, line2 in zip(cal1, cal2):
            if line1.strip() != line2.strip():
                return False
        return True

    scene_idx = 0
    scene_list = []
    visit = [-1 for _ in range(7481)]
    for i in range(7481):
        if visit[i] != -1:
            continue
        scene_list.append([i])
        for j in range(i + 1, 7481):
            if match_btw_calib(i, j):
                scene_list[scene_idx].append(j)
                visit[j] = scene_idx
        scene_idx += 1

    scene_dict = {'list': scene_list, 'sample': visit}
    pkl.dump(scene_dict, open('./data/kitti/kitti_dbinfos_scene_list.pkl',
                              'wb'))


if __name__ == '__main__':
    dbpkl_to_dbimgpkl()
    #  make_scene_list()
