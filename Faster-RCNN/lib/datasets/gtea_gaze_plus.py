# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

from lib.config import config as cfg
from lib.datasets.imdb import imdb


class gtea_gaze_plus(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'gtea-gaze-plus')
        self._data_path = cfg.FLAGS2["data_dir"]
        if image_set == "train":
          self._list_file = osp.join(self._data_path, "train_GTEA_GAZE_PLUS_voc.txt")
        elif image_set == "val":
          self._list_file = osp.join(self._data_path, "val_GTEA_GAZE_PLUS_voc.txt")
        elif image_set == "test":
          self._list_file = osp.join(self._data_path, "test_GTEA_GAZE_PLUS_voc.txt")
        elif image_set == "train-wholeBB":
          self._list_file = osp.join(self._data_path, "train_GTEA_GAZE_PLUS_voc_wholeBB.txt")
        elif image_set == "val-wholeBB":
          self._list_file = osp.join(self._data_path, "val_GTEA_GAZE_PLUS_voc_wholeBB.txt")
        elif image_set == "test-wholeBB":
          self._list_file = osp.join(self._data_path, "test_GTEA_GAZE_PLUS_voc_wholeBB.txt")
        else:
          assert False, "{} not found in GTEA Gaze+".format(image_set)

        self._classes = ('none',  # always index 0
                        'bacon', 'bottle', 'bowl', 'bread', 'cabinet', 'carrot',
                        'cereal', 'cheese', 'cheese bag', 'cream cheese', 'cup',
                        'egg', 'egg box', 'faucet', 'frying pan', 'glove', 'honey',
                        'jam', 'juice', 'ketchup', 'kettle', 'knife', 'mayonnaise',
                        'microwave', 'milk', 'mushroom', 'oil', 'onion', 'paper towel',
                        'pasta box', 'patty', 'peanut butter', 'pepper grinder', 'pizza',
                        'pizza bag', 'plastic bag', 'plate', 'pot', 'pot cover',
                        'refrigerator', 'salad dressing', 'salt', 'sausage', 'slice',
                        'spoon', 'stove', 'tea bag', 'tomato', 'turkey', 'vegetable',
                        'vinaigrette')

        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_list, self._annot_list = self._load_images_and_annotations()
        self._image_index = self._image_list
        # DEBUG
        #for img_path, annot_path in zip(self._image_list, self._annot_list):
        #    print(img_path, annot_path) 
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
        
        self._output_dir = "output"
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def get_classes(self):
        for each in ["train", "val", "test"]:
            list_file = "{}_GTEA_GAZE_PLUS_voc.txt".format(each)
            data_list = osp.join(self._data_path, list_file)
            # read each line in this file
            assert osp.exists(data_list), "{} not found".format(data_list)
            with open(data_list, 'r') as f:
                for line in f.readlines():
                    _, annot_file = line.split()


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        #print(i, type(i))
        return self._image_index[i]


    def _load_images_and_annotations(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        assert self._list_file, \
            'Path does not exist: {}'.format(self._list_file)

        with open(self._list_file) as f:
            image_list = []
            annot_list = []
            for line in f.readlines():
              # each line is "(image_path) (voc_annotation_path)"
              paths = line.split()
              image_list.append(osp.join(self._data_path, paths[0]))
              annot_list.append(osp.join(self._data_path, paths[1]))
            
        return image_list, annot_list


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_voc_annotation(path)
                    for path in self._annot_list]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        # DEBUG
        #print(gt_roidb)

        return gt_roidb


    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb


    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_voc_annotation(self, path):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = path
        assert osp.exists(filename), \
            "annotation not found at: {}".format(filename)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        """
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            not_none_objs = [
                obj for obj in objs if int(obj.find('none').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = not_none_objs
        """
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # bbs index is alrady 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id


    def _get_gtea_gaze_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = osp.join(
            self._output_dir,
            'results',
            'GTEA_GAZE_PLUS',
            'Main',
            filename)
        return path


    def _write_gtea_gaze_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == 'none':
                continue
            print('Writing {} GTEA GAZE+ results file'.format(cls))
            filename = self._get_gtea_gaze_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, path in enumerate(self.image_list):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(path, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
    
    
    def _coco_eval(self, all_boxes):
        return 0

    def _do_python_eval(self, output_dir='output'):
        # TODO: revise this
        annopath = self._output_dir + '\\GTEA_GAZE_PLUS' + '\\Annotations\\' + '{:s}.xml'
        imagesetfile = osp.join(
            self._output_dir,
            'GTEA_GAZE_PLUS',
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = osp.join(self._output_dir, 'annotations_cache')
                
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        #use COCO mAP metric
        m_aps = []
        for threshold in range(0.5, 0.05, 1.0):    
            aps = []
            for i, cls in enumerate(self._classes):
                if cls == 'none':
                    continue
                filename = self._get_gtea_gaze_results_file_template().format(cls)
                ap = self.voc_eval(
                    filename, annopath, imagesetfile, cls, cachedir,
                    ovthresh=threshold, use_07_metric=False)
                aps += [ap]
                print(('AP for {} = {:.4f}'.format(cls, ap)))
                pkl_output = osp.join(output_dir, "{}+{}_pr.pkl".format(threshold, cls))
                with open(pkl_output, 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            m_aps += [np.mean(aps)]
        print(('Mean AP = {:.4f}'.format(np.mean(m_aps))))
        print('~~~~~~~~')
        print('Results:')
        for m_ap in m_aps:
            print(('{:.3f}'.format(m_ap)))
        print(('{:.3f}'.format(np.mean(m_aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.FLAGS2["root_dir"], 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_gtea_gaze_results_file(all_boxes)
        self._do_coco_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == 'none':
                    continue
                filename = self._get_gtea_gaze_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = gtea_gaze_plus('train')
    res = d.roidb
    from IPython import embed;

    embed()
