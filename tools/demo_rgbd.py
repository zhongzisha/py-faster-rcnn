'''
Usage: 
python tools/demo_rgbd.py --gpu 0 \
    --prototxt models/potsdam/ZF/faster_rcnn_end2end/test.prototxt \
    --caffemodel output/faster_rcnn_end2end_update_means/voc_2014_train/zf_faster_rcnn_iter_70000.caffemodel \
    --im_filepath /media/slave1temp/data/voc/VOCdevkit2014/VOC2014/JPEGImages/2014_000001.jpg \
    --set PIXEL_MEANS '[[[84,86,117]]]'
    
python tools/demo_rgbd.py --gpu 0 \
    --prototxt models/potsdam/ZF/faster_rcnn_end2end/test.prototxt \
    --caffemodel output/faster_rcnn_end2end_update_means/voc_2014_train/zf_faster_rcnn_iter_70000.caffemodel \
    --rgb_filepath /media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area2.png \
    --dsm_filepath /media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area2.png \
    --set PIXEL_MEANS '[[[84,86,117]]]' 
    
python tools/demo_rgbd.py --gpu 0 \
    --prototxt models/potsdam_d/ZF/faster_rcnn_end2end/test.prototxt \
    --caffemodel output/faster_rcnn_end2end_update_means/voc_2014_train/d_zf_faster_rcnn_iter_70000.caffemodel \
    --rgb_filepath /media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area11.png \
    --dsm_filepath /media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area11.png \
    --set TEST.HAS_DSM True PIXEL_MEANS '[[[84,86,117]]]' DSM_MEANS '[32]' 
'''


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig("test.png")

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--prototxt', dest='prototxt', help='path of test.prototxt',
                        default=None, type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='path of caffemodel',
                        default=None, type=str)
    parser.add_argument('--rgb_filepath', dest='rgb_filepath', help='path of rgb_filepath',
                        default=None, type=str)
    parser.add_argument('--dsm_filepath', dest='dsm_filepath', help='path of dsm_filepath',
                        default=None, type=str)
    parser.add_argument('--save_prefix', dest='save_prefix', help='save_prefix',
                        default=None, type=str)
    parser.add_argument('--nms_threshold', dest='nms_threshold', help='nms_threshold',
                        default=None, type=float)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args() 
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    
    rgb_filepath = args.rgb_filepath
    rgb0 = cv2.imread(rgb_filepath)
    dsm0 = None
    if cfg.TEST.HAS_DSM == True:  
        dsm0 = cv2.imread(args.dsm_filepath, cv2.IMREAD_GRAYSCALE)
        
    # load net
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    
    BLOCK_SIZE = 500
    STEP_SIZE = 250

    height = rgb0.shape[0]
    width  = rgb0.shape[1]
    
    dets = np.zeros((0, 5), dtype=np.float32)
    j = 1
    dsm = None
    for y in xrange(0, height, STEP_SIZE):
        for x in xrange(0, width, STEP_SIZE):
            # yield the current window
            im = rgb0[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE, :]
            if dsm0 is not None:
                dsm = dsm0[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            # begin detection in a 500x500 image
            scores, boxes, seg = im_detect(net, im, dsm)
            cls_scores = scores[:, j]
            cls_boxes = boxes[:, j*4:(j+1)*4]
            cls_boxes[:, 0] += x
            cls_boxes[:, 1] += y
            cls_boxes[:, 2] += x
            cls_boxes[:, 3] += y
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False) 
            dets = np.vstack((dets, cls_dets))
            print "{}, {}".format(y, x)
    
    
    thresh = 0.05
    j = 1
    inds = np.where(dets[:, 4] > thresh)[0]
    dets = dets[inds, :] 
    inds = np.where(dets[:, 2] <= width)[0]
    dets = dets[inds, :] 
    inds = np.where(dets[:, 3] <= height)[0]
    dets = dets[inds, :] 
    cls_scores = dets[:, -1]
    cls_boxes = dets[:, 0:4]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
    print cls_dets.shape
    keep = nms(cls_dets, args.nms_threshold)
    cls_dets = cls_dets[keep, :]
    print cls_dets.shape
    
    # vis_detections(rgb0, 'car', cls_dets)
    inds = np.where(cls_dets[:, -1] >= 0.5)[0]
    if len(inds) == 0:
        exit()
        
    import cv2
    for i in inds:
        bbox = cls_dets[i, :4]
        score = cls_dets[i, -1]
        cv2.rectangle(rgb0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0)) 
    save_filepath = rgb_filepath[rgb_filepath.rfind('/')+1:rgb_filepath.rfind('.')] + '_' + args.save_prefix + '_box.jpg' 
    cv2.imwrite(save_filepath, rgb0)