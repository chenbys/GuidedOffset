import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/68/0-0-0.yaml')
# model_path, model_epoch = cur_path + '/../output/1-5.yaml/2007_trainval_2012_trainval/first', 7

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    # parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')
    # parser.add_argument('--model_prefix', default='output/1-5.yaml/2007_trainval_2012_trainval/first')
    parser.add_argument('--model_prefix',
                        default='/home/chenjunjie/workspace/Deformable-ConvNets/output/0e-4/w0/2007_trainval_2012_trainval/0e-4',
                        action='store_true')

    parser.add_argument('--model_epoch', default=7, action='store_true')

    args = parser.parse_args()
    return args


args = parse_args()


def show_conv_offset(output_all, im, im_name):
    from utils.show_offset import mshow_dconv_offset
    pa = output_all['mean0_output'].asnumpy()
    pb = output_all['mean1_output'].asnumpy()
    pc = output_all['mean2_output'].asnumpy()
    # [72=4*18,39,38]
    conv_offset_a = output_all['res5a_branch2b_offset_output'].asnumpy()
    conv_offset_b = output_all['res5b_branch2b_offset_output'].asnumpy()
    conv_offset_c = output_all['res5c_branch2b_offset_output'].asnumpy()
    # [42,7,7]
    roipool_offset = output_all['rfcn_cls_offset_output'].asnumpy()

    # a,b,c three layers, 0,1,2,3 four groups
    a0, a1, a2, a3 = conv_offset_a[:, 0:18, :], conv_offset_a[:, 18:36, :], \
                     conv_offset_a[:, 36:54, :], conv_offset_a[:, 54:72, :]
    b0, b1, b2, b3 = conv_offset_b[:, 0:18, :], conv_offset_b[:, 18:36, :], \
                     conv_offset_b[:, 36:54, :], conv_offset_b[:, 54:72, :]
    c0, c1, c2, c3 = conv_offset_c[:, 0:18, :], conv_offset_c[:, 18:36, :], \
                     conv_offset_c[:, 36:54, :], conv_offset_c[:, 54:72, :]
    mshow_dconv_offset(im, [a0, b0, c0], save_name=im_name + '@0')
    mshow_dconv_offset(im, [a1, b1, c1], save_name=im_name + '@1')
    mshow_dconv_offset(im, [a2, b2, c2], save_name=im_name + '@2')
    mshow_dconv_offset(im, [a3, b3, c3], save_name=im_name + '@3')

    return


def show_roipool_offset(im, all_boxes, class_names, output_all, im_name):
    pa = output_all['mean0_output'].asnumpy()
    pb = output_all['mean1_output'].asnumpy()
    pc = output_all['mean2_output'].asnumpy()
    # [72=4*18,39,38]
    conv_offset_a = output_all['res5a_branch2b_offset_output'].asnumpy()
    conv_offset_b = output_all['res5b_branch2b_offset_output'].asnumpy()
    conv_offset_c = output_all['res5c_branch2b_offset_output'].asnumpy()
    # [300,42,7,7]
    roipool_offset = output_all['rfcn_cls_offset_output'].asnumpy()
    #   [300,42,7,7]=>[300*21,2,7,7]
    # rp
    # same :    roipool_offset[0, 2, :, :], rp[1, 0, :, :]
    #           roipool_offset[0, 3, :, :], rp[1, 1, :, :]
    #           roipool_offset[0, 4, :, :], rp[2, 0, :, :]
    #           roipool_offset[1, 0, :, :], rp[21, 0, :, :]
    #           roipool_offset[2, 0, :, :], rp[42, 0, :, :]
    #           roipool_offset[2, 3, :, :], rp[43, 1, :, :]
    rp = roipool_offset.reshape((-1, 2, 7, 7))
    boxes, classes = [], []
    for i, class_name in enumerate(class_names):
        iboxes = all_boxes[i]
        for j in range(iboxes.shape[0]):
            boxes.append(iboxes[j][:4])
            classes.append(i)
    from utils.show_offset import mshow_dpsroi_offset
    mshow_dpsroi_offset(im, boxes, roipool_offset, classes, save_name=im_name)
    return


def main():
    # get symbol
    # pprint.pprint(config)
    # config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
    # sym_instance = eval(config.symbol + '.' + config.symbol)()
    # sym = sym_instance.get_symbol(config, is_train=False)
    sym_instance = resnet_v1_101_rfcn_dcn.resnet_v1_101_rfcn_dcn()
    sym = sym_instance.get_symbol(config, is_train=False, is_demo=True)
    # set up class names
    # num_classes = 21
    classes = ['aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike ',
               'person',
               'pottedplant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor']

    # load demo data
    # image_names = ['COCO_test2015_000000000891.jpg', 'COCO_test2015_000000001669.jpg']
    image_names = ['000057.jpg']
    data = []
    for im_name in image_names:
        im = cv2.imread(cur_path + '/../demo/mdemo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    # data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    # sym_instance.infer_shape(data_shape_dict)

    arg_params, aux_params = load_param(args.model_prefix, args.model_epoch, process=True)
    ##############################################
    conv_kernel = mx.ndarray.array([[1, 1, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 1, 1],
                                    [0, 0, 1, 1, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, -2, -2, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, -2, -2, 0, 0, 1, 1, 0, 0, 0, 0]])
    arg_params['smoothness_penalty_kernel'] = conv_kernel.expand_dims(2).expand_dims(3)

    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up ???
    # for j in xrange(2):
    #     data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
    #                                  provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
    #                                  provide_label=[None])
    #     scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    #     # just output roi from RPN
    #     scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        output_all = predictor.predict(data_batch)[0]
        tic()
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        im = cv2.imread(cur_path + '/../demo/mdemo/' + im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # save_name = args.model_prefix.replace('output/', '').replace('.yaml/2007_trainval_2012_trainval/', '@')
        save_name = 'source'
        show_boxes(im, dets_nms, classes, 1, im_name)
        show_roipool_offset(im, dets_nms, classes, output_all, im_name + save_name)
        # show_conv_offset(output_all, im, im_name)
        print 'done'


if __name__ == '__main__':
    main()
