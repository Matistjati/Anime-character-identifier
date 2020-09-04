import sys
sys.path.insert(0,"anime-face-detector")
import numpy as np
import cv2
from faster_rcnn_wrapper import FasterRCNNSlim
from _tf_compat_import import compat_tensorflow as tf
import argparse
import os
import json
import time
from nms_wrapper import NMSType, NMSWrapper
import requests
import io
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

nms_type = NMSType.CPU_NMS


nms = NMSWrapper(nms_type)

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.Session(config=cfg)

net = FasterRCNNSlim()
saver = tf.train.Saver()

saver.restore(sess, "anime-face-detector/model/res101_faster_rcnn_iter_60000.ckpt")

def detect(sess, rcnn_cls, image):
    # pre-processing image for Faster-RCNN
    img_origin = image.astype(np.float32, copy=True)
    img_origin -= np.array([[[102.9801, 115.9465, 112.7717]]])

    img_shape = img_origin.shape
    img_size_min = np.min(img_shape[:2])
    img_size_max = np.max(img_shape[:2])

    img_scale = 600 / img_size_min
    if np.round(img_scale * img_size_max) > 1000:
        img_scale = 1000 / img_size_max
    img = cv2.resize(img_origin, None, None, img_scale, img_scale, cv2.INTER_LINEAR)
    img_info = np.array([img.shape[0], img.shape[1], img_scale], dtype=np.float32)
    img = np.expand_dims(img, 0)

    # test image
    _, scores, bbox_pred, rois = rcnn_cls.test_image(sess, img, img_info)

    # bbox transform
    boxes = rois[:, 1:] / img_scale

    boxes = boxes.astype(bbox_pred.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = bbox_pred[:, 0::4]
    dy = bbox_pred[:, 1::4]
    dw = bbox_pred[:, 2::4]
    dh = bbox_pred[:, 3::4]
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros_like(bbox_pred, dtype=bbox_pred.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    # clipping edge
    pred_boxes[:, 0::4] = np.maximum(pred_boxes[:, 0::4], 0)
    pred_boxes[:, 1::4] = np.maximum(pred_boxes[:, 1::4], 0)
    pred_boxes[:, 2::4] = np.minimum(pred_boxes[:, 2::4], img_shape[1] - 1)
    pred_boxes[:, 3::4] = np.minimum(pred_boxes[:, 3::4], img_shape[0] - 1)
    return scores, pred_boxes



def save_image_cropped(link, file_name):
    nms_thresh = 0.3
    conf_thresh = 0.8
    crop_width = 256
    crop_height = 256
    start_output_number = 0
    extra_size = 0.069

    result = {}
    file = requests.get(link).content

    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

    # CV
    #img = cv.CreateImageHeader((img_np.shape[1], img_np.shape[0]), cv.IPL_DEPTH_8U, 3)
    #cv.SetData(img, img_np.tostring(), img_np.dtype.itemsize * 3 * img_np.shape[1])
    #img = cv2.imread(img_np)
    scores, boxes = detect(sess, net, img)
    boxes = boxes[:, 4:8]
    scores = scores[:, 1]
    keep = nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), nms_thresh)
    boxes = boxes[keep, :]
    scores = scores[keep]
    inds = np.where(scores >= conf_thresh)[0]
    scores = scores[inds]
    boxes = boxes[inds, :]

    result[file] = []

    if scores.shape[0] > 1:
        return

    for i in range(scores.shape[0]):
        x1, y1, x2, y2 = boxes[i, :].tolist()
        new_result = {'score': float(scores[i]),
                      'bbox': [x1, y1, x2, y2]}
        result[file].append(new_result)


        #print("Saving")
        #print(x1,x2,y1,y2)

        #print(img.shape)

        min_side = min(img.shape[0], img.shape[1])
        x1 -= min_side * extra_size
        x1 = max(0, x1)
        x2 += min_side * extra_size
        x2 = min(min_side, x2)

        y1 -= min_side * extra_size
        y1 = max(0, y1)
        y2 += min_side * extra_size
        y2 = min(min_side, y2)


        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]

        cropped_image = cv2.resize(cropped_image,
                                  (crop_width, crop_height),
                                  interpolation = cv2.INTER_AREA)

        cv2.imwrite(file_name, cropped_image)
        start_output_number += 1


def save_image(thumbnail, i, folder_name):
    linkContainer = thumbnail.findChildren("span", recursive=False)[0].findChildren("a", recurisve=False)[0]
    href = linkContainer.attrs["href"]
    href = href[2:]
    response = requests.get("https://" + href)
    sub_html_parsed = BeautifulSoup(response.text, "html.parser")

    right_col = sub_html_parsed.find('div', {'id': 'right-col'})

    final_link = right_col.findChildren("div", recursive=True)[0].findChildren("img", recursive=False)[0].attrs["src"]
    startIndex = href.find("id=") + 3
    name = ""
    for o in range(20):
        n = href[o+startIndex]
        if n.isdigit():
            name += n
        else:
            break
    save_image_cropped(final_link, folder_name + name + ".jpg")

def get_dataset_path(character):
    return f"datasets/{dataset_name}/{character[1]}/"

limit = 700

dataset_name = "aqua reimu face"
"""characters = [("aqua_%28konosuba%29","aqua"), ("shirakami_fubuki", "fubuki"), ("makise_kurisu", "makise kurisu"),
              ("megumin", "megumin"), ("rem_%28re%3azero%29", "rem"), ("hakurei_reimu", "reimu hakurei"),
              ("hatsune_miku", "hatsune miku")]"""

#characters = [("hatsune_miku", "hatsune miku"),("shirakami_fubuki", "fubuki"), ("cirno", "cirno")]
characters = [("hatsune_miku", "hatsune miku")]
extra_tags = "blue_hair"


executor = ThreadPoolExecutor(limit)
for character in characters:
    i = 350
    if not os.path.exists(get_dataset_path(character)):
        os.makedirs(get_dataset_path(character))

    path, dirs, files = next(os.walk(get_dataset_path(character)))
    file_count = len(files)
    if file_count < 10:
        print("Starting " + character[1])
        while True:
            if i > limit:
                break

            r = requests.get(f"https://gelbooru.com/index.php?page=post&s=list&tags=1girl+-1boy+-2boys+-3boys+-4boys+-5boys+-6boys+-6boys+-greyscale+-grayscale+-double_handjob+-buttjob+-vore+-handjob+-fellatio+-picture_(object)+-character_doll+-pov_feet+-:i+-choroidoragon+-explosion+-3d+-fake_cover+-pow+-artist_name+-satou_kazuma+-penis+-close-up+-lower_body+-monochrome+-ass_focus+-asuna_%28sao-alo%29+-copyright_name+-multiple_views+-eyepatch_removed+-from_behind+-double_penetration+-futanari+-futa+-cum+-cum_in_mouth+-multiple_boys+-multiple_girls+-crying+-crying_with_eyes_open+-hole_in_wall+-cosplay+-looking_back+-speech_bubble+-yamcha_pose+-alternative_costume+-alternate_costume+-alternate_costume+-1other+-2others+-3others+-4others+-5others+-japanese_text+-crossover+-tentacle+-tentacles+-sequential+-partial_commentary+-tentacle_sex+-partially_colored+-partial_penetration+-toilet+->o<+-faceless+->_<+-:/+-beastiality+-god+-origami+-head_out_of_frame+-lamia+-scales+-chibi+-;d+-meme+-partially_translated+-crossover%2B+{extra_tags}+{character[0]}&pid=" + str(i))
            parsed_html = BeautifulSoup(r.text, "html.parser")
            thumbnail_container = parsed_html.find('div', {'class': 'thumbnail-container'})
            thumbnails = thumbnail_container.find_all("div", {'class': 'thumbnail-preview poopC'})
            if len(thumbnails) < 5:
                break

            for thumbnail in thumbnails:
                executor.submit(save_image, thumbnail, i, get_dataset_path(character))
                #save_image(thumbnail, i, character[1] + "/")
                i += 1


    print("Done with " + character[1])
