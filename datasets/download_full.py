import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
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


limit = 2500
"""characters = [("aqua_%28konosuba%29","aqua"), ("shirakami_fubuki", "fubuki"), ("makise_kurisu", "makise kurisu"),
              ("megumin", "megumin"), ("rem_%28re%3azero%29", "rem"), ("hakurei_reimu", "reimu hakurei"),
              ("hatsune_miku", "hatsune miku")]"""

characters = [("aqua_%28konosuba%29","aqua"), ("hakurei_reimu", "reimu hakurei")]

def save_image(thumbnail, i, folder_name):
    linkContainer = thumbnail.findChildren("span", recursive=False)[0].findChildren("a", recurisve=False)[0]
    href = linkContainer.attrs["href"]
    href = href[2:]
    response = requests.get("https://" + href)
    sub_html_parsed = BeautifulSoup(response.text, "html.parser")

    right_col = sub_html_parsed.find('div', {'id': 'right-col'})

    final_link = right_col.findChildren("div", recursive=True)[0].findChildren("img", recursive=False)[0].attrs["src"]
    final_response = requests.get(final_link, stream=True)
    final_response.raw.decode_content = True
    image = Image.open(final_response.raw)
    if max(image.size)/min(image.size) <= 2.3:
        width, height = image.size
        max_size = [256, 256]
        if width > max_size[0] or height > max_size[1]:
            image = image.resize(max_size)
        try:
            image = image.convert('RGB')
        except:
            pass
        image.save(folder_name + str(i) + ".jpg", "JPEG", quality=80, optimize=True, progressive=True)


executor = ThreadPoolExecutor(42)
for character in characters:
    i = 0
    if not os.path.exists(character[1]):
        os.makedirs(character[1])

    path, dirs, files = next(os.walk(character[1]))
    file_count = len(files)
    if file_count < 10:
        print("Starting " + character[1])
        while True:
            if i > limit:
                break
            
            r = requests.get(f"https://gelbooru.com/index.php?page=post&s=list&tags=1girl+-1boy+-2boys+-3boys+-4boys+-5boys+-6boys+-6boys+-greyscale+-grayscale+-double_handjob+-buttjob+-handjob+-fellatio+-penis+-close-up+-lower_body+-monochrome+-ass_focus+-asuna_%28sao-alo%29+-copyright_name+-multiple_views+-eyepatch_removed+-from_behind+-double_penetration+-futanari+-futa+-cum+-cum_in_mouth+-multiple_boys+-multiple_girls+-crying+-crying_with_eyes_open+-hole_in_wall+-cosplay+-looking_back+-speech_bubble+-yamcha_pose+-alternative_costume+-alternate_costume+-alternate_costume+-1other+-2others+-3others+-4others+-5others+-japanese_text+-crossover+-tentacle+-tentacles+-sequential+-partial_commentary+-tentacle_sex+-partially_colored+-partial_penetration+-toilet+->o<+-faceless+->_<+-:/+-beastiality+-god+-origami+-lamia+-scales+-chibi+-;d+-meme+-partially_translated+-crossover%2B+{character[0]}&pid=" + str(i))
            parsed_html = BeautifulSoup(r.text, "html.parser")
            thumbnail_container = parsed_html.find('div', {'class': 'thumbnail-container'})
            thumbnails = thumbnail_container.find_all("div", {'class': 'thumbnail-preview poopC'})
            if len(thumbnails) < 5:
                break
            
            for thumbnail in thumbnails:
                executor.submit(save_image, thumbnail, i, character[1] + "/")
                i += 1
                
    print("Done with " + character[1])
