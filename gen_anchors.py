from os import listdir
from os.path import isfile, join
from PIL import Image
# import cv2
import numpy as np
import sys
import os
import math
import xml.etree.ElementTree as ET

box_size = 32  # yolo (given)
cfg_width = 2400
cfg_height = 2400
cfg_classes = 1


'''
Creates 5 anchors: min width, min height, average, max_width, max_height
'''


def getCellDimensions(img_width, img_height):
    # get the dimensions of a cell relative to the image

    width_box_count = cfg_width/box_size
    height_box_count = cfg_height/box_size

    width = img_width / width_box_count
    height = img_height / height_box_count
    return width, height


def getAnchor(img_width, img_height, pixel_box_width, pixel_box_height):
    cell_width, cell_height = getCellDimensions(img_width, img_height)
    anchor_width = pixel_box_width/cell_width
    anchor_height = pixel_box_height/cell_height
    return anchor_width, anchor_height


def getAnchors(annotations_dir, img_dir, output_dir='anchors'):
    # get multiple anchors (width, height) for xml files

    # avg
    width_total = 0
    height_total = 0
    count = 0

    # min
    min_width_dim = [1000, 1000]
    min_height_dim = [1000, 1000]

    # max
    max_width_dim = [0, 0]
    max_height_dim = [0, 0]

    annotations = os.listdir(annotations_dir)
    images = os.listdir(img_dir)

    for filename in annotations:
        # select all xml files from 'annotations directory'
        if filename.endswith('.xml'):
            tree = ET.parse(annotations_dir+'/'+filename)
            root = tree.getroot()

            # find the image width/height from file dimensions
            img_width = 3840
            img_height = 2160

            JPG_filename = filename.replace('.xml', '.JPG')
            jpg_filename = filename.replace('.xml', '.jpg')
            png_filename = filename.replace('.xml', '.png')
            possible_filenames = [JPG_filename, jpg_filename, png_filename]

            for img_filename in images:
                if img_filename in possible_filenames:
                    # get dimensions of image
                    im = Image.open(img_dir+'/'+img_filename)
                    img_width, img_height = im.size
                    break

            # get the total height and width of all bounding boxes
            for child in root:
                if child.tag == 'object':
                    for child in child:
                        if child.tag == 'bndbox':
                            xmin = int(child[0].text)
                            ymin = int(child[1].text)
                            xmax = int(child[2].text)
                            ymax = int(child[3].text)

                            pixel_box_width = xmax-xmin
                            pixel_box_height = ymax-ymin

                            if(pixel_box_width > 0 and pixel_box_height > 0):
                                # normalize to anchors
                                w, h = getAnchor(
                                    img_width, img_height, pixel_box_width, pixel_box_height)

                                # min and max
                                if w < min_width_dim[0]:
                                    min_width_dim = [w, h]
                                elif w > max_width_dim[0]:
                                    max_width_dim = [w, h]
                                if h < min_height_dim[1]:
                                    min_height_dim = [w, h]
                                elif h > max_height_dim[1]:
                                    max_height_dim = [w, h]

                                # avg
                                width_total += w
                                height_total += h
                                count += 1

    # calculate averages
    width_average = width_total / count
    height_average = height_total / count
    avg_dim = [width_average, height_average]

    # create anchors
    anchors = [min_width_dim, min_height_dim,
               avg_dim, max_width_dim, max_height_dim]
    return np.asarray(anchors).flatten()


def main(argv):
    anchors = getAnchors('annotations', 'images')

    # print anchors
    anchors_str = ""
    for anchor in anchors[:-1]:
        anchors_str += str(anchor) + ', '
    anchors_str += str(anchors[-1])
    print(anchors_str)

    # print details
    num_anchors = int(len(anchors)/2)
    print('num:', num_anchors)
    num_filters = int(num_anchors*(cfg_classes+5))
    print('filters:', num_filters)


if __name__ == "__main__":
    main(sys.argv)
