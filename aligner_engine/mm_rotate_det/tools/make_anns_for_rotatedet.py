import os
import csv
import tqdm
from tkinter import filedialog
import shutil
import xml.etree.ElementTree as etree
from xml.etree import ElementTree

SOURCE_IMAGES_PATH = "D:\\__sample_dataset_skrew_spread\\Images_xmls"
TARGET_ANNS_PATH = "D:\\__sample_dataset_skrew_spread\\annfiles"


def scan_all_images(folder_path):
    extensions = ["jpg"]

    images = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "_mask.png" in file:
                continue
            if file.lower().endswith(tuple(extensions)):
                relativePath = root.replace("\\", '/') + '/' + file
                images.append(relativePath)
    images.sort(key=lambda x: x.lower())
    return images


def main():
    images = scan_all_images(SOURCE_IMAGES_PATH)
    for img_path in tqdm.tqdm(images):
        xml_path = img_path[:-4] + '.xml'
        img_name = img_path.split('/')[-1]
        label_path = os.path.join(TARGET_ANNS_PATH, img_name[:-4] + '.txt')

        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(xml_path, parser=parser).getroot()

        anns = []
        for object_iter in xmltree.findall('object'):
            typeItem = object_iter.find('type')


            if typeItem.text == 'robndbox':
                robndbox = object_iter.find('robndbox')
                label = object_iter.find('name').text

                x1 = robndbox.find('x1').text
                y1 = robndbox.find('y1').text

                x2 = robndbox.find('x2').text
                y2 = robndbox.find('y2').text

                x3 = robndbox.find('x3').text
                y3 = robndbox.find('y3').text

                x4 = robndbox.find('x4').text
                y4 = robndbox.find('y4').text

                anns.append(x1 + ' ' + y1 + ' ' +
                            x2 + ' ' + y2 + ' ' +
                            x3 + ' ' + y3 + ' ' +
                            x4 + ' ' + y4 + ' ' +
                            label + ' ' + '0')

        # make file
        f = open(label_path, 'w')
        for ann in anns:
            f.write(ann)
        f.close()


if __name__ == "__main__":
    main()