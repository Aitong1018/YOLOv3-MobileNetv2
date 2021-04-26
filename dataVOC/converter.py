from PIL import Image
import os, glob
import datetime
import shutil
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']

CLASSES = ['soccer-ball-field','helicopter','swimming-pool','roundabout','large-vehicle',
           'small-vehicle','bridge','harbor','ground-track-field','basketball-court','tennis-court',
           'basketball-diamond','storage-tank','ship','plane']
running_from_path = os.getcwd()
created_images_dir = 'images'
created_labels_dir = 'labels'
# data_dir = 'dataVOC'
data_dir = 'data'

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def generate_dir(set_name, root_path):
    images_dir = os.path.join(root_path, 'JPEGImages')
    annotation_dir = os.path.join(root_path, 'Annotations')

    new_images_dir = os.path.join(created_images_dir, set_name)
    new_annotation_dir = os.path.join(created_labels_dir, set_name)

    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)

    if not os.path.exists(new_annotation_dir):
        os.makedirs(new_annotation_dir)

    for img in glob.glob(os.path.join(images_dir, "*.jpg")):
        shutil.copy(img, new_images_dir)

    os.chdir(annotation_dir)
    xml_annotations = glob.glob("*.xml")
    os.chdir(running_from_path)

    for xml in xml_annotations:
        filename = xml.split(".")[0]
        print(filename)
        xml_path = os.path.join(annotation_dir, xml)
        pil_image = Image.open(os.path.join(images_dir, filename+".jpg"))
        width, height = pil_image.size
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')
        with open(os.path.join(new_annotation_dir, filename + ".txt"), "w") as hs:
            for box_idx,obj in enumerate(objects):
                difficult = obj.find('difficult').text.strip()
                if (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = CLASSES.index(obj.find('name').text.lower().strip())
                xmin = float(bbox.find('xmin').text.strip())
                xmax = float(bbox.find('xmax').text.strip())
                ymin = float(bbox.find('ymin').text.strip())
                ymax = float(bbox.find('ymax').text.strip())

                maxX = min(xmax, width-1)
                minX = max(xmin, 0)
                maxY = min(ymax, height-1)
                minY = max(ymin, 0)


                norm_width = (maxX - minX) / width

                norm_height = (maxY - minY) / height

                center_x, center_y = (maxX + minX) / 2, (maxY + minY) / 2

                norm_center_x = center_x / width
                norm_center_y = center_y / height

                if box_idx != 0:
                    hs.write("\n")

                hs.write("%s %f %f %f %f" % (str(class_ind), norm_center_x, norm_center_y, norm_width, norm_height))

def create_txt(dirlist, filename):
    with open(filename, "w") as txtfile:
        imglist = []

        for dir in dirlist:
            imglist.extend(glob.glob(os.path.join(dir, "*.jpg")))

        for idx, img in enumerate(imglist):
            if idx != 0:
                txtfile.write("\n")
            txtfile.write(os.path.join(data_dir, img))

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    generate_dir("train2007", "VOCROOT/VOC2007")
    generate_dir("train2012", "VOCROOT/VOC2012")
    generate_dir("validation", "VOCROOT/VOC2007TEST")

    create_txt((os.path.join(created_images_dir, 'train2007'),
                os.path.join(created_images_dir, 'train2012')),
               'train.txt')
    create_txt((os.path.join(created_images_dir, 'validation'), ),
               'valid.txt')

    end_time = datetime.datetime.now()
    seconds_elapsed = (end_time - start_time).total_seconds()
    print("It took {} to execute this".format(hms_string(seconds_elapsed)))
