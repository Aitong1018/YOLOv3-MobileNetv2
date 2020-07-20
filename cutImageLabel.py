import os
import cv2
import PIL
import xml.etree.ElementTree as ET
import math

from xml.dom.minidom import Document
category_set = ['plane','small-vehicle', 'large-vehicle', 'ship',
'tennis-court','storage-tank','harbor','swimming-pool', ]

def WriteError(str):
    out_txt = 'D:/毕业设计/YOLOv3-Mobilenetv2-DOTA/程序状态日志.txt'
    with open(out_txt, 'a') as w:
      w.write('----------------------------------' + '\n')
      w.writelines(str + '\n')

def custombasename(fullname):
  return os.path.basename(os.path.splitext(fullname)[0])

def limit_value(a, b):
  if a < 1:
    a = 1
  if a >= b:
    a = b - 1
  return a
def computecordinate(xx1,xx2,yy1,yy2,height, width,dxmin,dxmax,dymin,dymax):
    if xx1 >=dxmin:
        xx1 = xx1-dxmin
    elif xx1 < dxmin:
        xx1 = 0
    if yy1 >=dymin:
        yy1 = yy1-dymin
    elif yy1 < dymin:
        yy1 = 0
    if xx2 >= dxmax:
        xx2 = width
    elif xx2 < dxmax:
        xx2 = xx2-dxmin
    if yy2 >= dymax:
        yy2 = height
    elif yy2 < dymax:
        yy2 = yy2-dymin
    return xx1, xx2, yy1, yy2

def readlabeltxt(filename, txtpath, height, width, dxmin,dxmax,dymin,dymax,hbb=True):
  #print(txtpath)

  with open(txtpath, 'r') as f_in:  # 打开txt文件
    lines = f_in.readlines()
    splitlines = [x.strip().split(' ') for x in lines]  # 根据空格分割
    boxes = []
    for i, splitline in enumerate(splitlines):
      if i in [0, 1, 2, 3]:  # DOTA数据集前两行对于我们来说是无用的,不知道为啥前两行总读出空格，没办法只能跳过了
        continue
      label = splitline[8]
      if label not in category_set:  # 只书写指定的类别
        continue
      x1 = int(float(splitline[0]))
      y1 = int(float(splitline[1]))
      x2 = int(float(splitline[2]))
      y2 = int(float(splitline[3]))
      x3 = int(float(splitline[4]))
      y3 = int(float(splitline[5]))
      x4 = int(float(splitline[6]))
      y4 = int(float(splitline[7]))
      # 如果是hbb
      if hbb:
        xx1 = min(x1, x2, x3, x4)
        xx2 = max(x1, x2, x3, x4)
        yy1 = min(y1, y2, y3, y4)
        yy2 = max(y1, y2, y3, y4)

        #xx1 = limit_value(xx1, width)
        #xx2 = limit_value(xx2, width)
        #yy1 = limit_value(yy1, height)
        #yy2 = limit_value(yy2, height)
        #比对坐标大小是否在裁剪图像范围内,算出框在裁剪图为基准的坐标
        #判断两个矩形是否相交
        wx = xx2 - xx1
        wy = yy2 - yy1
        if xx1 + wx > dxmin and dxmin + width > xx1 and yy1 + wy > dymin and dymin + height > yy2:
            xx1, xx2, yy1, yy2 = computecordinate(xx1,xx2,yy1,yy2, height, width, dxmin, dxmax, dymin, dymax)
            #计算边缘被截取的部分面积占比决定是否留下来
            Acut = (xx2 - xx1)*(yy2 - yy1)
            A = wx*wy
            P = Acut/A
            if P > 0.7:
                box = [xx1, yy1, xx2, yy2, label]
                boxes.append(box)
      #obb形式
      else:
          xx1 = x1
          yy1 = y1
          xx2 = x2
          yy2 = y2
          xx3 = x3
          yy3 = y3
          xx4 = x4
          yy4 = y4
          #判断是否和裁剪图像相交
          wx = xx2-xx1
          wy = yy2-yy1
          xx_min = min(x1, x2, x3, x4)
          yy_min = min(y1, y2, y3, y4)
          yy_max = max(y1, y2, y3, y4)
          if xx_min + wx > dxmin and dxmin + width > xx_min and yy_min + wy > dymin and dymin + height > yy_max:
              box = [xx1, yy1, xx2, yy2, xx3, yy3, xx4, yy4, label]
          boxes.append(box)

  return boxes

def writeXml(tmp, imgname, w, h, d, bboxes, hbb=True):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
      # threes#
      object_new = doc.createElement("object")
      annotation.appendChild(object_new)

      name = doc.createElement('name')
      object_new.appendChild(name)
      name_txt = doc.createTextNode(str(bbox[-1]))
      name.appendChild(name_txt)

      pose = doc.createElement('pose')
      object_new.appendChild(pose)
      pose_txt = doc.createTextNode("Unspecified")
      pose.appendChild(pose_txt)

      truncated = doc.createElement('truncated')
      object_new.appendChild(truncated)
      truncated_txt = doc.createTextNode("0")
      truncated.appendChild(truncated_txt)

      difficult = doc.createElement('difficult')
      object_new.appendChild(difficult)
      difficult_txt = doc.createTextNode("0")
      difficult.appendChild(difficult_txt)
      # threes-1#
      bndbox = doc.createElement('bndbox')
      object_new.appendChild(bndbox)

      if hbb:
        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(bbox[0]))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(bbox[1]))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(bbox[2]))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(bbox[3]))
        ymax.appendChild(ymax_txt)
      else:
        x0 = doc.createElement('x0')
        bndbox.appendChild(x0)
        x0_txt = doc.createTextNode(str(bbox[0]))
        x0.appendChild(x0_txt)

        y0 = doc.createElement('y0')
        bndbox.appendChild(y0)
        y0_txt = doc.createTextNode(str(bbox[1]))
        y0.appendChild(y0_txt)

        x1 = doc.createElement('x1')
        bndbox.appendChild(x1)
        x1_txt = doc.createTextNode(str(bbox[2]))
        x1.appendChild(x1_txt)

        y1 = doc.createElement('y1')
        bndbox.appendChild(y1)
        y1_txt = doc.createTextNode(str(bbox[3]))
        y1.appendChild(y1_txt)

        x2 = doc.createElement('x2')
        bndbox.appendChild(x2)
        x2_txt = doc.createTextNode(str(bbox[4]))
        x2.appendChild(x2_txt)

        y2 = doc.createElement('y2')
        bndbox.appendChild(y2)
        y2_txt = doc.createTextNode(str(bbox[5]))
        y2.appendChild(y2_txt)

        x3 = doc.createElement('x3')
        bndbox.appendChild(x3)
        x3_txt = doc.createTextNode(str(bbox[6]))
        x3.appendChild(x3_txt)

        y3 = doc.createElement('y3')
        bndbox.appendChild(y3)
        y3_txt = doc.createTextNode(str(bbox[7]))
        y3.appendChild(y3_txt)

    xmlname = os.path.splitext(imgname)[0]
    tempfile = os.path.join(tmp, xmlname + '.xml')
    with open(tempfile, 'wb') as f:
      f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

if __name__ == '__main__':
  imagefile = './DOTA_dataset/DOTA_val/images'
  labelfile = './DOTA-v1.5_val_hbb'
  cut_labelfile = './xml_cut/val'
  cut_imagefile = './images_cut/val'
  images = os.listdir(imagefile)
  labels = os.listdir(labelfile)
  #这些坐标都是在以大图为基准的坐标，要求的是以裁剪图的坐标为基准的框的坐标。
  W = 416  # 裁剪图的长宽
  H = 416

  xmin = 0;ymin=0 #标注框的左上角坐标
  xmax=0;ymax=0#标注框的右下角坐标
  for image_name in images:
    dxmin = 0;dymin = 0 #裁剪图像的左上角坐标
    dxmax = W;dymax = H #裁剪图像的右下角坐标
    #读取图片和图片名称
    image_name_without_extention = os.path.splitext(image_name)[0]
    image = cv2.imread(imagefile + '/' + image_name)
    image_w,image_h,_ = image.shape
    # 找到同名的xml文件
    label_path = labelfile + '/' + image_name_without_extention + '.txt'
    count = 1#记录裁剪出几个小图像
    if not os.path.exists(label_path):
      WriteError(image_name_without_extention + ' 找不到对应的标签数据！！！')
    for i in range(image_h//H):
      dymin = 0
      dymax = H
      for j in range(image_w//W):
        # 以原图像的右上角为坐标原点
        image_cut = image[dymin:dymax,dxmin:dxmax,:]
        cut_image_name = image_name_without_extention + '_'+str(count) + '.png'
        count += 1
        h, w, d = image_cut.shape
        boxes = readlabeltxt(cut_image_name,label_path,h,w,dxmin,dxmax,dymin,dymax,hbb=True)#读原始data的txt,返回在裁剪范围内的框框。
        if len(boxes) == 0:
          print('裁剪图范围内没有标签物体！！！', label_path)
        else:

            #忽略切割完没有目标的图片，把得到的标签写入新的xml或txt中。
            writeXml(cut_labelfile, cut_image_name, h, w, d, boxes, hbb=True)
            cv2.imwrite(cut_imagefile + '/' + cut_image_name, image_cut)
            dymin = dymax
            dymax = dymax + H
        # os.remove(os.path.join(cut_image_name))
        # os.remove(os.path.join(label_path))


      dxmin = dxmax
      dxmax = dxmax + W














