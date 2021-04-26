import numpy as np
import albumentations as A


def horizontal_flip(images, boxes):
    images = np.flip(images, [-1])
    boxes[:, 1] = 1 - boxes[:, 1]
    return images, boxes


def augment(image, boxes):
    h, w, _ = image.shape
    labels, boxes_coord = boxes[:, 0], boxes[:, 1:]
    labels = labels.tolist()
    boxes_coord = boxes_coord * h     
    boxes_coord[:, 0] = np.clip(boxes_coord[:, 0]-boxes_coord[:, 2]/2, a_min=0, a_max=None)  
    boxes_coord[:, 1] = np.clip(boxes_coord[:, 1]-boxes_coord[:, 3]/2, a_min=0, a_max=None)
    boxes_coord = boxes_coord.tolist()
    aug = A.Compose([A.HorizontalFlip(p=0.5),], bbox_params={'format':'coco', 'label_fields': ['category_id']})
    augmented = aug(image=image, bboxes=boxes_coord, category_id=labels)
    if augmented['bboxes']:
        image = augmented['image']
        boxes_coord = np.array(augmented['bboxes'])
        boxes_coord[:, 0] = boxes_coord[:, 0] + boxes_coord[:, 2]/2
        boxes_coord[:, 1] = boxes_coord[:, 1] + boxes_coord[:, 3]/2
        boxes_coord = boxes_coord / h
        labels = np.array(augmented['category_id'])[:, None]
        boxes = np.concatenate((labels, boxes_coord), 1)

    return image, boxes
