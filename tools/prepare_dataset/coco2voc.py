import os

from pascal_voc_writer import Writer
from pycocotools.coco import COCO


def coco2voc(ann_file, output_dir):
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']
    for img in coco.imgs:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)
        if len(annIds) > 0:
            img_fname = coco.imgs[img]['file_name']
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            writer = Writer(img_fname, coco.imgs[img]['width'],
                            coco.imgs[img]['height'])
            anns = coco.loadAnns(annIds)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, bbox[0], bbox[1], bbox[2], bbox[3])
                writer.save(output_dir + '/' + label_fname)


output_dir = 'data'
ann_file = r'/data/coco/annotations/instances_train.json'
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

coco2voc(ann_file=ann_file, output_dir=output_dir)
