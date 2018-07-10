from src.data.coco import COCODet
imdb = COCODet('/mnt/dataset/coco', 'person_val2017')

imdb._do_detection_eval('/mnt/ckpt/pytorchSSD/RefineDet_Renset/v1/validation/detections_person_val2017_results.json', './')
