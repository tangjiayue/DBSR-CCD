from tidecv import TIDE, datasets

tide = TIDE()

tide.evaluate_range(
    datasets.COCO("/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cocolike-cat10.json"),
    datasets.COCOResult("predictions.json"),
    mode=TIDE.BOX
)

tide.summarize()