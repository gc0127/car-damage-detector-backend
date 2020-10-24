import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from cv2 import cv2
import os
from shapely.geometry import Polygon
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


car_parts_mapping = { 0: 'Bonnet', 1: 'Bumper', 2: 'Light', 3: 'Windshield'}
car_damage_mapping = { 0: 'Broken', 1: 'Dent on', 2: 'Scratches on' }
damage_severity_mapping = { 0:'High', 1:'Low', 2:'Medium' }


def getPolygon(bounding_box_coordinates):
  x1,y1,x2,y2 = bounding_box_coordinates
  return Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

def getIntersectionRatio(part_bounding_box, damage_bounding_box):
  part_rectangle = getPolygon(part_bounding_box)
  damage_rectangle = getPolygon(damage_bounding_box)
  intersection = part_rectangle.intersection(damage_rectangle)
  return intersection.area/damage_rectangle.area
  
def get_union_bboxes(bounding_box1, bounding_box2):
  rectangle1 = getPolygon(bounding_box1)
  rectangle2 = getPolygon(bounding_box2)
  union = rectangle1.union(rectangle2)
  return union.bounds

def get_intersection(bounding_box1, bounding_box2):
  rectangle1 = getPolygon(bounding_box1)
  rectangle2 = getPolygon(bounding_box2)
  intersection = rectangle1.intersection(rectangle2)
  return intersection.area
  
def group_by_class(prediction_outputs, damage_part_classes):
  output_mapping = {}
  bounding_boxes = prediction_outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
  classes = prediction_outputs["instances"].to("cpu").pred_classes.numpy()
  print(bounding_boxes)
  print(classes)
  for index in range(len(bounding_boxes)):
    if damage_part_classes[index] in output_mapping:
      output_mapping[damage_part_classes[index]].append((bounding_boxes[index], classes[index]))
    else:
      output_mapping[damage_part_classes[index]] = [(bounding_boxes[index], classes[index])]
  print(output_mapping)
  return output_mapping
  
def filter_predictions(prediction_outputs, damage_part_classes):
  output_mapping = group_by_class(prediction_outputs, damage_part_classes)
  filtered_bboxes = []
  filtered_damage_classes = []
  filtered_part_classes = []
  for part in output_mapping:
    print(output_mapping[part])
    damages = output_mapping[part]
    i = 0
    while i < len(damages):
      j = i+1
      while j < len(damages):
        if damages[i][1] == damages[j][1] and get_intersection(damages[i][0], damages[j][0]) > 0:
          damages[i] = (get_union_bboxes(damages[i][0], damages[j][0]), damages[i][1])
          del damages[j]
        else:
          j = j + 1
      i = i + 1
    bboxes = [x[0] for x in damages]
    classes = [x[1] for x in damages]
    filtered_bboxes.extend(bboxes)
    filtered_damage_classes.extend(classes)
    filtered_part_classes.extend([part]*len(bboxes))
    print(filtered_bboxes)
    print(filtered_damage_classes)
    print(filtered_part_classes)
  return filtered_bboxes, filtered_damage_classes, filtered_part_classes

def getDamagePartClasses(car_parts_outputs, damage_bounding_boxes):
  part_bounding_boxes = car_parts_outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
  part_classes = car_parts_outputs["instances"].to("cpu").pred_classes.numpy()
  damage_part_classes = []
  for damage_bounding_box in damage_bounding_boxes:
    max_intersection = -1
    max_intersection_class = -1
    for index in range(len(part_bounding_boxes)):
      part_bounding_box = part_bounding_boxes[index]
      intersection = getIntersectionRatio(part_bounding_box, damage_bounding_box)
      if intersection > 0 and intersection > max_intersection:
          max_intersection = intersection
          max_intersection_class = part_classes[index]
    damage_part_classes.append(max_intersection_class)
  return damage_part_classes

def getDamageSeverityClasses(damage_bounding_boxes, damage_severity_outputs):
  severity_bounding_boxes = damage_severity_outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
  severity_classes = damage_severity_outputs["instances"].to("cpu").pred_classes.numpy()
  damage_severity_classes = []
  for damage_bounding_box in damage_bounding_boxes:
    max_intersection = -1
    max_intersection_class = -1
    for index in range(len(severity_bounding_boxes)):
      severity_bounding_box = severity_bounding_boxes[index]
      intersection = getIntersectionRatio(severity_bounding_box, damage_bounding_box)
      if intersection > 0 and intersection > max_intersection:
          max_intersection = intersection
          max_intersection_class = severity_classes[index]
    damage_severity_classes.append(max_intersection_class)
  return damage_severity_classes

def init_model():
    # model for detecting the parts of car
    car_parts_cfg = get_cfg()
    car_parts_cfg.MODEL.DEVICE = 'cpu'
    car_parts_cfg.merge_from_file("static/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    car_parts_cfg.MODEL.WEIGHTS = os.path.join("static/savedmodels/", "car_parts_model_new.pth")
    car_parts_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   
    car_parts_cfg.DATASETS.TEST = ("car_parts", )
    car_parts_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    car_parts_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 3 classes (bumper, bonnet, light, windshield)
    car_parts_predictor = DefaultPredictor(car_parts_cfg)

    # model for finding out the damage in the car
    car_damage_cfg = get_cfg()
    car_damage_cfg.MODEL.DEVICE = 'cpu'
    car_damage_cfg.merge_from_file("static/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    car_damage_cfg.MODEL.WEIGHTS = os.path.join("static/savedmodels/", "car_damage_model_new.pth")
    car_damage_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    car_damage_cfg.DATASETS.TEST = ("car_damage", )
    car_damage_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    car_damage_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (scratch, dent, broken)
    car_damage_predictor = DefaultPredictor(car_damage_cfg)

    # model for finding out the severity of the damage(Low,Medium, High)
    damage_severity_cfg = get_cfg()
    damage_severity_cfg.MODEL.DEVICE = 'cpu'
    damage_severity_cfg.merge_from_file("static/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    damage_severity_cfg.MODEL.WEIGHTS = os.path.join("static/savedmodels/", "damage_severity_model.pth")
    damage_severity_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    damage_severity_cfg.DATASETS.TEST = ("damage_severity", )
    damage_severity_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    damage_severity_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (low, medium, high)
    damage_severity_predictor = DefaultPredictor(damage_severity_cfg)
    return car_parts_predictor, car_damage_predictor, damage_severity_predictor

def predict(image):
  car_parts_outputs = car_parts_predictor(image)
  car_damage_outputs = car_damage_predictor(image)
  damage_severity_outputs = damage_severity_predictor(image) 
  return car_parts_outputs, car_damage_outputs, damage_severity_outputs
  
def getDefectsInfo(image):
  car_parts_outputs, car_damage_outputs, damage_severity_outputs = predict(image)
  damage_bbox = car_damage_outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
  damage_part_classes = getDamagePartClasses(car_parts_outputs, damage_bbox)
  damage_bbox, damage_classes, damage_part_classes = filter_predictions(car_damage_outputs, damage_part_classes)
  damage_severity_classes = getDamageSeverityClasses(damage_bbox, damage_severity_outputs)
  print(damage_severity_classes)
  result = []
  for index in range(len(damage_part_classes)):
    if damage_severity_classes[index] != -1 and damage_part_classes[index] != -1:
      result.append(
        {
          'part': car_parts_mapping[damage_part_classes[index]],
          'type': car_damage_mapping[damage_classes[index]],
          'severity': damage_severity_mapping[damage_severity_classes[index]],
          #'confidence': damage_scores[index]*100,
          'bbox': damage_bbox[index]
        }
      )
  return result

car_parts_predictor, car_damage_predictor, damage_severity_predictor = init_model()