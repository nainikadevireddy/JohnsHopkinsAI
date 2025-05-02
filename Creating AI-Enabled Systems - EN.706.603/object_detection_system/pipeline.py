import cv2
import os
import json

from src.inference import object_detection
from src.inference import non_maximal_suppression
from src.inference import video_processing
from src.rectification import hard_negative_mining
import src.metrics as metrics



class InferenceService:
    def __init__(self, stream, detector, nms, output_format='YOLO', save_dir='storages/prediction'):
        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.frame_counter = 0
        self.output_format = output_format
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def detect(self, frame):
        # Detect objects in the frame
        detections = self.detector.predict(frame)
        detections = self.detector.process_output(detections)

        # Apply non-maximal suppression
        detections = self.nms.filter(detections)

        return detections

    def _save(self, frame, detections):
        frame_filename = os.path.join(self.save_dir, f'frame_{self.frame_counter}.jpg')
        frame = self.detector.draw_labels(frame, detections)
        cv2.imwrite(frame_filename, frame)

        if self.output_format == 'YOLO':
            self._save_yolo_format(frame_filename, detections)
        elif self.output_format == 'PASCAL':
            self._save_pascal_format(frame_filename, detections)
        elif self.output_format == 'COCO':
            self._save_coco_format(frame_filename, detections)

    def _save_yolo_format(self, frame_filename, detections):
        detection_filename = frame_filename.replace('.jpg', '.txt')
        with open(detection_filename, 'w') as f:
            for class_id, confidence, box in zip(*detections):
                x_center = (box[0] + box[2] / 2) / self.width
                y_center = (box[1] + box[3] / 2) / self.height
                width = box[2] / self.width
                height = box[3] / self.height
                f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

    def _save_pascal_format(self, frame_filename, detections):
        detection_filename = frame_filename.replace('.jpg', '.xml')
        with open(detection_filename, 'w') as f:
            f.write('<annotation>\n')
            f.write(f'    <filename>{os.path.basename(frame_filename)}</filename>\n')
            for class_id, confidence, box in zip(*detections):
                f.write('    <object>\n')
                f.write(f'        <name>{self.detector.classes[class_id]}</name>\n')
                f.write(f'        <bndbox>\n')
                f.write(f'            <xmin>{box[0]}</xmin>\n')
                f.write(f'            <ymin>{box[1]}</ymin>\n')
                f.write(f'            <xmax>{box[0] + box[2]}</xmax>\n')
                f.write(f'            <ymax>{box[1] + box[3]}</ymax>\n')
                f.write(f'        </bndbox>\n')
                f.write('    </object>\n')
            f.write('</annotation>\n')

    def _save_coco_format(self, frame_filename, detections):
        detection_filename = frame_filename.replace('.jpg', '.json')
        coco_annotations = []
        for class_id, confidence, box in zip(*detections):
            annotation = {
                "category_id": class_id,
                "bbox": [box[0], box[1], box[2], box[3]],
                "score": confidence
            }
            coco_annotations.append(annotation)
        with open(detection_filename, 'w') as f:
            json.dump(coco_annotations, f, indent=4)
