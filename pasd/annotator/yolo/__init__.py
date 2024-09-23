from ultralytics import YOLO

from annotator.util import annotator_path

class YoLoDetection(object):
    def __init__(self, device='cuda'):
        MODEL =  f'{annotator_path}/ckpts/yolov8n.pt'
        self.model = YOLO(MODEL)
    
    def detect(self, image, imgsz=640):
        results = self.model.predict(image, imgsz=imgsz, verbose=False, save=False)
        for result in results[:1]:
            boxes = result.boxes  # Boxes object for bbox outputs
            cls = (boxes.cls).cpu().numpy()
            conf = (boxes.conf).cpu().numpy()
            names = result.names
        return cls, conf, names
        
