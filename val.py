from ultralytics import YOLO

# Load a model
model = YOLO('models/yolov8s_seg_1920x1080.pt')

# Validate the model
metrics = model.val(data='data.yaml', device=4, batch=2)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category