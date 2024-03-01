from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-seg.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs=300, imgsz=[1080, 1920], device=4, batch=2, lr0=0.001, rect=True)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category