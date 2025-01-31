from ultralytics import YOLO

#loading the model
model = YOLO("yolov9t_trained_best.pt")

# predicting the result
result= model(source="test7.png",conf = 0.5)

# displaying the result
result[0].show()
print(result[0])