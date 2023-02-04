import cv2
import matplotlib.pyplot as plt
config_file='D:/Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='D:/Object_Detection/frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = []
file_name='D:/Object_Detection/labels.txt'
with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')
#print(classLabels)
    
img=cv2.imread('D:/Object_Detection/img2.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
           
print('img')

