import cv2
import numpy as np
cap = cv2.VideoCapture(0)
confidence_threshold = 0.5
nms_threshold = 0.3     # The lower it is, the more aggressive it will be and we'll have less boxes

# Step-2: Collecting the coco class names
classes_file = "coco.names.txt"
class_names = []
with open(classes_file, "rt") as f:
    class_names = f.read().rstrip('\n').split('\n')
# print(class_names)
# print(len(class_names))

# Step-3: Importing the configuration and weights files to create the network
model_config = "yolov3.cfg"
model_weights = "yolov3.weights"

# Defining the network
network = cv2.dnn.readNetFromDarknet(model_config, model_weights)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Step-4: Converting the images into a specific format "blob", inside the while loop

# Step-5: Finding if the prediction of an object is good enough for us or not
def find_object(outputs, img):
    height, width, channels = img.shape

    bounding_boxes = []       # This will contain the x, y, width, height
    class_ids = []          # This will contain all the class ids
    confidence_values = []
    # So, whenever we find a good object detection, we'll put the values in these lists

    for output in outputs:
        for detection in output:
            scores = detection[5:]          # Omitting the first 5 values
            class_id = np.argmax(scores)    # the index with the highest probability
            confidence = scores[class_id]   # the class name of the index with the highest probability

            if confidence > confidence_threshold:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int((detection[0] * width) - w/2)
                y = int((detection[1] * height) - h/2)

                bounding_boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    print(len(bounding_boxes))

    # Step-6: Creating bounding boxes, removing the extra bounding boxes
    #         and printing the class name with the confidence score

    indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_values, confidence_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = bounding_boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"{class_names[class_ids[i]].upper()} {int(confidence_values[i] * 100)}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)



# Step-1: Getting the webcam ready
while True:
    success, img = cap.read()

    # Step-4:
    blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), [0, 0, 0], 1, crop=False)
    network.setInput(blob)

    # To know the names of the layers of our network
    layer_names = network.getLayerNames()
    # print(layer_names)
    # We only need to extract the output layers
    # print(network.getUnconnectedOutLayers())    # This prints the indices of the layers
    output_layers = [layer_names[i[0]-1] for i in network.getUnconnectedOutLayers()]
    # print(output_layers)        # Shows the names of the output layers
    # Now we can send this image as a forward pass to our network and
    # we can find the output of the 3 output layers
    outputs = network.forward(output_layers)
    # print(len(outputs))
    # print(type(outputs[0]))
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    find_object(outputs, img)


    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break