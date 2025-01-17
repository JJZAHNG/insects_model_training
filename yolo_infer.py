import cv2
import numpy as np

# 加载YOLO
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# with open("models/insects.names", "r") as f:
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    return blob, width, height

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # 读取帧
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # 预处理图像
        blob, width, height = preprocess_image(frame)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # 置信度阈值
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if class_ids[i] < len(classes):
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    print(f"Warning: Detected class_id {class_ids[i]} is out of range for classes list.")

        # 显示结果
        cv2.imshow("Insect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
