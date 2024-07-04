import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('models/insect_classifier_model.h5')

# 昆虫类别
classes = ['beetle', 'butterfly', '...']  # 替换为你的类别名称

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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

        # 昆虫种类识别
        preprocessed_image = preprocess_image(frame)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)

        # 在图像上显示识别结果
        cv2.putText(frame, f'Insect: {classes[predicted_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
