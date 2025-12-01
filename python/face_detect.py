import cv2
import numpy as np
import mediapipe as mp

# 設定方法
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 人臉偵測設定
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/Admin/test/model/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

# 執行人臉偵測
with FaceDetector.create_from_options(options) as detector:
    img = cv2.imread('C:/Users/Admin/test/image/image.jpg')               # 讀取圖片
    if img is None:
            print("Cannot receive image")   # 如果讀取錯誤，印出訊息
            exit()

    
    h = img.shape[0]                            # 取得圖片高度
    w = img.shape[1]                            # 取得圖片寬度
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img) # 轉換成 mediapipe 圖片物件
    detection_result = detector.detect(mp_image)  # 偵測人臉
    
    
        # ret, frame = cap.read()             # 讀取影片的每一幀
        # w = frame.shape[1]                  # 畫面寬度
        # h = frame.shape[0]                  # 畫面高度
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # 轉換成 mediapipe 圖片物件
        # detection_result = detector.detect(mp_image)  # 偵測人臉  

    for detection in detection_result.detections:
        bbox = detection.bounding_box     # 人臉外框
        x = bbox.origin_x                # 人臉左上 x 座標
        y = bbox.origin_y                # 人臉左上 y 座標
        width = bbox.width                # 人臉寬度
        height = bbox.height              # 人臉高度
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,0,255),5) # opencv 繪圖
            # 取出人臉特徵值
        for keyPoint in detection.keypoints:
            cx = int(keyPoint.x*w)      # 特徵值 x 座標，乘以畫面寬度，因為特徵值是比例
            cy = int(keyPoint.y*h)      # 特徵值 y 座標，乘以畫面高度，因為特徵值是比例
            cv2.circle(img,(cx,cy),10,(0,0,255),-1) 
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)    
    cv2.imshow('showimg', img)     # 如果讀取成功，顯示該畫面
    cv2.waitKey(0)
    cv2.destroyAllWindows()                 # 結束所有視窗
    