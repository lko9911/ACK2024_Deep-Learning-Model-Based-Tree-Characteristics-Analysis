# 라이브러리 가져오기 
from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import time  # 시간 측정을 위한 모듈
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

A = [96, 78, 54]  # 등갈색
B = [87, 81, 77]  # 회갈색
C = [66, 63, 58]  # 흑갈색
# A-B : 24.879710609249457 / B-C : 33.555923471125034 / C-A : 33.77869150810907

D = [136, 156, 35]  # 녹황색
E = [59, 87, 49]  # 녹색
F = [49, 79, 65]  # 청록색
# D-E : 104.33599570617994 / E-F : 20.493901531919196 / D-F : 119.99166637729472

G = [195, 139, 128] # 연자주색
H = [181, 113, 113] # 자주색
I = [169, 106, 144] # 적자색
# G-H : 33.12099032335839 / H-I : 33.97057550292606 / G-I : 44.955533585978046

J = [170, 147, 91] # 담황색
K = [196, 153, 85] # 황색

standard_Bark = [A, B, C]
standard_Leaf = [D, E, F]
standard_Flower = [G, H, I, J, K]

color_names_Bark = {tuple(A): "등갈색", tuple(B): "회갈색", tuple(C): "흑갈색"}
color_names_Leaf = {tuple(D): "녹황색", tuple(E): "녹색", tuple(F): "청록색"}
color_names_Flower = {tuple(G): "연자주색", tuple(H): "자주색", tuple(I): "적자색",tuple(J): "담황색", tuple(K): "황색"}

# 유클리드 거리 계산
def distance_cal(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# 클러스터링된 색상 추출
def calculate_main_color(image, k):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    main_colors = kmeans.cluster_centers_.astype(int)
    return main_colors

# 클러스터링 비교
# 클러스터링으로 추출한 색상들(colors_clustering)
def comparison_Bark(colors_clustering):
    clustering_dic = {}
    for x in colors_clustering:
        for y in standard_Bark:
            distance = distance_cal(x, y)
            clustering_dic[tuple(y)] = distance
    
    # 최단 거리(min_distance) & 대표 색상(main_color) 찾기
    main_color = min(clustering_dic, key=clustering_dic.get)
    min_distance = clustering_dic[main_color]
    
    return min_distance, list(main_color)

def comparison_Leaf(colors_clustering):
    clustering_dic = {}
    for x in colors_clustering:
        for y in standard_Leaf:
            distance = distance_cal(x, y)
            clustering_dic[tuple(y)] = distance
    
    # 최단 거리(min_distance) & 대표 색상(main_color) 찾기
    main_color = min(clustering_dic, key=clustering_dic.get)
    min_distance = clustering_dic[main_color]
    
    return min_distance, list(main_color)

def comparison_Flower(colors_clustering):
    clustering_dic = {}
    for x in colors_clustering:
        for y in standard_Flower:
            distance = distance_cal(x, y)
            clustering_dic[tuple(y)] = distance
    
    # 최단 거리(min_distance) & 대표 색상(main_color) 찾기
    main_color = min(clustering_dic, key=clustering_dic.get)
    min_distance = clustering_dic[main_color]
    
    return min_distance, list(main_color)

def color_main(image, n, class_label):
    whole_dic = {}
    
    if image is None:
        print("이미지를 불러올 수 없습니다.")
    
    for k in range(1, n+1):
        # 클러스터링된 색상 추출
        colors_clustering = calculate_main_color(image, k)
        if class_label == 0 or class_label == 5:
            # 클러스터링된 색상 비교
            min_distance, main_color = comparison_Bark(colors_clustering)  
            whole_dic[(tuple(main_color), k)] = min_distance
        elif class_label == 1 or class_label == 3:
            min_distance, main_color = comparison_Leaf(colors_clustering)  
            whole_dic[(tuple(main_color), k)] = min_distance
        elif class_label == 2:
            min_distance, main_color = comparison_Flower(colors_clustering) 
            whole_dic[(tuple(main_color), k)] = min_distance

    # 전체 딕셔너리(whole_dic)에서 최종 최단 거리(min_distance_fin) 찾기
    color_fin, _ = min(whole_dic, key=whole_dic.get)
    min_distance_fin = whole_dic[(color_fin, _)]
    return min_distance_fin, list(color_fin)

# 기준 색상 이름 출력
def print_color_name(color, class_label):
    color_tuple = tuple(color)

    if class_label == 0 or class_label == 5:
        color_name = color_names_Bark.get(color_tuple, "색상을 알 수 없습니다.")
        print("\n수피의 색상 : ", color_name)
    elif class_label == 1 or class_label == 3:
        color_name = color_names_Leaf.get(color_tuple, "색상을 알 수 없습니다.")
        print("\n잎의 색상 : ", color_name)
    elif class_label == 2:
        color_name = color_names_Flower.get(color_tuple,"색상을 알 수 없습니다.")
        print("\n꽃의 색상 : ", color_name)
    else:
        print("색상을 분석할수 없는 대상입니다.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "image/pine_flower.jpeg"

# YOLO 모델 로드
yolo_model = YOLO('Model/YOLOv10x_ver9.pt')

image = cv2.imread(img_path)

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)

# cv2.putText에서 사용할 폰트와 관련된 매개변수 설정
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.8
font_thickness = 2
text_color = (0, 0, 0)  

# 각 클래스에 대한 색상 정의 (BGR 형식)
class_colors = {
    'Tree': (0, 255, 0),    
    'Trunk': (255, 0, 0),   
    'Bark': (255, 204, 204),   
    'Crown': (212, 231, 0),
    'Flower' : (0,226,255),
    'Leaf' : (0,126,0)  
}

for result in results:
    for bbox in result.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        cropped_img = image[y1:y2, x1:x2]
        class_label = int(bbox.cls)  # 클래스 레이블을 인덱스로 변환
        label = result.names[class_label]

        # 기본값 설정
        class_name = 'Unknown'
        color = (0, 0, 0)  # 기본 색상을 검정으로 설정
        text_color = (255, 255, 255)  # 기본 텍스트 색상을 흰색으로 설정
              
        if label == 'Tree':  

            color = class_colors['Tree']

        elif label == 'Bark':
            
            color = class_colors['Bark']
            text_color = (255, 255, 255)            
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)

        elif label == 'Leaf':

            color = class_colors['Leaf']
            text_color = (255, 255, 255) 
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)

        elif label == 'Trunk':  
            color = class_colors['Trunk']
            text_color = (255, 255, 255) 
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)

        elif label == 'Crown': 
            color = class_colors['Crown']
            text_color = (255, 255, 255)
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)
        
        elif label == 'Flower': 
            color = class_colors['Flower']
            text_color = (0,0,0)
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)            

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        confidence = bbox.conf.item()
        
        # 텍스트 그리기
        label_text = f'{label} {confidence:.2f}'
        label_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        text_width, text_height = label_size
        
        if label in ['Tree', 'Trunk']:
            cv2.rectangle(image, (x1, y1 + text_height + 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 5, y1 + text_height + 5), font, font_scale, text_color, font_thickness)
        elif label == 'flower':
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 5, y1 - text_height + 10), font, font_scale, text_color, font_thickness)          
        else:
            cv2.rectangle(image, (x2, y1 + text_height + 10), (x2 - text_width - 15, y1), color, -1)
            cv2.putText(image, label_text, (x2 - 5 - text_width, y1 + text_height + 5), font, font_scale, text_color, font_thickness)
    
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
