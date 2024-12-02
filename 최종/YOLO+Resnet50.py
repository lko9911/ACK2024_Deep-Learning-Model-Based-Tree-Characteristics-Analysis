 라이브러리 가져오기 (YOLO, ResNet, sklearn 종합)
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
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans

# 피쳐경고 무시
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#---------------------------------색상 분석 알고리즘-----------------------------#
'''
# 기준 색상(standard) 정의 (전역 변수)
A = [126, 78, 50]  # 등갈색
B = [126, 126, 126]  # 회갈색
C = [0, 0, 0]  # 흑갈색
D = [67, 244, 58]  # 녹황색
E = [7, 154, 0]  # 녹색
F = [31, 94, 28]  # 청록색
G = [179, 80, 179] # 연자주색
H = [139, 101, 140] # 자주색
I = [174, 143, 171] # 적자색
J = [214, 193, 131] # 담황색
K = [219, 226, 170] # 황색
'''

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
              
#---------------------------------YOLO_ResNet50 실행부-----------------------------#

# 프로그램 시작 시간 기록
start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLO 모델 로드
yolo_model = YOLO('Model/YOLOv10x_ver9.pt')

# 품종 클래스
species_names = ['소나무','주목']

# 다중 클래스 특성 분류 (소나무)
bark_names = ['깊다','중간','얇다']  
leaf_names = ['있다','없다']  
shape_names = ['구형','좁은 원추형','부정형','넓은 원추형']  
tree_names = ['교목','아교목','관목']  
trunk_names = ['구부러진다','옆으로 긴다','굳게 자란다']  
trunk2_names = ['갈라짐','갈라지지 않음']  

num_bark_classes = len(bark_names)    
num_leaf_classes = len(leaf_names)
num_shape_classes = len(shape_names)
num_tree_classes = len(tree_names)
num_trunk_classes = len(trunk_names)
num_trunk2_classes = len(trunk_names)

# 다중 클래스 특성 분류 (주목)
leaf_yew_names = ['아래로 처짐','수평','위로 향함']
shape_yew_names = ['원추형','장타원형','광타원형']
trunk_yew_names = ['분지형','포복형','직립형']
trunk2_yew_names = ['다간','단간']

num_leaf_yew_classes = len(leaf_yew_names)
num_shape_yew_classes = len(shape_yew_names)
num_trunk_yew_classes = len(trunk_yew_names)
num_trunk2_yew_classes = len(trunk2_yew_names)

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV 이미지를 PIL 이미지로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원을 추가
    return image

# 단일 모델
def load_resnet_model(weights_path, class_names):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model

# 다중 모델용 (소나무)    
class MultiTaskModel_Pine(nn.Module):
    def __init__(self, num_ftrs, num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes):
        super(MultiTaskModel_Pine, self).__init__()
        self.shared = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-1])
        self.fc_bark = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_bark_classes)
        )
        self.fc_leaf = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_leaf_classes)
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_shape_classes)
        )
        self.fc_tree = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_tree_classes)
        )
        self.fc_trunk = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_trunk_classes)
        )
        self.fc_trunk2 = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_trunk2_classes)
        )

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        bark_out = self.fc_bark(x)
        leaf_out = self.fc_leaf(x)
        shape_out = self.fc_shape(x)
        tree_out = self.fc_tree(x)
        trunk_out = self.fc_trunk(x)
        trunk2_out = self.fc_trunk2(x)
        return bark_out, leaf_out, shape_out, tree_out, trunk_out, trunk2_out

def load_resnet_model_Pine(model_path, num_ftrs, num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes):
    model = MultiTaskModel_Pine(num_ftrs, num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval() 
    return model

# 다중 모델용 (주목)    
class MultiTaskModel_Yew(nn.Module):
    def __init__(self, num_ftrs, num_leaf_yew_classes, num_shape_yew_classes, num_trunk_yew_classes, num_trunk2_yew_classes):
        super(MultiTaskModel_Yew, self).__init__()
        self.shared = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-1])
        self.fc_leaf = nn.Linear(num_ftrs, num_leaf_yew_classes)
        self.fc_shape = nn.Linear(num_ftrs, num_shape_yew_classes)
        self.fc_trunk = nn.Linear(num_ftrs, num_trunk_yew_classes)
        self.fc_trunk2 = nn.Linear(num_ftrs, num_trunk2_yew_classes)

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)  # 텐서를 평탄화
        leaf_out = self.fc_leaf(x)
        shape_out = self.fc_shape(x)
        trunk_out = self.fc_trunk(x)
        trunk2_out = self.fc_trunk2(x)
        return leaf_out, shape_out, trunk_out, trunk2_out

def load_resnet_model_Yew(model_path, num_ftrs, num_leaf_yew_classes, num_shape_yew_classes, num_trunk_yew_classes, num_trunk2_yew_classes):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model = MultiTaskModel_Yew(num_ftrs, num_leaf_yew_classes, num_shape_yew_classes, num_trunk_yew_classes, num_trunk2_yew_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

species_model = load_resnet_model('resnet50_species.pth', species_names)
pine_model = load_resnet_model_Pine('resnet50_pine_model_ver4.pth',2048,num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes)
yew_model = load_resnet_model_Yew('resnet50_yew_model_ver2.pth',2048,num_leaf_yew_classes, num_shape_yew_classes, num_trunk_yew_classes, num_trunk2_yew_classes)

# 예측 수행 함수 (단일) 
def predict(image, model, class_names):
    image = preprocess_image(image)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds.item()]

# 예측 수행 함수 (다중) : 소나무
def predict_pine_Multi(image, model, label):
    image = preprocess_image(image).to(device)
    model.to(device)   

    with torch.no_grad():
        bark_outputs, leaf_outputs, shape_outputs ,tree_outputs, trunk_outputs, trunk2_outputs = model(image)
        _, bark_pred = torch.max(bark_outputs, 1)
        _, leaf_pred = torch.max(leaf_outputs, 1)
        _, shape_pred = torch.max(shape_outputs, 1)
        _, tree_pred = torch.max(tree_outputs, 1)
        _, trunk_pred = torch.max(trunk_outputs, 1)
        _, trunk2_pred = torch.max(trunk2_outputs, 1)

    if label == 'Tree':
        return shape_names[shape_pred.item()],tree_names[tree_pred.item()], trunk_names[trunk_pred.item()] ,trunk2_names[trunk2_pred.item()]
    elif label == 'Bark':
        return bark_names[bark_pred.item()]
    elif label == 'Leaf':
        return leaf_names[leaf_pred.item()]

# 예측 수행 함수 (다중) : 주목
def predict_yew_Multi(image, model, label):
    image = preprocess_image(image).to(device)
    model.to(device)   

    with torch.no_grad():
        leaf_yew_outputs, shape_yew_outputs, trunk_yew_outputs, trunk2_yew_outputs = model(image)
        _, leaf_yew_pred = torch.max(leaf_yew_outputs, 1)
        _, shape_yew_pred = torch.max(shape_yew_outputs, 1)
        _, trunk_yew_pred = torch.max(trunk_yew_outputs, 1)
        _, trunk2_yew_pred = torch.max(trunk2_yew_outputs, 1)

    if label == 'Tree':
        return shape_yew_names[shape_yew_pred.item()], trunk_yew_names[trunk_yew_pred.item()], trunk2_yew_names[trunk2_yew_pred.item()]
    elif label == 'Leaf':
        return leaf_yew_names[leaf_yew_pred.item()]


#---------------------------------코드 실행부-----------------------------#
## 입력
img_path = "image/sample.jpg"

# 이미지 읽기
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

#---------------------------------YOLO 인터페이스 조정-----------------------------#

font_path = 'C:\WINDOWS\FONTS\MALGUNSL.TTF'  
font_prop = fm.FontProperties(fname=font_path)

# 탐지된 객체 크롭 및 분류
prediction_results = []

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
            predicted_class = predict(cropped_img, species_model, species_names)
            result_text = f'품종의 예측 결과 : {predicted_class}'
            print(f'\n{result_text}')
            prediction_results.append(result_text)

            if predicted_class == '소나무':
                shape, tree, trunk, trunk2 = predict_pine_Multi(cropped_img, pine_model, label)
                result_text = f'소나무의 수형 모양 예측 결과 : {tree}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
                result_text = f'소나무의 수관 모양 예측 결과 : {shape}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
                result_text = f'소나무 줄기의 생장 습성 예측 결과 : {trunk}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
                result_text = f'소나무의 줄기 갈라진 모양 예측 결과 : {trunk2}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
            
            elif predicted_class == '주목':
                shape, trunk, trunk2 = predict_yew_Multi(cropped_img, yew_model, label)
                result_text = f'주목의 수형 모양 예측 결과 : {shape}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
                result_text = f'주목 원줄기의 생장 습성 예측 결과 : {trunk}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
                result_text = f'주목 원줄기의 분지수 예측 결과 : {trunk2}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)

            color = class_colors['Tree']

        elif label == 'Bark':
            bark = predict_pine_Multi(cropped_img, pine_model, label)
            result_text = f'소나무의 수피 예측 결과 : {bark}'
            print(f'\n{result_text}')
            prediction_results.append(result_text)
            color = class_colors['Bark']
            text_color = (255, 255, 255)            
            min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
            print_color_name(color_fin, class_label)

        elif label == 'Leaf':
            predicted_class = predict(cropped_img, species_model, species_names)
            result_text = f'품종의 예측 결과 : {predicted_class}'
            print(f'\n{result_text}')
            prediction_results.append(result_text)
            if predicted_class == '소나무':
                leaf = predict_pine_Multi(cropped_img, pine_model, label)
                result_text = f'소나무 가지의 처짐 예측 결과 : {leaf}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)
            elif predicted_class == '주목':
                leaf = predict_yew_Multi(cropped_img,  yew_model, label)
                result_text = f'주목 소지의 성장 방향 예측 결과 : {leaf}'
                print(f'\n{result_text}')
                prediction_results.append(result_text)

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
        elif label == 'Flower':
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(image, label_text, (x1 + 5, y1 - text_height + 10), font, font_scale, text_color, font_thickness)          
        else:
            cv2.rectangle(image, (x2, y1 + text_height + 10), (x2 - text_width - 15, y1), color, -1)
            cv2.putText(image, label_text, (x2 - 5 - text_width, y1 + text_height + 5), font, font_scale, text_color, font_thickness)

    # 프로그램 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산 및 출력
    execution_time = end_time - start_time
    execution_time_text = f'프로그램 실행 시간: {execution_time:.2f}초'
    print(f'\n{execution_time_text}')
    prediction_results.append(execution_time_text)

# 이미지를 BGR에서 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 출력
plt.figure(figsize=(15, 15))
plt.imshow(image_rgb)
plt.axis('off')  # 축 제거

# 예측 결과 텍스트 추가
text_y = 1  # 텍스트가 이미지 아래에 위치하도록 설정
for i, text in enumerate(prediction_results):
    plt.figtext(0.5, text_y - (0.05 * i), text, ha='center', fontsize=12, fontproperties=font_prop, wrap=True)

plt.show()
