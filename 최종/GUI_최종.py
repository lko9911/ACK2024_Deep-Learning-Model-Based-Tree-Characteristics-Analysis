#--------------------------------------------GUI 구현--------------------------------------------#
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QGraphicsPixmapItem, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGraphicsView, QSizePolicy, QGraphicsScene, QGraphicsTextItem, QFrame, QLabel)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QFont, QPixmap, QImage,  QFontDatabase, QFont

class MyUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_path = ""

        # 그래픽스씬 객체, 픽스맵 객체 생성 및 연결
        self.scene = QGraphicsScene(self)
        self.video_frame = QGraphicsPixmapItem()
        self.scene.addItem(self.video_frame)
        
        # 그래픽스 뷰 1 & scene 객체 연결
        self.graphicsView1.setScene(self.scene)
        
        # 영상 파일 변수 생성
        self.video = None
        
        # QTimer 객체 생성
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def init_ui(self):

        # 제목 설정
        title_label = QLabel('Menu')
        title_font = QFont('Arial', 18)  
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('background-color: rgb(180, 197, 153);')

        # 버튼 생성
        self.btn1 = QPushButton('Load Data')
        self.btn2 = QPushButton('Capture')
        self.btn3 = QPushButton('Analyze')
        self.btn4 = QPushButton('Play / Pause')
        self.btn5 = QPushButton('Save Result')

        # 버튼 속성 설정 (일괄 적용)
        buttons = [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5]
        for btn in buttons:
            btn.setFixedSize(150, 50)
            font = QFont('Arial', 11)  
            btn.setFont(font)

        # 버튼 연결
        self.btn1.clicked.connect(self.load_data)
        self.btn2.clicked.connect(self.capture_image)
        self.btn3.clicked.connect(self.analyze_image)
        self.btn4.clicked.connect(self.toggle_play_pause)
        self.btn5.clicked.connect(self.save_image)

        # 버튼 1~4 박스 생성 및 레이아웃 설정 
        button_box = QFrame(self)
        button_box.setStyleSheet('background-color: rgb(190, 207, 163);')
        button_box_layout = QVBoxLayout()
        button_box_layout.setSpacing(70)

        button_box_layout.addWidget(self.btn1)
        button_box_layout.addWidget(self.btn4)
        button_box_layout.addWidget(self.btn2)
        button_box_layout.addWidget(self.btn3)

        button_box.setLayout(button_box_layout)
        button_box.setFrameShape(QFrame.Box)
        button_box.setFrameShadow(QFrame.Raised)

        # 버튼 5 텍스트 레이블 생성 (저장 버튼)
        btn5_label = QLabel('Save Result')
        btn5_font = QFont('Arial', 18)  
        btn5_font.setBold(True)
        btn5_label.setFont(btn5_font)
        btn5_label.setAlignment(Qt.AlignCenter)
        btn5_label.setStyleSheet('background-color: rgb(180, 197, 153);')

        # 버튼 5 박스 생성 및 레이아웃 설정
        button5_layout = QVBoxLayout()
        button5_layout.addWidget(self.btn5)

        button5_frame = QFrame(self)
        button5_frame.setStyleSheet('background-color: rgb(190, 207, 163);')
        button5_frame.setLayout(button5_layout)
        button5_frame.setFrameShape(QFrame.Box)
        button5_frame.setFrameShadow(QFrame.Raised)

        # vbox1: 첫 번째 섹션 (title_label + button_box) 버튼 1 ~ 4 영역
        vbox1 = QVBoxLayout()
        vbox1.addWidget(title_label)
        vbox1.addWidget(button_box)
        vbox1.addStretch(1)  
        explain = QLabel('▶ Use "Capture" to\ninput images as data\n\n▶ Use "Analyze" to \ndetect and analyze trees\n\n▶ If detection fails, try\na different angle\n\n▶ Only works for pine\n and yew trees')
        font = QFont('Arial', 12)  
        explain.setFont(font)
        vbox1.addWidget(explain)
        vbox1.addStretch(1)  

        # vbox2: 두 번째 섹션 (btn5_label + button5_frame) 버튼 5 영역
        vbox2 = QVBoxLayout() 
        vbox2.addWidget(btn5_label)
        vbox2.addWidget(button5_frame)
        vbox2.addStretch(1)
        explain_save = QLabel('▶ Save detected results \nand analysis results')
        font = QFont('Arial', 12)  
        explain_save.setFont(font)
        vbox2.addWidget(explain_save)
        vbox2.addStretch(1)

        # 각각의 섹션 별도 관리 (frame 틀 이용)
        self.button_frame = QFrame(self)
        self.button_frame.setFrameShape(QFrame.Box)
        self.button_frame.setFrameShadow(QFrame.Raised)
        self.button_frame.setStyleSheet('background-color: rgb(190, 207, 163);')
        self.button_frame.setLayout(vbox1)
        self.button_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.button_frame_2 = QFrame(self)
        self.button_frame_2.setFrameShape(QFrame.Box)
        self.button_frame_2.setFrameShadow(QFrame.Raised)
        self.button_frame_2.setStyleSheet('background-color: rgb(190, 207, 163);')
        self.button_frame_2.setLayout(vbox2)
        self.button_frame_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 그래픽 뷰 설정 및 제목 레이블 설정
        self.graphicsView1 = QGraphicsView(self)
        self.graphicsView2 = QGraphicsView(self)
        self.graphicsView3 = QGraphicsView(self)

        # 그래픽 뷰와 제목 레이블을 포함하는 레이아웃 설정 (일괄 적용)
        def create_view_with_title(title, view):
            layout = QVBoxLayout()
            label = QLabel(title)
            font = QFont('Arial', 14)  
            font.setWeight(QFont.Bold)  
            label.setFont(font)
            label.setStyleSheet(
                'background-color: rgb(200, 220, 180); '
                'padding: 5px; '
                'border: 2px solid rgb(150, 170, 120); '  
                'border-radius: 5px; '  
            )
            label.setAlignment(Qt.AlignCenter) 
            layout.addWidget(label)
            layout.addWidget(view)
            return layout

        view1_layout = create_view_with_title('Original Video', self.graphicsView1)
        view2_layout = create_view_with_title('Tree Detection', self.graphicsView2)
        view3_layout = create_view_with_title('Tree Analysis Results', self.graphicsView3)

        # 레이아웃 설정
        hbox1 = QHBoxLayout()
        hbox1.addLayout(view1_layout)
        hbox1.addLayout(view2_layout)

        vbox3 = QVBoxLayout()
        vbox3.addLayout(hbox1)
        vbox3.addLayout(view3_layout)

        # 최종 레이아웃 설정 (전체 화면)
        hbox2 = QVBoxLayout()  # 수직으로 배치
        hbox2.addWidget(self.button_frame)  # 첫 번째 섹션
        hbox2.addWidget(self.button_frame_2)  # 두 번째 섹션
        hbox2.setStretch(0, 1)  # 첫 번째 섹션의 비율 설정
        hbox2.setStretch(0, 1)  # 두 번째 섹션의 비율 설정

        hbox3 = QHBoxLayout()
        hbox3.addLayout(hbox2)  
        hbox3.addLayout(vbox3)   
        hbox3.setStretch(1, 4)  # 그래픽 뷰 레이아웃의 너비 비율 설정

        # 전체 화면 추가 설정
        self.setLayout(hbox3)
        self.setGeometry(QRect(100, 100, 1500, 1000))  

        # UI 설정
        self.setWindowTitle('TG Tree Analysis')
        self.setFocusPolicy(Qt.StrongFocus)


    def load_data(self):
        file_dialog = QFileDialog()     
        self.file_path, _ = file_dialog.getOpenFileName(self, '데이터 불러오기', '', '이미지 파일 (*.png *.jpg *.bmp) ;; 비디오 파일(*.mp4 *.avi *.mov *.mkv)')
              
        if self.file_path:
            if self.file_path.endswith(('.png', '.jpg', '.bmp')):
                self.image = cv2.imread(self.file_path)  
                pixmap = QPixmap(self.file_path)
                scaled_pixmap = pixmap.scaled(self.graphicsView1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                scene = QGraphicsScene(self)
                scene.addItem(QGraphicsPixmapItem(scaled_pixmap))
                self.graphicsView1.setScene(scene)
            
            elif self.file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.video = cv2.VideoCapture(self.file_path) #cv2.VC 객체 생성
                if self.video.isOpened():
                    self.timer.start(30)  #self.timer를 30ms 간격으로 시작 -> 30ms 마다 update_frame 실행
                    
                    #캡쳐 후 비디오 재업로드
                    self.scene = QGraphicsScene(self)
                    self.video_frame = QGraphicsPixmapItem()
                    self.scene.addItem(self.video_frame)
                    self.graphicsView1.setScene(self.scene)
                    
                else:
                    print("비디오를 불러오지 못했습니다.")
            else:
                print("비디오를 선택하지 않았습니다.")  
    
    
    def update_frame(self):
        if self.video and self.video.isOpened(): #파일이 실제 있는지 확인
            ret, frame = self.video.read()    #VC객체의 read메서드/ ret : 프레임 읽기 성공 여부, frame : 이미지 데이터
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Qt에서는 RGB를 사용하므로 BGR을 RGB로 변환하는 작업
                height, width, channel = frame.shape #프레임의 높이, 너비, 채널 수를 변수로 저장
                bytes_per_line = 3 * width #
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                #Qimage(프레임 이미지 데이터의 메모리 주소, 너비, 높이, 바이트 수, RGB 이미지 형식)
                
                # graphicsView1의 크기를 가져옴
                view_width = self.graphicsView1.width()
                view_height = self.graphicsView1.height()
                
                # scale 조정
                scaled_video = q_image.scaled(view_width, view_height, Qt.KeepAspectRatio)
                
                self.video_frame.setPixmap(QPixmap.fromImage(scaled_video)) #Qimage 객체를 QPixmap 객체로 변환하는 메서드
            
            else:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    
    def toggle_play_pause(self):
        
        if self.timer.isActive():  # 현재 재생 중이면
            self.timer.stop()  # 일시정지
            self.btn4.setText('Play')
            
        else:  # 현재 일시정지 상태면
            self.timer.start(30)  # 재생 시작 (30ms 간격, 필요에 따라 조절)
            self.btn4.setText('Pause')
 

    def capture_image(self):
        if self.video and self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                file_dialog = QFileDialog()
                self.file_path, _ = file_dialog.getSaveFileName(self, '이미지 저장', '', '이미지 파일 (*.png *.jpg *.bmp)')
                if self.file_path:
                    cv2.imwrite(self.file_path, frame)
                    print(f"이미지가 {self.file_path}에 저장되었습니다.")

                    pixmap = QPixmap(self.file_path)
                    scaled_pixmap = pixmap.scaled(self.graphicsView1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    scene = QGraphicsScene(self)
                    scene.addItem(QGraphicsPixmapItem(scaled_pixmap))
                    self.graphicsView1.setScene(scene)
                else:
                    print("저장할 파일 경로를 선택하지 않았습니다.")
            else:
                print("캡쳐할 프레임이 없습니다.")
        else:
            print("비디오가 로드되지 않았습니다.")
  
    
    def save_image(self):
        
        file_dialog = QFileDialog(self)
        file_dialog.setDefaultSuffix('png')  
        file_path, _ = file_dialog.getSaveFileName(self, '이미지 저장', '', '이미지 파일 (*.png *.jpg *.bmp *.jpeg)')

        if self.file_path:
            # graphicsView2의 이미지 저장
            try:
                if self.graphicsView2.scene() is not None:
                    view2_image = self.graphicsView2.grab().toImage()
                    view2_image.save(f"{file_path}_view2.png", "PNG")
                    print(f"graphicsView2의 이미지가 {file_path}_view2.png에 저장되었습니다.")
                else:
                    print("graphicsView2에 표시된 이미지가 없습니다.")
            except Exception as e:
                print(f"graphicsView2 이미지 저장 중 오류가 발생했습니다: {e}")

            # graphicsView3의 이미지 저장
            try:
                if self.graphicsView3.scene() is not None:
                    view3_image = self.graphicsView3.grab().toImage()
                    view3_image.save(f"{file_path}_view3.png", "PNG")
                    print(f"graphicsView3의 이미지가 {file_path}_view3.png에 저장되었습니다.")
                else:
                    print("graphicsView3에 표시된 이미지가 없습니다.")
            except Exception as e:
                print(f"graphicsView3 이미지 저장 중 오류가 발생했습니다: {e}")
        else:
            print("저장할 파일 경로를 선택하지 않았습니다.")

    def analyze_image(self):

        # 프로그램 시작 시간 기록
        start_time = time.time()
        
        if self.file_path:
            image = cv2.imread(self.file_path)

            self.image = image  # 이미지 저장
            # YOLO 분석 코드 실행
            results = yolo_model.predict(image, imgsz=640, conf=0.5, save=False, show=False)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            font_thickness = 2
            text_color = (0, 0, 0)  
            analysis_text = ""  # 분석 결과 텍스트를 저장할 변수

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
                    cropped_img = self.image[y1:y2, x1:x2]
                    class_label = int(bbox.cls)  
                    label = result.names[class_label]

                    class_name = 'Unknown'
                    color = (0, 0, 0)
                    text_color = (255, 255, 255)  
                    
                    if label == 'Tree':  
                        predicted_class = predict(cropped_img, species_model, species_names)
                        analysis_text += f'▶ 품종의 예측 결과 : {predicted_class}\n'
                        if predicted_class == '소나무':
                            shape, tree, trunk, trunk2 = predict_pine_Multi(cropped_img, pine_model, label)
                            analysis_text += f'▶ 소나무의 수형 모양 예측 결과 : {tree}\n'
                            analysis_text += f'▶ 소나무의 수관 모양 예측 결과 : {shape}\n'
                            analysis_text += f'▶ 소나무 줄기의 생장 습성 예측 결과 : {trunk}\n'
                            analysis_text += f'▶ 소나무의 줄기 갈라진 모양 예측 결과 : {trunk2}\n'
                        elif predicted_class == '주목':
                            shape, trunk, trunk2 = predict_yew_Multi(cropped_img, yew_model, label)
                            analysis_text += f'▶ 주목의 수형 모양 예측 결과 : {shape}\n'
                            analysis_text += f'▶ 주목 원줄기의 생장 습성 예측 결과 : {trunk}\n'
                            analysis_text += f'▶ 주목 원줄기의 분지수 예측 결과 : {trunk2}\n'

                        color = class_colors['Tree']

                    elif label == 'Bark':
                        bark = predict_pine_Multi(cropped_img, pine_model, label)
                        analysis_text += f'▶ 소나무의 수피 예측 결과 : {bark}\n'
                        color = class_colors['Bark']
                        text_color = (255, 255, 255)            
                        min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
                        analysis_text += print_color_name(color_fin, class_label)

                    elif label == 'Leaf':
                        predicted_class = predict(cropped_img, species_model, species_names)
                        analysis_text += f'▶ 품종의 예측 결과 : {predicted_class}\n'
                        if predicted_class == '소나무':
                            leaf = predict_pine_Multi(cropped_img, pine_model, label)
                            analysis_text += f'▶ 소나무 가지의 처짐 예측 결과 : {leaf}\n'
                        elif predicted_class == '주목':
                            leaf = predict_yew_Multi(cropped_img,  yew_model, label)
                            analysis_text += f'▶ 주목 소지의 성장 방향 예측 결과 : {leaf}\n'

                        color = class_colors['Leaf']
                        text_color = (255, 255, 255) 
                        min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
                        analysis_text += print_color_name(color_fin, class_label)

                    elif label == 'Trunk':  
                        color = class_colors['Trunk']
                        text_color = (255, 255, 255) 
                        min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
                        analysis_text += print_color_name(color_fin, class_label)

                    elif label == 'Crown': 
                        color = class_colors['Crown']
                        text_color = (255, 255, 255)
                        min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
                        analysis_text += print_color_name(color_fin, class_label)
                    
                    elif label == 'Flower': 
                        color = class_colors['Flower']
                        text_color = (0,0,0)
                        min_distance_fin, color_fin = color_main(cropped_img, 5, class_label)
                        analysis_text += print_color_name(color_fin, class_label)            

                    cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 3)
                    confidence = bbox.conf.item()
                    label_text = f'{label} {confidence:.2f}'
                    label_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                    text_width, text_height = label_size
                    
                    if label in ['Tree', 'Trunk']:
                        cv2.rectangle(self.image, (x1, y1 + text_height + 10), (x1 + text_width + 10, y1), color, -1)
                        cv2.putText(self.image, label_text, (x1 + 5, y1 + text_height + 5), font, font_scale, text_color, font_thickness)
                    elif label == 'flower':
                        cv2.rectangle(self.image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                        cv2.putText(self.image, label_text, (x1 + 5, y1 - text_height + 10), font, font_scale, text_color, font_thickness)          
                    else:
                        cv2.rectangle(self.image, (x2, y1 + text_height + 10), (x2 - text_width - 15, y1), color, -1)
                        cv2.putText(self.image, label_text, (x2 - 5 - text_width, y1 + text_height + 5), font, font_scale, text_color, font_thickness)


            end_time = time.time()

            # 실행 시간 계산 및 출력
            execution_time = end_time - start_time
            time_text = f"\n▶ 연산 시간 : {execution_time:.2f}초"
            analysis_text += time_text

            # 분석 결과 텍스트를 graphicsView3에 표시
            font_id = QFontDatabase.addApplicationFont('NanumMyeongjoExtraBold.ttf')
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            scene = QGraphicsScene(self)
            text_item = QGraphicsTextItem(analysis_text)
            font = QFont(font_family, 25)
            text_item.setFont(font)
            text_item.setDefaultTextColor(Qt.black)
            scene.addItem(text_item)
            self.graphicsView3.setScene(scene)

            # 분석된 이미지를 graphicsView2에 표시
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            result_qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
            result_pixmap = QPixmap.fromImage(result_qimage)
            scaled_pixmap = result_pixmap.scaled(self.graphicsView2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scene = QGraphicsScene(self)
            scene.addItem(QGraphicsPixmapItem(scaled_pixmap))
            self.graphicsView2.setScene(scene)

#--------------------------------------------모델 구현 파트--------------------------------------------#

# 라이브러리 가져오기 (YOLO, ResNet, sklearn 종합)
from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import time  
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 피쳐경고 무시
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#---------------------------------색상 분석 알고리즘-----------------------------#

A = [96, 78, 54]  # 등갈색
B = [87, 81, 77]  # 회갈색
C = [66, 63, 58]  # 흑갈색

D = [136, 156, 35]  # 녹황색
E = [59, 87, 49]  # 녹색
F = [49, 79, 65]  # 청록색

G = [195, 139, 128] # 연자주색
H = [181, 113, 113] # 자주색
I = [169, 106, 144] # 적자색

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
    
    if class_label in [0, 5]:
        pre = "수피의 색상 : "
        color_name = color_names_Bark.get(color_tuple, "색상을 알 수 없습니다.")
    elif class_label in [1, 3]:
        pre = "잎의 색상 : "
        color_name = color_names_Leaf.get(color_tuple, "색상을 알 수 없습니다.")
    elif class_label == 2:
        pre = "꽃의 색상 : "
        color_name = color_names_Flower.get(color_tuple, "색상을 알 수 없습니다.")
    else:
        return "색상을 분석할 수 없는 대상입니다."
    
    color_name = f"▶ {pre}{color_name}\n"
    
    return color_name
              
#---------------------------------YOLO_ResNet50 실행부-----------------------------#

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_ui = MyUI()
    my_ui.show()    
    sys.exit(app.exec_())
