import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import pyrealsense2 as rs
import numpy as np
import torch
from threading import Thread

class Realsense:
    Max_Instance = 1
    Cnt_Instance = 0

    def __new__(cls, *args, **kwargs):  # protective
        if (cls.Cnt_Instance >= cls.Max_Instance):
            raise ValueError("Cannot create more objects")
        cls.Cnt_Instance += 1
        return super().__new__(cls)

    def __init__(self, img_size=640, stride=32, auto=True):
        # yolov5와 호환되도록 만든 attribute
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference # 선택사항
        self.mode = 'stream'                # predict.py에서 dataset의 형태(이미지, 웹캠, 영상 등) 판별
        self.img_size = img_size            # 프레임의 w, h
        self.stride = stride                # 이후 letterbox(패딩에 사용)에 사용되어, 크기 조절에 사용
        self.threads = [None]

        self.pipeline = rs.pipeline()       # 뎁스캠에서 프레임을 받아오는 걸 래핑
        self.config = rs.config()           # 프레임 configuration(창 크기, 해상도, frame rate)
        self.align = np.array([])           # RGB Cam과 stereo의 alignment(프레임, 초점 맞춰줌). 현재 RGB 시점
        self.color_image = np.array([])     # RGB 프레임 데이터(영상 데이터) - 데이터 형식이 0~2^8 (8bit).
                                            # image 붙은거 - opencv에서 rendering(호환)되게 끔 만든 데이터
        self.depth_image = np.array([])     # Depth 프레임 데이터(opencv에서 호환되도록 변환해준 행렬)
        self.distance_frame = np.array([])  # Depth 프레임의 raw 데이터 - 생으로 거리 행렬

        # 객체 생성 시 바로 set_cam 실행되도록 함
        self.set_cam(640, 480, 30)
        self.stream_once()                  # self.color_image에 초기값 저장

        self.threads[0] = Thread(target=self.streaming, daemon=True)  # update 부분을 스레드로 무한 반복
        self.threads[0].start()

        '''
        self.calib_data_path = 0            # *지금 안씀. 나중에 캘리브레이션 파라미터 받아오는 형식으로 사용할 것
        self.cam_mat = 0                    # *위와 동일
        self.dist_coef = 0                  # *위와 동일(캘리브레이션)
        self.r_vectors = 0                  # *위와 동일
        self.t_vectors = 0                  # *위와 동일(자세 추정) - 여기서 안씀
        self.depth_scale = 0                # 뎁스 스케일(metric -> mm).
        '''

        """
        pipeline  클래스는 애플리케이션에 맞는 출력에 최적화된 카메라 설정을 선택하게 해준다.
        카메라를 활성화하고 다른 스트림의 스레드를 관리하고 활성 스트림 중 시간에 동기화된 프레임을 제공해 준다.
        파이프라인은 저수준의 장치 인터페이스에 접근하도록 하고, 이를 감싸준 것이다. 센서의 정보 및 미세 조정 능력 역시 앱에서 접근 가능하다. 
        파이프라인 API는 저리 블럭을 통해 가능하며 동통의 이미지 처리 동작을 시행할 수 있는 툴을 제공한다.
        [출처] [Realsense API] API에 대한 문서 내용 조금씩|작성자 초초
        """

    def set_cam(self,width,height,frame_rate):  # 파이프라인 설정
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)       # user interaction with the device 도와줌
        pipeline_profile = self.config.resolve(pipeline_wrapper)    # 모든 요청을 충족할 수 있는 장치를 찾고, 첫 번째 스트림 구성을 선택
        device = pipeline_profile.get_device()                      # 장치 찾기
        device_product_line = str(device.get_info(rs.camera_info.product_line)) # 장치 이름 인식
        found_rgb = False                                           # 색 센서가 있는 장치인지 확인
        for s in device.sensors:                                    # s => pipeline의 device정보 객체임.
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)
        if device_product_line == "L500":                           # 장치 이름
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)                  # 구성 요소 정보를 가지고 파이프라인 시작
        depth_sensor = profile.get_device().first_depth_sensor()    # 장치 정보에서 첫 뎁스 센서 가져옴
        self.depth_scale = depth_sensor.get_depth_scale()           # depth scale
        print("Depth Scale is = ", self.depth_scale)
        align_to = rs.stream.color                                  # sampling, occlusion 등 보정하는 역할
        self.align = rs.align(align_to)

    def stream_once(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        self.color_image = np.asanyarray(color_frame.get_data())

    def streaming(self):  # Yolov5 Dataloader의 update에 사용
        '''
        # 주의! 이 메서드 안에서 self.show_image 메서드 실행 시키면 yolov5와의 충돌로 오류남
        '''
        try:
            while True:
                frames = self.pipeline.wait_for_frames()                # (color, depth). 프레임 셋을 선언하고, 파이프에서 프레인셋을 기다림
                aligned_frames = self.align.process(frames)             # (color, depth)2가지 프레임 데이터를 캘브해주는 객체
                color_frame = aligned_frames.get_color_frame()          # (color) align frame을 받아와서 color frame을 추출.
                aligned_depth_frame = aligned_frames.get_depth_frame()  # (depth) color frame 기준으로 depth frame을 캘브한 객체
                depth_color_frame = rs.colorizer().colorize(aligned_depth_frame)    # (depth) rs 라이브러리에서 depth맵을 거리에 따라 컬러화
                self.color_image = np.asanyarray(color_frame.get_data())     # (color) 컬러 데이터를 opencv 호환 데이터로 변환
                self.depth_image = np.asanyarray(depth_color_frame.get_data())           # (depth) 컬러 데이터를 opencv 호환 데이터로 변환
                self.distance_frame = aligned_depth_frame

        except Exception as e:
            print(e)

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()

    def __iter__(self):
        self.count = -1
        return self

    # @property <- 이거 들어가 있으면 오류남
    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        im0 = self.color_image.copy()
        im0 = np.array([im0])  # 반환 형태를 맞추기 위해
        # print(len(im0))
        depth0 = self.depth_image.copy()  # 왜 리스트 데이터로 나오는지
        # im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=True)[0] for x in im0])  # resize
        # im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        # im = np.ascontiguousarray(im)  # contiguous
        depth0 = np.ascontiguousarray(depth0)  # contiguous

        return 0, im, im0, depth0, ''  # depth0[0]도 depth0으로 바꿔야될듯 나중에

    def __len__(self):
        return 1

    def distance(self,x,y):         # depth의 raw값
        distance = self.distance_frame(x,y)
        print(distance)

    def show_image(self):           # 렌더링
        cv2.imshow("frame", self.color_image)
        cv2.imshow('depth', self.depth_image)

if __name__ == "__main__" :
    RS = Realsense()

    while True:
        RS.show_image()
        key = cv2.waitKey(1)
        if key == ord("w"):
            print('w')
        if key == ord("q"):
            break

        pass