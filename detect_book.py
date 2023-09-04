import easyocr
import numpy as np
import cv2
import numpy.linalg as LA
import math as m
import traceback


class DetectBook:  # 책의 특징점을 찾아서 tf를 만드는 클래스
    # def __init__(self,calib_data_path): # calib_data를 처음에 선언하고 기억하는게 바람직해 보임
    def __init__(self, calib_data_path):  # calib_data를 처음에 선언하고 기억하는게 바람직해 보임
        #################realsense 객체#############################################
        self.distance_frame = 0  # raw depth map/ not a numpy data
        self.color_image = np.array([])
        self.depth_image = np.array([])  # depth_color_image로 변환된 데이터
        self.gray_image = np.array([])  # 글자 읽을 때 사용
        self.depth_scale = 0

        self.book_idx = None
        self.label_idx = None
        self.center_label_idx = None

        self.mask_centers = np.array([])
        self.contours = []

        self.label_centers = np.array([])

        #################도현이가 줘야할것############################################
        self.BC_point = np.array([])  # 책의 중점(x,y)
        self.LC_point = np.array([])  # 레이블의 중점(x,y)
        self.BLC_point = np.array([])  # 책+레이블의 중점(x,y)

        self.B_corner = np.array([])  # 책의 코너점  _ minareaRect로 뽑은 corner점
        self.L_corner = np.array([])  # 레이블의 코너점_ minareaRect로 뽑은 corner점
        self.BL_corner = np.array([])  # 책과 레이블의 코너점_ minareaRect로 뽑은 corner점

        if calib_data_path != 0:
            calib_data = np.load(calib_data_path)  # 추가할 정보
            self.dist_coef = calib_data["distCoef"]
            self.cam_mat = calib_data["camMatrix"]

        self.status = 0  # 물체 인식 실패 또는 뭐시기 -> 케이스에 대한 파라미터// status에 대한 내용 정리해서 올려야 할듯
        #################창민쪽#####################################################

        self.ttform = np.array([])
        self.cp2 = np.array([])  # 중점 x,y 좌표
        self.cp3 = np.array([])  # 중점 x,y,z 좌표

    ##### 김도현 메서드 #####
    def get_image(self, dataset):
        '''
        :param dataset: Realsense 객체를 불러오기
        '''
        # 0. 이미지 불러오기 (컬러맵, 뎁스맵, 뎁스 컬러맵, 회색조)
        self.color_image = dataset.color_image
        self.depth_image = dataset.depth_image
        self.gray_image = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)  # 글자 읽을 때 필요함
        self.origin_h, self.origin_w, _ = self.color_image.shape  # reshape(선택사항)
        self.distance_frame = dataset.distance_frame
        self.depth_scale = dataset.depth_scale

    def get_video_image(self, frame):
        '''
        비디오 형식일 경우
        :param frame: 영상의 이미지 데이터를 불러오기
        '''
        self.color_image = frame
        self.gray_image = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)  # 글자 읽을 때 필요함
        self.origin_h, self.origin_w, _ = self.color_image.shape  # reshape(선택사항)

    def get_mask(self, det, masks, names, resize=False) -> int:
        '''
        마스크를 가져와 self에 저장.
        라벨지만 따로 분리하여 saliency map 생성

        :param det: 검출된 bbox 리스트
        :param masks: 검출된 이진마스크 리스트
        :param names: 이름 리스트
        :param resize: 원본 크기로 resize할 지 여부
        :return: 0:책과 라벨지 검출, 1:라벨지만 검출, 2:책만 검출, 3:둘다 검출X
        '''
        det_len = len(det)

        if not det_len:  # 물체가 검출되지 않았을 때
            return 3

        # 지역변수 경고 해결
        label_masks = np.array([])

        # 각 리스트의 길이
        det_len = det.shape[0]  # 검출 물체 수
        cls_list = [int(det.cls[i]) for i in range(det_len)]  # 물체 분류 저장 # names = {0: 'Books', 1: 'Labels'}
        book_len, label_len = cls_list.count(0), cls_list.count(1)  # 책의 수, 라벨지의 수
        mask_len = det_len  # 마스크 수

        if 0 not in cls_list:  # 'Books' not exist
            return 1
        if 1 not in cls_list:  # 'Labels' not exist
            return 2

        # 빈 컨테이너 생성 (라벨 리스트, 책 리스트, 라벨 인덱스, 책 인덱스)
        label_list = np.zeros((label_len, 3), dtype=int)
        book_list = np.zeros((book_len, 3), dtype=int)
        mask_label_idx = np.zeros((mask_len,), dtype=int)
        book_idx = np.zeros((mask_len,), dtype=int)

        label_centers = np.zeros((label_len, 2), dtype=float)  # 라벨 중점을 저장
        mask_centers = np.zeros((mask_len, 2), dtype=float)  # 중점을 저장
        self.contours = []

        book_cnt = 0  # 책의 수 카운터
        label_cnt = 0  # 라벨지 수 카운터

        for i in range(mask_len):  # 마스크 각각의 컨투어를 찾는 코드
            mask = masks[i].byte().cpu().numpy()  # 나중에 이진마스크와 합쳐서 라벨지 마스크만 따로 떼오기

            if resize:
                mask = cv2.resize(mask, (self.origin_w, self.origin_h))

            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = sorted(list(contours), key=len, reverse=True)
            mmt = cv2.moments(contours[0])
            mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
            mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점

            mask_centers[i] = [mask_cx, mask_cy]  # 중점을 저장하는 리스트
            self.contours.append(contours[0])

            if names[int(cls_list[i])] == 'Books':
                xy_list = det.xyxy[i]  # 책의 bbox

                book_list[book_cnt] = [mask_cx, mask_cy, book_cnt]
                book_idx[book_cnt] = i

                book_cnt += 1
            else:  # names[int(cls_list[i])] == 'Labels':
                label_list[label_cnt] = [mask_cx, mask_cy, label_cnt]
                mask_label_idx[label_cnt] = i  # 나중에 글자 찾은 마스크 찾을 때

                label_centers[label_cnt] = [mask_cx, mask_cy]

                if label_cnt:  # 라벨지의 마스크를 합치는 코드
                    label_masks = cv2.bitwise_or(mask, label_masks)
                else:
                    label_masks = mask.copy()
                label_cnt += 1

        if resize:
            self.gray_image = cv2.resize(self.gray_image, (self.origin_w, self.origin_h))
        label_masks *= self.gray_image

        self.det = det
        self.masks = masks

        self.book_len, self.label_len = book_len, label_len
        self.label_list = label_list
        self.book_list = book_list
        self.mask_label_idx = mask_label_idx
        self.book_idx_list = book_idx
        self.label_masks = label_masks
        self.mask_centers = mask_centers
        self.label_centers = label_centers

        return 0

    def ocr(self, word='6229', save_points=True, resize=False, visualize=False) -> tuple:
        '''
        글자를 읽고 라벨지와 책 매칭 후 트래킹 시작
        :param word: 찾을 단어
        :param save_points: 중심점과 코너점 저장
        :param resize: 이미지를 원래 크기로 resize할 지 여부
        :param visualize: 시각화
        :return: 글자를 찾았는지 여부, 트래킹할 라벨지 id, 트래킹할 책 id
        '''
        if not isinstance(word, str):  # string인지 검사
            word = str(word)

        # 지역변수
        find_label = np.array([])
        find_book = np.array([])
        dist_min_idx = 0

        sharpening_img = self.sharpening(self.label_masks, strength=3)

        if visualize:
            cv2.imshow('original_mask', self.label_masks)
            cv2.imshow('sharpening_mask', sharpening_img)

        # 글자 읽기
        reader = easyocr.Reader(['en'])  # 한글도 적용하고 싶으면 ['ko', 'en']. 다만 여기선 자음만 인식 안돼서 안씀
        result = reader.readtext(sharpening_img,
                                 slope_ths=0.3)  # result = reader.readtext(self.label_masks, slope_ths=0.3)

        # word(찾는 글자)가 있는지 판단, 있으면 해당 책의 마스크 반환
        for i in range(len(result)):
            if visualize and '6' in result[i][1]:  # 확인용
                print(result[i][1])

            if word in result[i][1]:  # 'find-detect mode 0'
                word_x = int((result[i][0][0][0] + result[i][0][2][0]) / 2)
                word_y = int((result[i][0][0][1] + result[i][0][2][1]) / 2)
                dist_min = 1E6
                for j, item in enumerate(self.label_list):  # 글자와 라벨지 간의 최소 거리로 글자에 해당하는 라벨지 찾기
                    dist = (item[0] - word_x) ** 2 + (item[1] - word_y) ** 2
                    if int(dist) < dist_min:
                        dist_min = int(dist)
                        dist_min_idx = j

                # 라벨지를 찾았을 때
                self.label_idx = self.mask_label_idx[dist_min_idx]  # 찾은 라벨지의 인덱스

                if save_points:
                    self.LC_point = self.mask_centers[self.label_idx]  # 찾은 라벨지의 중심점
                    label_contour = self.contours[self.label_idx]  # 찾은 라벨지의 minAreaRect bbox
                    rect = cv2.minAreaRect(label_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    self.L_corner = box

                if resize:
                    find_label = cv2.resize(self.masks[self.label_idx].byte().cpu().numpy() * 255,
                                            (self.origin_w, self.origin_h))
                else:
                    find_label = self.masks[self.mask_label_idx[dist_min_idx]].byte().cpu().numpy() * 255

                # 책과 라벨지 매칭
                label_contour = self.contours[self.label_idx]
                (x, y), (MA, ma), angle = cv2.fitEllipse(label_contour)
                a = np.tan(np.deg2rad(90 - angle))
                b = (self.origin_h - y) - a * x

                # 점과 직선 사이 거리. |ax - y + b| / sqrt(a**2 + b**2)
                book_min_dist = 1E6
                book_min_idx = -1

                denominator = np.sqrt(a ** 2 + b ** 2)

                for book_idx, item in enumerate(self.book_list):
                    numerator = abs(a * item[0] - (self.origin_h - item[1]) + b)
                    if (numerator / denominator) < book_min_dist:
                        book_min_dist = (numerator / denominator)
                        book_min_idx = book_idx

                # 책을 찾았을 때
                self.book_idx = self.book_idx_list[book_min_idx]  # 찾은 책의 인덱스

                if save_points:
                    self.BC_point = self.mask_centers[self.book_idx]  # 찾은 책의 중심점
                    book_contour = self.contours[self.book_idx]  # 찾은 책의 minAreaRect bbox
                    rect = cv2.minAreaRect(book_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    self.B_corner = box

                if resize:
                    find_book = cv2.resize(self.masks[self.book_idx].byte().cpu().numpy() * 255,
                                           (self.origin_w, self.origin_h))
                else:
                    find_book = self.masks[self.book_idx].byte().cpu().numpy() * 255
                break

        if (len(find_label) == 0):
            print('not found label')  # 원하는 글자를 못 읽음
            return False, 0, 0
        else:
            if save_points:
                book_label_mask = cv2.bitwise_or(find_book, find_label)

                contours = cv2.findContours(book_label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = sorted(list(contours), key=len, reverse=True)
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                self.BL_corner = box

                mmt = cv2.moments(contours[0])
                mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
                mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점
                self.BLC_point = [mask_cx, mask_cy]

                if visualize:
                    cv2.circle(find_book, (int(self.BC_point[0]), int(self.BC_point[1])), 2, (0, 0, 0), 2)
                    cv2.circle(find_label, (int(self.LC_point[0]), int(self.LC_point[1])), 2, (0, 0, 0), 2)
                    cv2.circle(book_label_mask, (int(self.BLC_point[0]), int(self.BLC_point[1])), 2, (0, 0, 0), 2)

                    cv2.drawContours(find_book, [self.B_corner], 0, (255, 255, 255), 5)
                    cv2.drawContours(find_label, [self.L_corner], 0, (255, 255, 255), 5)
                    cv2.drawContours(book_label_mask, [self.BL_corner], 0, (255, 255, 255), 2)

                    cv2.imshow('find book', find_book)
                    cv2.imshow('find label', find_label)
                    cv2.imshow('find book_label_mask', book_label_mask)

            self.book_id = self.det.id.int().tolist()[self.book_idx]
            self.label_id = self.det.id.int().tolist()[self.label_idx]

            return True, self.label_id, self.book_id

        # find_label : 글자 인식을 통해 찾은 라벨지 마스크를 저장하고 있는 변수
        # find_book : 글자 인식을 통해 찾은 책 마스크를 저장하고 있는 변수
        # book_label_mask : 책 마스크 + 라벨 마스크

    def sharpening(self, image, strength):
        '''
        이미지 선명화
        :param image: 인풋 이미지
        :param strength: 선명화 정도
        :return: 아웃풋 이미지
        '''

        b = (1 - strength) / 8
        sharpening_kernel = np.array([[b, b, b],
                                      [b, strength, b],
                                      [b, b, b]])
        output = cv2.filter2D(image, -1, sharpening_kernel)
        return output

    def center_book(self, visualize=False):
        '''
        화면 상의 중점에서 가장 가까운 라벨지 찾고, 그와 매칭되는 책 찾기
        '''
        # 프레임의 중점
        cx, cy = int(self.origin_w / 2), int(self.origin_h / 2)

        # 프레임 중점과 최소 거리에 있는 라벨지
        L_min = 1E9
        L_min_idx = 0
        for idx, item in enumerate(self.label_centers):
            dist = (item[0] - cx) ** 2 + (item[1] - cy) ** 2

            if dist < L_min:
                L_min = dist
                L_min_idx = idx

        center_label_idx = self.mask_label_idx[L_min_idx]
        center_label = self.masks[center_label_idx]  # 찾은 라벨지의 인덱스
        center_label = center_label.byte().cpu().numpy() * 255  # 찾은 라벨지의 마스크
        center_label_contour = self.contours[center_label_idx]  # 찾은 라벨지의 컨투어
        self.LC_point = self.mask_centers[center_label_idx]  # 찾은 책의 중심점
        self.center_label_idx = center_label_idx

        contours = cv2.findContours(center_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(list(contours), key=len, reverse=True)

        (x, y), (MA, ma), angle = cv2.fitEllipse(contours[0])

        # 라벨지와 매칭되는 책 컨투어 찾기

        a = np.tan(np.deg2rad(90 - angle))
        b = (self.origin_h - y) - a * x

        B_min = 1E6
        B_min_idx = -1

        denominator = np.sqrt(a ** 2 + b ** 2)

        for idx, item in enumerate(self.book_list):
            numerator = abs(a * item[0] - (self.origin_h - item[1]) + b)
            if (numerator / denominator) < B_min:
                B_min = (numerator / denominator)
                B_min_idx = idx

        center_book_idx = self.book_idx_list[B_min_idx]
        center_book = self.masks[center_book_idx]  # 찾은 책의 인덱스
        center_book = center_book.byte().cpu().numpy() * 255  # 찾은 책의 마스크
        center_book_contour = self.contours[center_book_idx]  # 찾은 책의 minAreaRect bbox
        self.BC_point = self.mask_centers[center_book_idx]  # 찾은 책의 중심점

        # minAreaRect
        rect = cv2.minAreaRect(center_label_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.L_corner = box

        rect = cv2.minAreaRect(center_book_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.B_corner = box

        # book+label
        center_book_label = cv2.bitwise_or(center_label, center_book)

        kernel = np.ones((5, 5), np.uint8)
        center_book_label = cv2.dilate(center_book_label, kernel, iterations=2)  # // make dilation image

        contours = cv2.findContours(center_book_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(list(contours), key=len, reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.BL_corner = box

        mmt = cv2.moments(contours[0])
        mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
        mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점

        self.BLC_point = [mask_cx, mask_cy]  # 중점을 저장하는 리스트

        # 시각화 (확인용)
        if visualize:
            cv2.circle(center_book, (int(self.BC_point[0]), int(self.BC_point[1])), 2, (0, 0, 0), 2)
            cv2.circle(center_label, (int(self.LC_point[0]), int(self.LC_point[1])), 2, (0, 0, 0), 2)
            cv2.circle(center_book_label, (int(self.BLC_point[0]), int(self.BLC_point[1])), 2, (0, 0, 0), 2)

            cv2.drawContours(center_label, [self.L_corner], 0, (255, 255, 255), 2)
            cv2.drawContours(center_book, [self.B_corner], 0, (255, 255, 255), 2)
            cv2.drawContours(center_book_label, [self.BL_corner], 0, (255, 255, 255), 2)

            cv2.imshow('CB', center_book)
            cv2.imshow('CL', center_label)
            cv2.imshow('CBL', center_book_label)

    def tracking_getter(self, label_idx, book_idx, visualize=False):
        '''
        책과 라벨지 트래킹하며 중앙 점과 코너 점 검출
        :param label_idx: 검출할 라벨지의 인덱스 (그 프레임에서 몇 번째인지)
        :param book_idx: 검출할 책의 인덱스
        '''
        if visualize:
            trackL = self.masks[label_idx].cpu().byte().numpy() * 255
            trackB = self.masks[book_idx].cpu().byte().numpy() * 255

        contours = self.contours[book_idx]
        mmt = cv2.moments(contours)
        mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
        mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점
        self.BC_point = np.array([mask_cx, mask_cy])  # 책의 중점(x,y)
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.B_corner = box

        contours = self.contours[label_idx]
        mmt = cv2.moments(contours)
        mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
        mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점
        self.LC_point = np.array([mask_cx, mask_cy])  # 레이블의 중점(x,y)
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.L_corner = box

        trackBL = cv2.bitwise_or(self.masks[label_idx].cpu().byte().numpy() * 255, \
                                 self.masks[book_idx].cpu().byte().numpy() * 255)
        contours = cv2.findContours(trackBL, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(list(contours), key=len, reverse=True)
        mmt = cv2.moments(contours[0])
        mask_cx = int(mmt['m10'] / (mmt['m00'] + 1e-5))
        mask_cy = int(mmt['m01'] / (mmt['m00'] + 1e-5))  # 마스크의 중점
        self.BLC_point = np.array([mask_cx, mask_cy])  # 책+레이블의 중점(x,y)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.BL_corner = box
        # 시각화
        if visualize:
            cv2.circle(trackB, (int(self.BC_point[0]), int(self.BC_point[1])), 2, (0, 0, 0), 2)
            cv2.circle(trackL, (int(self.LC_point[0]), int(self.LC_point[1])), 2, (0, 0, 0), 2)
            cv2.circle(trackBL, (int(self.BLC_point[0]), int(self.BLC_point[1])), 2, (0, 0, 0), 2)

            cv2.drawContours(trackB, [self.B_corner], 0, (255, 255, 255), 5)
            cv2.drawContours(trackL, [self.L_corner], 0, (255, 255, 255), 5)
            cv2.drawContours(trackBL, [self.BL_corner], 0, (255, 255, 255), 2)

            cv2.imshow('trackB', trackB)
            cv2.imshow('trackL', trackL)
            cv2.imshow('trackBL', trackBL)

    def IDcheck(self, label_id, book_id):
        '''
        ID를 검사하여 ID switching이 발생했는지를 확인. 오류 방지
        getmask가 선행되어야됨
        :return:
        '''
        id_list = self.det.id.int().tolist()

        label_idx = -1
        book_idx = -1
        label_id_ = label_id
        book_id_ = book_id

        if len(self.det) == 0:
            return -1, -1, -1, -1

        # print('IDcheck - 0')

        if label_id == self.label_id and book_id == self.book_id and label_id in id_list and book_id in id_list:
            ''' case 1 : 만약 id를 정상적으로 받아왔으면, id_list에서 각각의 id를 통해 마스크를 검출 '''
            label_idx = id_list.index(label_id)
            book_idx = id_list.index(book_id)
            # print('IDcheck - 1')
        else:
            if label_id not in id_list:
                return -1, -1, -1, -1
            '''
            case 2 : 책의 id만 바뀐 경우. 라벨지 마스크를 통해 각도를 계산하고, 점과 직선사이 거리를 통해 책의 마스크를 추정
                     이 때, 거리가 어느 범위 이내일 경우 id를 업데이트
            '''
            # print('IDcheck - 2')

            label_idx = id_list.index(label_id)
            label_contour = self.contours[label_idx]

            (x, y), (MA, ma), angle = cv2.fitEllipse(label_contour)
            a = np.tan(np.deg2rad(90 - angle))
            b = (self.origin_h - y) - a * x

            # 점과 직선 사이 거리. |ax - y + b| / sqrt(a**2 + b**2)
            min_dist = 1E6
            min_idx = -1

            denominator = np.sqrt(a ** 2 + b ** 2)

            for idx, item in enumerate(self.book_list):
                numerator = abs(a * item[0] - (self.origin_h - item[1]) + b)
                if (numerator / denominator) < min_dist and (min_idx != label_idx):
                    min_dist = (numerator / denominator)
                    min_idx = idx

            if min_dist < 30:
                book_idx = self.book_idx_list[min_idx]
                print('book_idx', book_idx)
                matched_book = self.masks[self.book_idx_list[min_idx]].byte().cpu().numpy() * 255  # 찾은 책의 인덱스
                book_id_ = self.det.id.int().tolist()[book_idx]  # 리턴에 필요
                self.book_id = book_id_

        return label_id_, book_id_, label_idx, book_idx

    def read_center_word(self, visualize=False):
        '''
        center_book 메서드에서 저장된 corner, center points를 이용하여, 그 마스크의 글자를 읽음
        :return: self.center_word에 일련번호의 숫자를 저장
        '''

        center_label = self.masks[self.center_label_idx].byte().cpu().numpy()
        center_label = center_label * self.gray_image

        # 글자 읽기
        reader = easyocr.Reader(['en'])  # 한글도 적용하고 싶으면 ['ko', 'en']. 다만 여기선 자음만 인식 안돼서 안씀
        result = reader.readtext(center_label,
                                 slope_ths=0.3)  # result = reader.readtext(self.label_masks, slope_ths=0.3)

        found_word = []
        for read_word in result:
            if ('.' not in read_word[1]) and len(read_word[1]) >= 6:  # 확인용
                if read_word[1][-6:].isdigit():
                    found_word.append(read_word[1])

        self.center_word = None

        if len(found_word) == 1:
            self.center_word = found_word[0][-6:]
        else:
            for j in found_word:
                if ('Em' not in j) and ('EM' not in j):
                    continue
                else:
                    self.center_word = j[-6:]

        if visualize:
            print(self.center_word)
            cv2.imshow('img', center_label)
            cv2.waitKey(0)

        if self.center_word != None:
            return True
        else:
            return False

    #####이창민 메서드#######
    def limit_range(self, box):  # frame의 configuration을 벗어나지 않게 제한/
        '''
        :param box : 4 corner point
        :return:  box : constraint
        '''
        # input : box = [box1,box2,box3,box4], box1: (x,y)
        # output: box = [box1,box2,box3,box4]
        image = self.color_image
        np.putmask(box[:, 0], box[:, 0] >= image.shape[1], image.shape[1] - 1)
        np.putmask(box[:, 1], box[:, 1] >= image.shape[0], image.shape[0] - 1)
        np.putmask(box[:, 0], box[:, 0] <= 0, 0)
        np.putmask(box[:, 1], box[:, 1] <= 0, 0)
        return box

    # def rearrange_all(self, box, color, visual=False):  # 바운딩 박스를 순서에 맞게 재정렬 / box: 4개의 점을 담은 리스트
    #     # 수정점: 책이 없고 레이블만검출 되었을때는 오류가 뜰수밖에 없음
    #     # 해결책: 레이블지만 검출됬을때 쓸수있는 방법 모색
    #     '''
    #     :param box : 4개의 점
    #     :param visual: color image에 코너 포인트 시각화
    #     :return:  corners: 4개의 점
    #     '''
    #
    #     # input : box = [box1,box2,box3,box4], box1: (x,y)
    #     # output: box = [box1,box2,box3,box4]
    #
    #     box = self.limit_range(box)
    #     image = self.color_image
    #     lc_point = self.LC_point  # 레이블 center
    #     blc_point = self.BLC_point  # 책전체의 center
    #     cx_Label = lc_point[0]
    #     cy_Label = lc_point[1]
    #     cx_total = blc_point[0]  # 책 전제의 중심
    #     cy_total = blc_point[1]
    #     L_point, R_point = list(), list()
    #     P_LU, P_LD, P_RU, P_RD = 0, 0, 0, 0
    #     try:
    #         for i, b in enumerate(box):
    #             x = box[i, 0]
    #             y = box[i, 1]
    #             if cx_total != cx_Label:  # 라벨지의 중점X와 책 전체의 중점 X가 같지 않을때 -> 일직선이 아닐때
    #                 # 레이블지의 중점과 책의 중점을 연결하는 직선을 reference line으로 삼고, 점들의 x좌표를 대입했을 때 크기 비교해서 배열함.
    #                 # if cy_Label > cy_total:
    #                 eq = ((cy_total - cy_Label) / (cx_total - cx_Label)) * (x - cx_Label) + cy_Label
    #                 # elif cy_Label < cy_total:
    #                 #     eq = ((cy_total - cy_Label) / (cx_total - cx_Label)) * (x - cx_Label) + cy_Label
    #                 # else:
    #                 #     pass
    #
    #                 if cy_total > cy_Label and cx_total > cx_Label:  # 우하향 형상 => Y축이 거꾸로 되어있기 때
    #                     if y > eq:  # 왼쪽
    #                         R_point.append(box[i])
    #                     elif y < eq:  # 오른쪽
    #                         L_point.append(box[i])
    #                     elif y == eq:  # 이런일 없음
    #                         pass
    #                 elif cy_total > cy_Label and cx_total < cx_Label:  # 좌 하향 형상
    #                     if y > eq:  # 왼쪽
    #                         L_point.append(box[i])
    #                     elif y < eq:  # 오른쪽
    #                         R_point.append(box[i])
    #                     elif y == eq:  # 이런일 없음.
    #                         pass
    #                 elif cy_total < cy_Label and cx_total > cx_Label:  # 우 상향 형상
    #                     if y > eq:  # 오른쪽
    #                         R_point.append(box[i])
    #
    #                     elif y < eq:  # 왼쪽
    #                         L_point.append(box[i])
    #                     elif y == eq:  # 이런일 없음.
    #                         pass
    #                 elif cy_total < cy_Label and cx_total < cx_Label:  # 좌 상향  형상
    #                     if y > eq:  # 왼쪽
    #                         L_point.append(box[i])
    #                     elif y < eq:  # 오른쪽
    #                         R_point.append(box[i])
    #                     elif y == eq:  # 이런일 없음.
    #                         pass
    #             ####################### 여기까지 좌표거꾸로
    #             elif cx_total == cx_Label:  # 수직일 경우
    #                 if cy_total > cy_Label:  # 수직 하강 # (카메라 좌표계 아래 방향)
    #                     if x > cx_Label:  # 왼쪽에 있을때
    #                         L_point.append(box[i])
    #                     elif x < cx_Label:  # 오른쪽에 있을때
    #                         R_point.append(box[i])
    #                 elif cy_total < cy_Label:  # 수직 상승
    #                     if x > cx_Label:  # 오른쪽에 있을때
    #                         R_point.append(box[i])
    #                     elif x < cx_Label:  # 왼쪽에 있을때
    #                         L_point.append(box[i])
    #                 else:
    #                     pass
    #             # print(R_point, L_point)
    #         if R_point[0][1] > R_point[1][1]:
    #             P_RD = R_point[0]
    #             P_RU = R_point[1]
    #             # print("1P_RD,P_RU=",P_RD,P_RU)
    #         elif R_point[0][1] < R_point[1][1]:
    #             P_RD = R_point[1]
    #             P_RU = R_point[0]
    #             # print("2P_RD,P_RU=", P_RD, P_RU)
    #         if L_point[0][1] > L_point[1][1]:
    #             P_LD = L_point[0]
    #             P_LU = L_point[1]
    #             # print("1P_LD,P_LU=", P_LD, P_LU)
    #         elif L_point[0][1] < L_point[1][1]:
    #             P_LD = L_point[1]
    #             P_LU = L_point[0]
    #         if visual:
    #             cv2.circle(image, (int(P_LD[0]), int(P_LD[1])), 10, color, 5)
    #             cv2.putText(image, 'LD' + str(P_LD), (int(P_LD[0]), int(P_LD[1])), cv2.FONT_ITALIC, 1, color, 2)
    #
    #             cv2.circle(image, (int(P_RD[0]), int(P_RD[1])), 10, color, 5)
    #             cv2.putText(image, 'RD' + str(P_RD), (int(P_RD[0]), int(P_RD[1])), cv2.FONT_ITALIC, 1, color, 2)
    #
    #             cv2.circle(image, (int(P_LU[0]), int(P_LU[1])), 10, color, 5)
    #             cv2.putText(image, 'LU' + str(P_LU), (int(P_LU[0]), int(P_LU[1])), cv2.FONT_ITALIC, 1, color, 2)
    #
    #             cv2.circle(image, (int(P_RU[0]), int(P_RU[1])), 10, color, 5)
    #             cv2.putText(image, 'RU' + str(P_RU), (int(P_RU[0]), int(P_RU[1])), cv2.FONT_ITALIC, 1, color, 2)
    #
    #         if type(P_LD) == int or type(P_RD) == int or type(P_LU) == int or type(P_RU) == int:
    #             # print("false")
    #             return False  # 4개의 원소를 가진 리스트
    #         else:
    #             corners = [P_LD, P_RD, P_LU, P_RU]
    #             # print('sd',P_LD, P_RD, P_LU, P_RU)
    #             return corners
    #     except Exception as e:
    #         err_msg = traceback.format_exc()
    #         # print(err_msg)
    #         print("Rearrange_all_error")
    #         return False

    def rearrange_all(self, box, color, visual=False):  # 바운딩 박스를 순서에 맞게 재정렬 / box: 4개의 점을 담은 리스트

        # 수정점: 책이 없고 레이블만검출 되었을때는 오류가 뜰수밖에 없음
        # 해결책: 레이블지만 검출됬을때 쓸수있는 방법 모색
        '''
        :param box : 4개의 점
        :param visual: color image에 코너 포인트 시각화
        :return:  corners: 4개의 점
        '''

        # input : box = [box1,box2,box3,box4], box1: (x,y)
        # output: box = [box1,box2,box3,box4]

        box = self.limit_range(box)
        image = self.color_image
        lc_point = self.LC_point  # 레이블 center
        blc_point = self.BLC_point  # 책전체의 center
        cx_Label = lc_point[0]
        cy_Label = lc_point[1]
        cx_total = blc_point[0]  # 책 전제의 중심
        cy_total = blc_point[1]
        L_point, R_point = list(), list()
        P_LU, P_LD, P_RU, P_RD = 0, 0, 0, 0
        try:
            for i, b in enumerate(box):
                x = box[i, 0]
                y = box[i, 1]
                if cx_total != cx_Label:  # 라벨지의 중점X와 책 전체의 중점 X가 같지 않을때 -> 일직선이 아닐때
                    # 레이블지의 중점과 책의 중점을 연결하는 직선을 reference line으로 삼고, 점들의 x좌표를 대입했을 때 크기 비교해서 배열함.
                    # if cy_Label > cy_total:
                    eq = ((cy_total - cy_Label) / (cx_total - cx_Label)) * (x - cx_Label) + cy_Label
                    # elif cy_Label < cy_total:
                    #     eq = ((cy_total - cy_Label) / (cx_total - cx_Label)) * (x - cx_Label) + cy_Label
                    # else:
                    #     pass

                    if cy_total > cy_Label and cx_total > cx_Label:  # 우하향 형상 => Y축이 거꾸로 되어있기 때
                        if y > eq:  # 왼쪽
                            R_point.append(box[i])
                        elif y < eq:  # 오른쪽
                            L_point.append(box[i])
                        elif y == eq:  # 이런일 없음
                            pass
                    elif cy_total > cy_Label and cx_total < cx_Label:  # 좌 하향 형상
                        if y > eq:  # 왼쪽
                            L_point.append(box[i])
                        elif y < eq:  # 오른쪽
                            R_point.append(box[i])
                        elif y == eq:  # 이런일 없음.
                            pass
                    elif cy_total < cy_Label and cx_total > cx_Label:  # 우 상향 형상
                        if y > eq:  # 오른쪽
                            R_point.append(box[i])

                        elif y < eq:  # 왼쪽
                            L_point.append(box[i])
                        elif y == eq:  # 이런일 없음.
                            pass
                    elif cy_total < cy_Label and cx_total < cx_Label:  # 좌 상향  형상
                        if y > eq:  # 왼쪽
                            L_point.append(box[i])
                        elif y < eq:  # 오른쪽
                            R_point.append(box[i])
                        elif y == eq:  # 이런일 없음.
                            pass
                ####################### 여기까지 좌표거꾸로
                elif cx_total == cx_Label:  # 수직일 경우
                    if cy_total > cy_Label:  # 수직 하강 # (카메라 좌표계 아래 방향)
                        if x > cx_Label:  # 왼쪽에 있을때
                            L_point.append(box[i])
                        elif x < cx_Label:  # 오른쪽에 있을때
                            R_point.append(box[i])
                    elif cy_total < cy_Label:  # 수직 상승
                        if x > cx_Label:  # 오른쪽에 있을때
                            R_point.append(box[i])
                        elif x < cx_Label:  # 왼쪽에 있을때
                            L_point.append(box[i])
                    else:
                        pass
                # print('R and L',R_point, L_point)
            if R_point[0][1] > R_point[1][1]:
                P_RD = R_point[0]
                P_RU = R_point[1]
                # print("1P_RD,P_RU=",P_RD,P_RU)
            elif R_point[0][1] < R_point[1][1]:
                P_RD = R_point[1]
                P_RU = R_point[0]
                # print("2P_RD,P_RU=", P_RD, P_RU)
            if L_point[0][1] > L_point[1][1]:
                P_LD = L_point[0]
                P_LU = L_point[1]
                # print("1P_LD,P_LU=", P_LD, P_LU)
            elif L_point[0][1] < L_point[1][1]:
                P_LD = L_point[1]
                P_LU = L_point[0]
            # print("all =",P_LD, P_RD, P_LU, P_RU)
            if visual:
                cv2.circle(image, (int(P_LD[0]), int(P_LD[1])), 10, color, 5)
                cv2.putText(image, 'LD' + str(P_LD), (int(P_LD[0]), int(P_LD[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_RD[0]), int(P_RD[1])), 10, color, 5)
                cv2.putText(image, 'RD' + str(P_RD), (int(P_RD[0]), int(P_RD[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_LU[0]), int(P_LU[1])), 10, color, 5)
                cv2.putText(image, 'LU' + str(P_LU), (int(P_LU[0]), int(P_LU[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_RU[0]), int(P_RU[1])), 10, color, 5)
                cv2.putText(image, 'RU' + str(P_RU), (int(P_RU[0]), int(P_RU[1])), cv2.FONT_ITALIC, 1, color, 2)

            if type(P_LD) == int or type(P_RD) == int or type(P_LU) == int or type(P_RU) == int:
                # print("false")
                return False  # 4개의 원소를 가진 리스트
            else:
                corners = [P_LD, P_RD, P_LU, P_RU]
                # print('sd',P_LD, P_RD, P_LU, P_RU)
                return corners
        except Exception as e:
            err_msg = traceback.format_exc()
            # print(err_msg)
            print("Rearrange_all_error")
            return False

    def rearrange_one(self, box, color, visual=False):  # 하나의 물체만
        '''
        :param box : 4개의 점
        :param visual: color image에 코너 포인트 시각화
        :return:  corners: 4개의 점
        '''

        image = self.color_image
        try:
            v3_0 = box[0] - box[3]
            v3_2 = box[2] - box[3]
            P_LD = np.array([0, 0])
            P_LU = np.array([0, 0])
            P_RD = np.array([0, 0])
            P_RU = np.array([0, 0])

            if LA.norm(v3_0) >= LA.norm(v3_2):
                Vx = v3_2
                Vy = v3_0
                vy = v3_0 / LA.norm(Vy)
                vx = v3_2 / LA.norm(Vx)
                vz = np.cross(vx, vy)
                vz = -vz / LA.norm(vz)
                P_LD[:2] = box[3]
                P_LU[:2] = box[0]
                P_RD[:2] = box[2]
                P_RU[:2] = box[1]

            # 세로가 더 길때
            elif LA.norm(v3_2) > LA.norm(v3_0):
                Vx = -v3_0
                Vy = v3_2
                vy = Vy / LA.norm(Vy)
                vx = Vx / LA.norm(Vx)  # v0를 reference 포인트로 만들기시
                vz = np.cross(vx, vy)
                vz = -vz / LA.norm(vz)

                P_LD = box[0]
                P_LU = box[1]
                P_RD = box[3]
                P_RU = box[2]
            else:
                pass
            if visual:
                cv2.circle(image, (int(P_LD[0]), int(P_LD[1])), 10, color, 5)
                cv2.putText(image, 'LD' + str(P_LD), (int(P_LD[0]), int(P_LD[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_RD[0]), int(P_RD[1])), 10, color, 5)
                cv2.putText(image, 'RD' + str(P_RD), (int(P_RD[0]), int(P_RD[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_LU[0]), int(P_LU[1])), 10, color, 5)
                cv2.putText(image, 'LU' + str(P_LU), (int(P_LU[0]), int(P_LU[1])), cv2.FONT_ITALIC, 1, color, 2)

                cv2.circle(image, (int(P_RU[0]), int(P_RU[1])), 10, color, 5)
                cv2.putText(image, 'RU' + str(P_RU), (int(P_RU[0]), int(P_RU[1])), cv2.FONT_ITALIC, 1, color, 2)

            corners = [P_LD, P_RD, P_LU, P_RU]
            return corners
        except Exception as e:
            err_msg = traceback.format_exc()
            print("Rearrange\n", err_msg)
            return False

    def inner_point(self, corner_aligned, color, visual=False):
        '''
        :param : corner_aligned = 정렬된 코너점 입력
        :return:  corner_in :
        '''
        image = self.color_image
        # P_LD, P_LU, P_RD, P_RU = corner_aligned[0], corner_aligned[1], corner_aligned[2], corner_aligned[3]
        P_LD, P_RD, P_LU, P_RU = corner_aligned[0], corner_aligned[1], corner_aligned[2], corner_aligned[3]

        if type(P_LU) == int or type(P_LU) == int or type(P_LU) == int or type(P_LU) == int:
            # print("inner_point/ P_LU, P_RD, P_LU,P_RU=", P_LU, P_RD, P_LU, P_RU)
            # print("type of inner point", type(P_LU), type(P_RD), type(P_LU), type(P_RU))
            # print('/////////////////////////////////////////////////////')
            # print("inner_point pass")
            # print('/////////////////////////////////////////////////////')
            return False
        # error occur#
        else:
            P_LD_in = ((3 * P_LD + P_RU) / 4).astype(np.int64)
            P_LU_in = ((3 * P_LU + P_RD) / 4).astype(np.int64)
            P_RD_in = ((3 * P_RD + P_LU) / 4).astype(np.int64)
            P_RU_in = ((3 * P_RU + P_LD) / 4).astype(np.int64)

            if visual:
                cv2.circle(image, (int(P_LD_in[0]), int(P_LD_in[1])), 2, color, 2)
                cv2.putText(image, 'LD' + str(P_LD_in), (int(P_LD_in[0]), int(P_LD_in[1])), cv2.FONT_ITALIC, 0.5, color,
                            1)

                cv2.circle(image, (int(P_RD_in[0]), int(P_RD_in[1])), 2, color, 2)
                cv2.putText(image, 'RD' + str(P_RD_in), (int(P_RD_in[0]), int(P_RD_in[1])), cv2.FONT_ITALIC, 0.5, color,
                            1)

                cv2.circle(image, (int(P_LU_in[0]), int(P_LU_in[1])), 2, color, 2)
                cv2.putText(image, 'LU' + str(P_LU_in), (int(P_LU_in[0]), int(P_LU_in[1])), cv2.FONT_ITALIC, 0.5, color,
                            1)

                cv2.circle(image, (int(P_RU_in[0]), int(P_RU_in[1])), 2, color, 2)
                cv2.putText(image, 'RU' + str(P_RU_in), (int(P_RU_in[0]), int(P_RU_in[1])), cv2.FONT_ITALIC, 0.5, color,
                            1)

            corner_in = [P_LD_in, P_LU_in, P_RD_in, P_RU_in]
            return corner_in

    def convert_3d(self, point2d):  # 2차원 데이터를 3차원로 변환 / 검증 필요!!
        '''
        :param point2d np.array([x,y])
        :return:  point3d : np.array([x,y,z])
        '''
        x, y = int(point2d[0]), int(point2d[1])
        # print(x,y)
        # print(type(x),type(y))
        int_mat = self.cam_mat.copy()
        int_mat = np.concatenate((int_mat, np.array([[0], [0], [0]])), axis=1)
        distance = self.distance_frame.get_distance(x, y) / self.depth_scale  # distance frame은 xy 서로 바꾸지 않음
        xc_normal = (point2d[0] - int_mat[0, 2]) / int_mat[0, 0]
        yc_normal = (point2d[1] - int_mat[1, 2]) / int_mat[1, 1]
        xw = xc_normal * distance  # distance의 단위에 맞춰짐
        yw = yc_normal * distance
        point3d = np.array([xw, yw, distance])
        return point3d

    def convert_3d_corner(self, box):  # box =[P_LD_in, P_LU_in, P_RD_in, P_RU_in]
        '''
        param: point2d x4 = corner point2d
        return:  point3d  x4 = corner point3d
        '''
        if box:
            point3d = []
            for a in box:
                point3d.append(self.convert_3d(a))
            return point3d
        else:
            return False

    def calc_ang(self, lc_point, blc_point, visual=False):
        '''
        :param: label의 2d 포인트, blc의 2d 포인트
        :return:  angle : camera와 물체와의 각도 계산
        '''
        ang = 0
        image = self.color_image
        depth = self.depth_image
        # lc_point = self.LC_point  # 레이블 center
        # blc_point = self.BLC_point  # 책전체의 center
        # print('lc_point = ',lc_point, 'blc_point=', blc_point)
        # print(type(lc_point),type(blc_point))
        cx_Label = int(lc_point[0])
        cy_Label = int(lc_point[1])
        cx_total = int(blc_point[0])  # 책 전제의 중심
        cy_total = int(blc_point[1])
        # print('x_label = ',cx_Label)
        # print('y_label = ', cy_Label)
        # print('x_total = ', cx_total)
        # print('y_total = ', cy_total)

        if visual:
            # cv2.arrowedLine(image, (cx_total, cy_total), (cx_Label, cy_Label), (0, 0, 255), thickness=2)  # X축 표시
            cv2.arrowedLine(depth, (cx_Label, cy_Label), (cx_total, cy_total), (0, 0, 255), thickness=2)  # X축 표시
        try:
            Vy_cam = np.array([0, 1])
            Vy_object = np.array([cx_Label, cy_Label]) - np.array([cx_total, cy_total])
            Vy_object = Vy_object / (LA.norm(Vy_object))

            # 1번 트레젝토리에서 6번 모터 각도 체크
            # 책이 거꾸로 있는 경우 배재
            if Vy_object[1] < 0:
                Vy_object = -Vy_object
            if np.cross(Vy_cam, Vy_object) > 0:  # 시계방향으로 돌림.
                ang = round(m.degrees(np.arccos(np.dot(Vy_object, Vy_cam))), 3)
            elif np.cross(Vy_cam, Vy_object) < 0:  # 시계반대 방향 돌림.
                ang = - round(m.degrees(np.arccos(np.dot(Vy_object, Vy_cam))), 3)
            elif np.cross(Vy_cam, Vy_object) == 0:
                if any(Vy_object == Vy_cam):  # any() or all()
                    ang = 0
                else:
                    ang = 180
            return ang

        except:

            print("calc_ang  =", Vy_object, LA.norm(Vy_object))
            return False

    def isAvail_depth(self, corner_3d):  # 3차원 데이터 4개 /  depth값이 유효하지 않은 값이 들어가는지 확인
        P_LD_in, P_LU_in, P_RD_in, P_RU_in = corner_3d[0], corner_3d[1], corner_3d[2], corner_3d[3]
        # print(P_LD_in[2],P_LU_in[2],P_RD_in[2],P_RU_in[2])
        if P_LD_in[2] != 0 and P_LU_in[2] != 0 and P_RD_in[2] != 0 and P_RU_in[2] != 0:
            return True
        else:
            return False

    def calc_ori(self, corner_arrange_in):  # 정렬된 포인트 배열들을이용해 orientation을 구함
        ''' 잦은 에러: norm(vx),norm(vy)=0 이 되는 경우
        :param corner_arrange_in: 안쪽에 있는 점을 기준으로 orientation계산  한 포인트당 3d
        :return:  rotation : 3x3 matrix
        '''
        if corner_arrange_in:
            # input : 4개의 2d 포인트의 리스트
            image = self.color_image
            color = (255, 0, 0)
            # print(corner_arrange_in)
            if self.isAvail_depth(corner_arrange_in):
                P_LD_in, P_LU_in, P_RD_in, P_RU_in = corner_arrange_in[0], corner_arrange_in[1], corner_arrange_in[2], \
                    corner_arrange_in[3]
                vx = P_RD_in - P_LD_in
                vy = P_LD_in - P_LU_in
                '''debugging'''
                if LA.norm(vx) == 0 or LA.norm(vy) == 0:  # 이 상황이 되면 조치를 해야할 것 같음
                    print("///////////////////////////////////////////////////////////////")
                    print("calc_ori norm is zero")
                    print("/point_LD_inner=", P_LD_in, "point_LU_inner=", P_LU_in, "point_RD_inner=",
                          P_RD_in)
                    print("norm(vx)=", LA.norm(vx), "norm(vy)=", LA.norm(vy))
                    print("///////////////////////////////////////////////////////////////")
                vx = vx / LA.norm(vx)
                vy = vy / LA.norm(vy)
                vz = np.cross(vx, vy)
                vz = vz / LA.norm(vz)
                rotation = np.concatenate([np.array([vx]).T, np.array([vy]).T, np.array([vz]).T], axis=1)  # 3x3
                return rotation
        else:
            return False

    def euler_rot(self, theta):  # euler rotation : z-y-x : 3x3 matrix
        theta1, theta2, theta3 = theta[0], theta[1], theta[2]
        rot_z = np.matrix([[m.cos(theta1), -m.sin(theta1), 0],
                           [m.sin(theta1), m.cos(theta1), 0],
                           [0, 0, 1]])
        rot_y = np.matrix([[m.cos(theta2), 0, m.sin(theta2)],
                           [0, 1, 0],
                           [-m.sin(theta2), 0, m.cos(theta2)]])
        rot_x = np.matrix([[1, 0, 0],
                           [0, m.cos(theta3), -m.sin(theta3)],
                           [0, m.sin(theta3), m.cos(theta3)]])
        rot = np.dot(np.dot(rot_z, rot_y), rot_x)
        return rot

    def visual_coord(self, Rvec, Tvec):  # 책의 코디네이트 #
        # print('out= ', type(Rvec), type(Tvec))
        # print('in= ', type(Rvec), type(Tvec))
        # print('in2= ', Rvec, Tvec.squeeze())
        # input : Tvec(3,1 vec) / Rvec(3,3 mat)
        image = self.color_image
        depth = self.depth_image
        try:
            cv2.drawFrameAxes(depth, self.cam_mat, self.dist_coef, Rvec, Tvec, 4, 4)
        except:
            print('visual fail')
            return False


if __name__ == '__main__':
    # db = DetectBook()
    print("hello world!")



