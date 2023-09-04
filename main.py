import time
import cv2
import numpy as np
from cam_operator import Operator
# from TCPmanager import TCP
from collections import deque #status queue로 정의하기
from collections import defaultdict
from ultralytics import YOLO
from realsense import Realsense
from detect_book import DetectBook
import math as m

realsense = True  # 지역변수 오류때문에 미리 선언해둠
'''로봇의 state 정의'''
# status = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
status = 0
wait = False
find_word = 624875 # 625223 # 623400
ip, port = '***.*.*.*', 2015

if __name__ == '__main__':
    kdh = True
    ChadChang = False

    if kdh:
        model = YOLO('/home/kdh/Downloads/best_v8.pt')
        book_detector = DetectBook(0)
        realsense = False  # True면 뎁스카메라, False면 비디오
    elif ChadChang:
        runner = Operator() # '''state마다 동작을 정해주는 클래스를 따로 만들음'''
        calb_path = '/home/chad-chang/PycharmProjects/23_HF071/calib_data/MultiMatrix.npz'
        model = YOLO('/home/chad-chang/PycharmProjects/23_HF071/best.pt')
        book_detector = DetectBook(calb_path)
        realsense = True  # True면 뎁스카메라, False면 비디오
        tcp = TCP(ip, port, available=True)

    # Flags
    tracking = False  # 글자 읽었을 때 트래킹
    detect_center_book = True   # 말 그대로 중앙에 있는 책을 계속 검출할 것인지.

    if realsense:
        RS = Realsense()
    else:  # video
        cap = cv2.VideoCapture('/home/kdh/Downloads/20230801_002342.mp4')

    # Lists
    track_history = defaultdict(lambda: [])

    frame = []
    book_id = 0
    label_id = 0

    # Loop through the video frames
    while True:
        starttime = time.time()
        if realsense:
            results = model.track(RS.color_image, persist=True) # Run YOLOv8 tracking on the frame, persisting tracks between frames
            book_detector.get_image(RS)
        else:  # video
            _, frame = cap.read()
            frame = cv2.resize(frame, (640, 384))
            results = model.track(frame, persist=True)
            book_detector.get_video_image(frame)

            '''통신값 받기 대기'''

        '''status0 : 통신 대기'''
        if ChadChang:
            # 유석 (아두이노-> 매트랩 -> 파이썬)
            '''status0 : 통신 대기'''
            if status == 0:
                status_buff = tcp.receive()  # 통신받은 값은 값은 status값임
                print(status_buff)
                if status_buff:
                    status = status_buff
                    print('status = 0->1', status)

                '''status1 : 책 읽으러 동작''' # 조금 수정할 필요있을듯
            elif status == 1:  # 책 읽으러 동작
                if not wait:
                    ret_r, H = runner.run_approach(book_detector, results, visual = True)
                    print(ret_r)
                    print(runner.TF)
                    # 보내는 값이 온전한지 확인
                    # 통신 보내는 것필요
                    if ret_r:  # 특징 추출 완료하면
                        tcp.send(H, matrix=True)
                    wait = True
                status_buff = tcp.receive()
                if status_buff:
                    status = status_buff
                    print('status = 1->2', status)
                    wait = False
                    # 잠깐 쉬는 시간 가질까

                '''status2 : 글자 읽기 + 오차 계산'''
            elif status == 2:  # reading by ocr
                ret_r = runner.read_ocr(book_detector, results, find_word, visual = True)
                # print(ret_r)
                if ret_r: # 읽었을 때 status3으로 가라
                    print('status = 2->3', status)
                    status = 3

                    # print('status', status)
                else: # 못 읽었으면 8번 상태로 가라 또는 다시 읽을까
                    print('status = 2->8', status)
                    status = 8
                    # print('status =', status)


                '''status3: 해당 책 위치로 이동'''
            elif status == 3:  # 해당책 위치로 이동:
                ret_r , H= runner.read_estimate(book_detector, visual =True )
                if not wait:
                    ret_r, H = runner.read_estimate(book_detector, visual=True)
                    # 보내는 값이 온전한지 확인
                    # 통신 보내는 것필요
                    if ret_r:  # 특징 추출 완료하면

                        tcp.send(H)
                    wait = True
                status_buff = tcp.receive()
                if status_buff:
                    status = status_buff
                    wait = False
                    print('status = 3->4', status)
                    # 잠깐 쉬는 시간 가질까

            elif status == 4: # 트래킹하면서 좌표 보내주기
                if results[0].boxes != None and results[0].boxes.id != None:
                    annotated_frame = results[0].plot()  # Visualize the results on the frame
                    if tracking:  # w 누르고 글자를 읽기 성공 시 트래킹
                        book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)  # 마스크 검출
                        label_id, book_id, label_idx, book_idx = book_detector.IDcheck(label_id, book_id)
                        if label_id == -1 and book_id == -1:
                            tracking = False
                            print('tracking = False')
                        else:
                            book_detector.tracking_getter(label_idx, book_idx, visualize=True)  # 글자 읽은 책, 글자 읽은 라벨지, 책+라벨지 트래킹
                            # 좌표 통신해 줌
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                else:  # not tracking
                    cv2.imshow("YOLOv8 Tracking", results[0].plot())

                print('status = 4->5', status)


            elif status == 5:
                if not wait:
                    lc_point, blc_point = book_detector.LC_point, book_detector.BLC_point
                    ang = book_detector.calc_ang(lc_point, blc_point,visual = True)
                    if ang:
                        tcp.send(ang)
                status_buff = tcp.receive()
                if status_buff:
                    status = status_buff
                    wait = False
                    print('status = 5->6', status)

            # elif status == 7: # 로봇팔 구동이 끝나고 유석이 쪽으로 신호줄때


        # 트래킹
        if results[0].boxes != None and results[0].boxes.id != None:
            annotated_frame = results[0].plot()  # Visualize the results on the frame

            if tracking:  # w 누르고 글자를 읽기 성공 시 트래킹
                book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)  # 마스크 검출
                label_id, book_id, label_idx, book_idx = book_detector.IDcheck(label_id, book_id)
                if label_id==-1 and book_id==-1:
                    tracking=False
                    print('tracking = False')
                else:
                    book_detector.tracking_getter(label_idx, book_idx, visualize=True)  # 글자 읽은 책, 글자 읽은 라벨지, 책+라벨지 트래킹

            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:  # not tracking
            cv2.imshow("YOLOv8 Tracking", results[0].plot())

        endtime = time.time()
        print('처리 시간= ',endtime-starttime)
        # Break the loop if 'q' is pressed
        k_ = cv2.waitKey(1) & 0xFF
        if k_ == ord("q"):
            tcp.close()
            break
        # elif k_ == ord('w'):  # w 누르면 프레임 멈추고 글자 읽기. 글자 읽기 성공 시 트래킹 시작
        #     if realsense:
        #         book_detector.get_image(RS)
        #     else :
        #         book_detector.get_video_image(frame)
        #     ret = runner.read_ocr(book_detector, results, 620123 ,visual = False)
        #     print('ret',ret)
        #     if ret:
        #         tracking = True
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
            # book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            # ret, label_id, book_id = book_detector.ocr(word=620123)
            # if ret:
            #     tracking = True

        elif k_ == ord('w'):  # w 누르면 프레임 멈추고 글자 읽기. 글자 읽기 성공 시 트래킹 시작
            if realsense:
                book_detector.get_image(RS)
            else :
                book_detector.get_video_image(frame)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            ret, label_id, book_id = book_detector.ocr(word=find_word, visualize=True)
            if ret:
                tracking = True
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif k_ == ord('e'):
            if realsense:
                book_detector.get_image(RS)
            else :
                book_detector.get_video_image(frame)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            book_detector.center_book(visualize=True)
            cv2.waitKey(0)

        elif k_ == ord('r'):
            if realsense:
                book_detector.get_image(RS)
            else :
                book_detector.get_video_image(frame)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            book_detector.center_book(visualize=False)
            book_detector.read_center_word(visualize=True)

