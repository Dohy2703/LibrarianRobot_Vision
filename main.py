from cam_operator import Operator
from ultralytics import YOLO
from realsense import Realsense
from detect_book import DetectBook
from TCPmanager import TCP
from label_reader import labelReader

from collections import defaultdict
from collections import deque  # status queue로 정의하기
import time
import cv2
import numpy as np
import math as m
import argparse
import threading  # ìœ ì„
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from rclpy.qos import QoSProfile
from rclpy.node import Node


# status = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''대기 : status_wait 0 
접근 : status_approach1
글자 읽을수 있는지 확인 : status_readable2
글자 읽기 : status_read 9
해당 책 위치로 이동 : status_appr2book 3
트래킹 : status_tracking 4
align 5
통신 기다리는 모드 : status_wait_insert 6
카메라 프레임안에 책 없을때 + 책 못읽었을때 : status_NotRead 8 
로봇팔구동 끝나고 유석이한테 신호 줄때 : status_end : 22
tracking코드 넣는거는 ocr에서 넣어야함.'''

status_ordinary, status_read_fail, status_end = '0', '1', '2'
status_wait,status_wait2, status_depth, status_approach, status_readable = '3', '3.5','4', '5', '6'
status_read, status_appr2book, status_tracking = '7', '8', '9'
status_align, status_wait_insert = '10', '11'
status_Not_Read = '12'
book_serial_num = ""

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
color_b, color_l, color_bl = (255, 0, 0), (0, 255, 0), (0, 0, 255)

def librarian_main(
        yolov8_weights=None,  # model.py path
        source='/home/kdh/Downloads/20230801_002342.mp4',
        realsense=True,  # realsense
        status = None,
        # find_word = 623398,
        # find_word = book_serial_num,
        book_detector = None,
        ip = '127.0.1.1',
        port = 2014,
        tcp_available = False,
        query = None,
        wait = False,
):
    global book_serial_num

    kdh = False
    ChadChang = True
    '''state마다 동작을 정해주는 클래스를 따로 만들음'''

    if kdh:
        model = YOLO('/home/kdh/Downloads/best_v8.pt')
        book_detector = DetectBook(0)
        realsense = False  # True면 뎁스카메라, False면 비디오
    elif ChadChang:
        runner = Operator()
        # calb_path = '/home/chad-chang/PycharmProjects/23_HF071/calib_data/MultiMatrix.npz'
        calb_path = './MultiMatrix.npz'
        # model = YOLO('/home/chad-chang/PycharmProjects/23_HF071/best.pt')
        model = YOLO('/home/kdh/Downloads/best_v8.pt')
        book_detector = DetectBook(calb_path)
        realsense = True  # True면 뎁스카메라, False면 비디오q
        tcp = TCP(ip, port, available=tcp_available)

    # Flags
    tracking = False  # 글자 읽었을 때 트래킹

    if realsense:
        RS = Realsense()
    else:  # video
        cap = cv2.VideoCapture(source)

    # reader instance
    book_detector.setup_reader()

    frame = []
    book_id = 0
    label_id = 0

    '''status0 : 통신 대기'''
    while status == status_wait:
        print("status_wait = ", status)
        status_buff = tcp.receive()  # 통신받은 값은 값은 status값임
        if status_buff:
            print("buf =", status_buff)
            status = status_buff

    # Loop through the video frames
    while True:
        # cv2.namedWindow("YOLOv8 Tracking",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("depth yolo", cv2.WINDOW_NORMAL)
        find_word = book_serial_num
        print("book number =", find_word)
        print("book number2 =", book_serial_num)
        start_time = time.time()
        if realsense:
            results = model.track(RS.color_image,
                                  persist=True)  # Run YOLOv8 tracking on the frame, persisting tracks between frames
            book_detector.get_image(RS)
        else:  # video
            _, frame = cap.read()
            frame = cv2.resize(frame, (640, 384))
            results = model.track(frame, persist=True)
            book_detector.get_video_image(frame)

        if ChadChang:
            # status send depth : 제일 가까운 책의 depth를 추정
            if status == status_depth:
                print("status_depth = ", status)
                if len(results[0].boxes):
                    if not wait:
                        ret = book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data,
                                                     results[0].names)  # 마스크 검출
                        if ret == 0:
                            book_detector.center_book()  # 모멘트법으로 만들어서 가끔 오류있을 수 있음
                            bc_p, lc_p, bl_p = book_detector.BC_point, book_detector.LC_point, book_detector.BLC_point
                            cv2.circle(book_detector.depth_image, (int(lc_p[0]), int(lc_p[1])), 2, color_l, 3)
                            cv2.circle(book_detector.depth_image, (int(bc_p[0]), int(bc_p[1])), 2, color_b, 3)
                            cv2.circle(book_detector.depth_image, (int(bl_p[0]), int(bl_p[1])), 2, color_bl, 3)
                            bc_3d = book_detector.convert_3d(bc_p)
                            print("bc_3d", bc_3d[2])
                            if int(bc_3d[2]) > 0:
                                tcp.send(bc_3d[2])
                                wait = True
                            else:
                                pass
                    status_buff = tcp.receive()
                    if status_buff:
                        print("buf = ", status_buff)
                        status = status_buff
                        wait = False
            elif status == status_wait2:  # 처음 시작이 아닌 상태
                print("status_wait = ", status)
                status_buff = tcp.receive()  # 통신받은 값은 값은 status값임
                if status_buff:
                    print("buf_wait2 =", status_buff)
                    status = status_buff

                '''status1 : 책 읽으러 동작'''  # 조금 수정할 필요있을듯
            elif status == status_approach:  # 책 읽으러 동작
                print("status_approach = ", status)
                if not wait:
                    ret_r, H = runner.run_approach(book_detector, results, visual=True)
                    print("H mat = ", H)
                    if ret_r:  # 특징 추출 완료하면
                        tcp.send(H, matrix=True)
                        wait = True
                    elif (ret_r == False and H == 'b'):
                        print("H == b", H)
                        tcp.send(H)
                        wait = True
                status_buff = tcp.receive()
                if status_buff:
                    print("buf = true")
                    status = status_buff
                    wait = False

                    # 잠깐 쉬는 시간 가질까

                '''status2 : 글자 읽기 + 오차 계산'''
            elif status == status_readable:  # readable한지 판단 & 방향 통신
                print("status_readable", status)
                if not wait:
                    ret, dir = runner.run_readable(book_detector, results, visual=False)
                    if (not ret and dir != None):  # 아무것도 안 잡힐때
                        print(dir)
                        tcp.send(dir)
                        wait = True
                    else:  # 읽을 수 있을때
                        tcp.send(6.5)
                        print("readable")
                        wait = True
                status_buff = tcp.receive()
                # print("status2 =", status_buff)
                if status_buff:
                    print("buf = true", status_buff)
                    status = status_buff
                    wait = False

            elif status == status_read:  # 글씨 확인
                print("status_read", status)
                ret_r = runner.read_ocr(book_detector, results, find_word, visual=True)
                if ret_r:
                    status = status_appr2book
                    print("succeed to read")
                    tcp.send('s')  # 글씨 읽지 못한경우

                else:  # 글씨 읽기 실패했을 때
                    status = status_read_fail  # fail = 1
                    print("fail to read")
                    cv2.destroyWindow('color')
                    cv2.destroyWindow('depth')
                    cv2.destroyWindow('image')
                    tcp.send('f')  # status_read_fail
                    status = status_wait2  # 읽는데 실패하면 처음상태로

                # elif status == status_read_fail:

                '''status3: 해당 책 위치로 이동'''
            elif status == status_appr2book:  # 해당책 위치로 이동:
                print("status_appr2book", status)
                if not wait:
                    ret_r, H = runner.read_estimate(book_detector, results, visual=True)
                    print(ret_r, H)
                    if ret_r:  # 특징 추출 완료하면
                        wait = True
                        tcp.send(H, matrix=True)
                    # elif (ret_r == False) and (H == 'b'):
                    elif (ret_r == False) and (H == None):
                        # print(H)
                        print("fail to extracting")
                        tcp.send('b')
                        wait = True
                        cv2.destroyWindow('color')
                        cv2.destroyWindow('depth')
                        cv2.destroyWindow('image')
                    else:  # 추출이 실패하면
                        wait = False
                status_buff = tcp.receive()
                if status_buff:
                    print("buf = true", status_buff)
                    status = status_buff
                    wait = False
                    # print('status = 3->5', status)  #트래킹하는 부분은 보류됨
                    # status = '5'
                    # 잠깐 쉬는 시간 가질까

            # elif status == status_tracking: # 트래킹하면서 좌표 보내주기
            #     print("status_tracking", status)
            #     if results[0].boxes != None and results[0].boxes.id != None:
            #         annotated_frame = results[0].plot()  # Visualize the results on the frame
            #         if tracking:  # w 누르고 글자를 읽기 성공 시 트래킹
            #             book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)  # 마스크 검출
            #             label_id, book_id, label_idx, book_idx = book_detector.IDcheck(label_id, book_id)
            #             if label_id == -1 and book_id == -1:
            #                 tracking = False
            #                 print('tracking = False')
            #             else:
            #                 book_detector.tracking_getter(label_idx, book_idx, visualize=True)  # 글자 읽은 책, 글자 읽은 라벨지, 책+라벨지 트래킹
            #                 # 좌표 통신해 줌
            #
            #         cv2.imshow("YOLOv8 Tracking", annotated_frame)
            #     else:  # not tracking
            #         cv2.imshow("YOLOv8 Tracking", results[0].plot())

            elif status == status_align:  # 최종적으로 넣기전에 그리퍼 align 맞추
                print("status_align", status)
                if not wait:
                    lc_point, blc_point = book_detector.LC_point, book_detector.BLC_point
                    # print(lc_point,blc_point)
                    ang = book_detector.calc_ang(lc_point, blc_point, visual=True)
                    print("angle = ", ang)

                    if ang and ang != 'z':
                        tcp.send(ang)
                    elif ang == 'z':  # 각도 0일때
                        tcp.send(0)
                wait = True
                status_buff = tcp.receive()
                if status_buff:
                    print("buf = true")
                    # if -1 < ang < 1:
                    status = status_buff
                    wait = False
                    # print('status = 5->6', status)

            elif status == status_wait_insert:  # 넣고 있을때 통신 기다리는 모드
                print("status_wait_insert = ", status)
                status_buff = tcp.receive()
                if status_buff:
                    print("buf = true", status_buff)
                    status = status_buff  # fail of suceed

            # elif status == '7':  # 그리핑 성공 여부 판단
            #     status_buff = tcp.receive
            #     if status_buff:
            #         status = 10

            # elif status == status_Not_Read: # 해당 프레임에 책이 없을때 + 목표 책을 못 읽었을 경우도 있음(책 인식 모델이 매우 정확도가 높은 것이기 때문에 고려하지 않기)
            #     print("status_NotRead", status)
            #     if not wait:
            #         tcp.send(1) # not read symbol
            #         wait = True
            #     status_buff = tcp.receive()
            #     if status_buff:
            #         print("buf = true")
            #         status = status_buff
            #         wait = False

            # if not wait:
            #     print(runner.TF)
            #     # 보내는 값이 온전한지 확인
            #     # 통신 보내는 것필요
            #     if ret_r:  # 특징 추출 완료하면
            #         tcp.send(H, matrix=True)
            #     wait = True
            # status_buff = tcp.receive()
            # if status_buff:
            #     status = status_buff
            #     print('status = 1->2', status)
            #     wait = False

            elif status == status_end:  # 로봇팔 구동이 끝나고 유석이 쪽으로 신호줄때
                print("status_end")
                status_buff = tcp.receive()

            # elif status == 's':

        if results[0].boxes != None and results[0].boxes.id != None:
            annotated_frame = results[0].plot()  # Visualize the results on the frame
            if tracking:  # w 누르고 글자를 읽기 성공 시 트래킹
                book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names,
                                       tracking=tracking)  # 마스크 검출
                label_id, book_id, label_idx, book_idx = book_detector.IDcheck(label_id, book_id)
                if label_id == -1 and book_id == -1:
                    tracking = False
                    print('tracking = False')
                else:
                    book_detector.tracking_getter(label_idx, book_idx, visualize=True)  # 글자 읽은 책, 글자 읽은 라벨지, 책+라벨지 트래킹
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:  # not tracking
            cv2.imshow("YOLOv8 Tracking", results[0].plot())

        end_time = time.time()
        # print('처리시 간= ',end_time-start_time)

        # Break the loop if 'q' is pressed
        k_ = cv2.waitKey(1) & 0xFF
        if k_ == ord("q"):
            tcp.close()
            break

        elif k_ == ord('w'):  # w 누르면 프레임 멈추고 글자 읽기. 글자 읽기 성공 시 트래킹 시작
            if realsense:
                book_detector.get_image(RS)
            else:
                book_detector.get_video_image(frame)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names, tracking=tracking)
            ret, label_id, book_id = book_detector.ocr(word=find_word)
            if ret:
                tracking = True
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif k_ == ord('e'):
            if realsense:
                book_detector.get_image(RS)
            else:
                book_detector.get_video_image(frame)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names, tracking=tracking)
            book_detector.center_book(visualize=True)
            # book_detector.BLC_point
            # ret_r, H = runner.run_approach(book_detector, results, visual = True)
            cv2.waitKey(0)


def thread_book_number(args=None):
    number = Book_Number()
    executor_number = SingleThreadedExecutor()
    executor_number.add_node(number)
    try:
        executor_number.spin()
    finally:
        number.destroy_node()
        executor_number.shutdown()

class Book_Number(Node):  # msgê°’ì´ Trueì¸ì§€ Falseì— ìƒê´€ì—†ì´ ê·¸ëƒ¥ callbackí•¨ìˆ˜ë§Œ ì‹¤í–‰ì‹œí‚¤ëŠ” ìš©ë„
    def __init__(self):
        super().__init__("Book_Number")
        self.subscription = self.create_subscription(
            String,
            'book_num',
            self.listener_callback,
            10)


    def listener_callback(self, msg):
        global book_serial_num
        book_serial_num = msg.data  # ìŠ¤íŠ¸ë§íƒ€ìž… ì±…ì¼ë ¨ë²ˆí˜¸.
        # print('zzzz')
        # while(1):
        #     print('zzzzzzzzzzzzzz')
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best_v8.pt', help='yolov8 model path(s)')
    parser.add_argument('--source', type=str, default=None, help='source other than realsense')
    parser.add_argument('--realsense', type=bool, default=True, help='if you use realsense depth cam')
    parser.add_argument('--status', type=str, default='0', help='initial robot arm status')
    parser.add_argument('--find-word', type=int, default=623400, help='book label number to find')
    parser.add_argument('--book-detector', default=None, help='detectBook instance')
    parser.add_argument('--ip', type=str, default='127.0.1.1', help='tcp ip communication')
    parser.add_argument('--port', type=int, default=2014, help='tcp ip port')
    parser.add_argument('--tcp-available',type=bool, default=True, help='tcp available')
    parser.add_argument('--query', default=None, help=' ')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    rclpy.init()
    t1 = threading.Thread(target=thread_book_number)  # ìœ ì„-ì±… ì¼ë ¨ë²ˆí˜¸ ì „ë‹¬.
    t1.start()
    opt = parse_opt()
    librarian_main(opt)


    # while(1):
    #     print(book_serial_num)
