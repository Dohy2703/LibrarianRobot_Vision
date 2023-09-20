import numpy as np
import cv2
import math as m
# from TCPmanager import TCP
class Operator():
    def __init__(self):
        self.rot = np.array([])  # rotation matrix 3x3
        self.trev = np.array([])  # translation vector 1x3
        self.TF = np.array([])  # total transform
        self.num = 0  # book number
        # 상속
        self.BC_point = np.array([])  # 책의 중점(x,y)
        self.LC_point = np.array([])  # 레이블의 중점(x,y)
        self.BLC_point = np.array([])  # 책+레이블의 중점(x,y)

        self.B_corner = np.array([])  # 책의 코너점  _ minareaRect로 뽑은 corner점
        self.L_corner = np.array([])  # 레이블의 코너점_ minareaRect로 뽑은 corner점
        self.BL_corner = np.array([])  # 책과 레이블의 코너점_ minareaRect로 뽑은 corner점


    '''처음 구동할때'''
    def run_approach(self, book_detector, results, visual = False): # matlab에서 orientation 정보 안씀
        H = np.eye(4)
        if len(results[0].boxes):
            ret = book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            print("ret =", ret)
            if ret == 0: # 둘다 볼때
                book_detector.center_book()
                '''corner point'''
                b_corner0, l_corner0, bl_corner0 = book_detector.B_corner, book_detector.L_corner, book_detector.BL_corner  # corner point wrt [(book),(label),(book_label)]

                '''center point '''
                bc_p, lc_p, bl_p = book_detector.BC_point, book_detector.LC_point, book_detector.BLC_point
                cv2.circle(book_detector.depth_image, (int(lc_p[0]),int(lc_p[1])),2,(0,0,255),3)
                color_b, color_l, color_bl = (255, 0, 0), (0, 255, 0), (0, 0, 255)

                '''rearrange corner point'''
                bl_corner = book_detector.rearrange_all(bl_corner0, color_bl, visual=False)
                b_corner = book_detector.rearrange_all(b_corner0, color_b, visual=False)
                l_corner = book_detector.rearrange_all(l_corner0, color_b, visual=False)
                '''when error occur in the process of getting corner point'''


                if type(bl_corner) != bool and type(b_corner) != bool and type(l_corner) != bool:
                    # print('coner type before0=',type(b_corner),type(l_corner),type(bl_corner))
                    # '''make inner corner point'''  # corner_in =[P_LD_in, P_LU_in, P_RD_in, P_RU_in]
                    # print('coner before0=', b_corner, l_corner, bl_corner)
                    b_corner  = book_detector.inner_point(b_corner, color_b,visual)
                    l_corner = book_detector.inner_point(l_corner, color_l, visual)
                    bl_corner = book_detector.inner_point(bl_corner, color_bl,visual)  # convert inner point
                    # print('coner before1=',b_corner,l_corner,bl_corner)
                    b_corner2, l_corner2, bl_corner2 = b_corner.copy(), l_corner.copy(), bl_corner.copy()

                    '''corner-2d => corner-3d'''
                    b_corner3d, l_corner3d, bl_corner3d = book_detector.convert_3d_corner(
                        b_corner), book_detector.convert_3d_corner(l_corner), book_detector.convert_3d_corner(bl_corner)

                    # print("coner = ", b_corner3d, l_corner3d, bl_corner3d)
                    bc_3d, lc_3d, blc_3d = book_detector.convert_3d(bc_p), book_detector.convert_3d(
                        lc_p), book_detector.convert_3d(bl_p)
                    Trev = np.array([lc_3d])  # label을 기준으로 로봇을 구동함
                    self.trev = Trev
                    # print("Trev = ",Trev,Trev.shape)

                    '''calculate orientation: (input : arranged_corner_point)/ (output : 3x3 rotation matrix) '''
                    Rot = book_detector.calc_ori(bl_corner3d)
                    if type(Rot) == np.ndarray:
                        H[:3, :3] = Rot
                        H[:3, 3] = Trev

                    '''performance evaluation'''
                    bl_corner3d_copy = bl_corner3d.copy()
                    # print(bl_corner3d_copy)
                    bl_corner3d_copy[0][0] = bl_corner2[0][0]
                    bl_corner3d_copy[0][1] = bl_corner2[0][1]
                    bl_corner3d_copy[1][0] = bl_corner2[1][0]
                    bl_corner3d_copy[1][1] = bl_corner2[1][1]
                    bl_corner3d_copy[2][0] = bl_corner2[2][0]
                    bl_corner3d_copy[2][1] = bl_corner2[2][1]
                    bl_corner3d_copy[3][0] = bl_corner2[3][0]
                    bl_corner3d_copy[3][1] = bl_corner2[3][1]

                    # depth_buf = []
                    # for i in range(4):
                    #     depth_buf.append(bl_corner3d_copy[i][2])
                    #
                    # if (max(depth_buf) - min(depth_buf)) > 500: # 잘못됨 -> 뒤로
                    #     print("approach - deviation is too big")
                    #     print("depth = ", depth_buf)
                    #     return False, 'b'
                    # elif 0 in depth_buf: # 잘못됨-> 뒤로
                    #     print("approach - depth 0 problem")
                    #     print("depth = ", depth_buf)
                    #     return False, 'b'
                    # else:
                    Rot = book_detector.calc_ori(bl_corner3d)
                    self.rot = Rot
                    Rot2 = book_detector.calc_ori(bl_corner3d_copy)
                    if type(Rot2) == np.ndarray and type(
                            Trev) == np.ndarray:  # translation vector와 rotation벡터가 제대로 들어왔을때 연산 시작
                        null_m = np.array([[0, 0, 0, 1]])
                        T = np.around(np.concatenate([Rot, np.reshape(Trev,(3,1))], axis=1), 4)
                        H = np.concatenate([T, null_m], axis=0)
                        self.TF = H
                    else:
                        self.TF = False
                        return False, None
                    if visual:
                        cv2.imshow('depth', book_detector.depth_image)
                        cv2.imshow('image', book_detector.color_image)
                    return True, H
                else:
                    print("approach - coner point is wrong")
                    return False, None
            else:
                print('approach - book mask False')
                return False, None
        else:
            print('approach - not detected')
            return False, None

    '''label이 잘 보이는지 확인'''
    def run_readable(self, book_detector,results,visual = False):  #label이 잘 보이는지 확인, return 하면 center point 확인
        ret = book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
        if ret != 3:
            # print("ret= ", ret)
            # if ret:
            book_detector.center_book()
            LC_P = book_detector.LC_point
            if visual:
                cv2.circle(book_detector.color_image, (int(LC_P[0]), int(LC_P[1])), 2, (0, 0, 255), 2)
                cv2.imshow('color',book_detector.color_image)
            # print('shape=', book_detector.color_image.shape, book_detector.color_image.shape[0], book_detector.color_image.shape[1])
            # print('LC_P=', LC_P)
            if LC_P[1] > int(book_detector.depth_image.shape[0]*6/7):
                # print("cannot read")
                return False, 'd'  # 로봇팔 아래로 해라

            elif LC_P[1] < int(book_detector.depth_image.shape[0]/7):
                # print("cannot read")
                return False, 'u'  # 로봇팔 위로 해라
            elif ret == 2: # 레이블지가 안 보일때
                return False, 'd'
            else:
                return True, None
        print("None of mask is detected")
        return False, None



    '''글씨 읽는코드'''
    def read_ocr(self,book_detector, results, num ,visual = False):
        book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
        ret, label_id, book_id = book_detector.ocr(word=num)

        # 읽자마자 코너점이랑중점을 반환하느ㅡㄴ것인가
        self.BC_point = book_detector.BC_point
        self.LC_point = book_detector.LC_point
        self.BLC_point = book_detector.BLC_point

        self.B_corner = book_detector.B_corner
        self.L_corner = book_detector.L_corner
        self.BL_corner = book_detector.BL_corner

        if ret:
            return True
    '''글씨 읽고 해당 책의 pose 추정''' # ret이 true일
    def read_estimate(self, book_detector, results, visual=False):  # 검출된 책에 대한 pose 추정
        # 읽은 책에 대한 중점과 코너점을 인자에 저장해 놓음
        # ret = book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
        H = np.eye(4)
        # cv2.imshow('depth', book_detector.depth_image)
        # if ret:
        if len(results[0].boxes):
            '''corner point'''
            b_corner0, l_corner0, bl_corner0 = book_detector.B_corner, book_detector.L_corner, book_detector.BL_corner  # corner point wrt [(book),(label),(book_label)]

            '''center point '''
            bc_p, lc_p, bl_p = book_detector.BC_point, book_detector.LC_point, book_detector.BLC_point
            '''color'''
            color_b, color_l, color_bl = (255, 0, 0), (0, 255, 0), (0, 0, 255)

            '''rearrange corner point'''
            bl_corner = book_detector.rearrange_all(bl_corner0, color_bl, visual=False)
            b_corner = book_detector.rearrange_all(b_corner0, color_b, visual=False)
            l_corner = book_detector.rearrange_all(l_corner0, color_b, visual=False)  #

            '''when error occur in the process of getting corner point'''
            print("tyep1", bl_corner)
            print("tyep2", b_corner)
            print("type3", l_corner)
            if type(bl_corner) != bool and type(b_corner) != bool and type(l_corner) != bool:
                print('read_estimate', 7)
                # print('coner type before0=',type(b_corner),type(l_corner),type(bl_corner))
                # '''make inner corner point'''  # corner_in =[P_LD_in, P_LU_in, P_RD_in, P_RU_in]
                # print('coner before0=', b_corner, l_corner, bl_corner)
                b_corner, l_corner, bl_corner = book_detector.inner_point(b_corner, color_b,
                                                                          visual=False), book_detector.inner_point(
                    l_corner, color_l, visual=False), book_detector.inner_point(bl_corner, color_bl,
                                                                                visual=False)  # convert inner point
                # print('coner before1=',b_corner,l_corner,bl_corner)
                b_corner2, l_corner2, bl_corner2 = b_corner.copy(), l_corner.copy(), bl_corner.copy()
                '''calculate_ang 2d'''
                ang2d = book_detector.calc_ang(lc_p, bc_p, visual=True)
                # print(ang2d)

                '''corner-2d => corner-3d'''
                b_corner3d, l_corner3d, bl_corner3d = book_detector.convert_3d_corner(
                    b_corner), book_detector.convert_3d_corner(l_corner), book_detector.convert_3d_corner(bl_corner)
                # print("coner = ", b_corner3d, l_corner3d, bl_corner3d)
                bc_3d, lc_3d, blc_3d = book_detector.convert_3d(bc_p), book_detector.convert_3d(
                    lc_p), book_detector.convert_3d(bl_p)
                Trev = np.array([lc_3d])  # label을 기준으로 로봇을 구동함
                self.trev = Trev
                # print("Trev = ",Trev,Trev.shape)

                '''calculate orientation: (input : arranged_corner_point)/ (output : 3x3 rotation matrix) '''
                Rot = book_detector.calc_ori(bl_corner3d)
                print('read_estimate', 13)
                if type(Rot) == np.ndarray:
                    # H = np.eye(4)
                    H[:3, :3] = Rot
                    H[:3, 3] = Trev

                '''performance evaluation'''
                bl_corner3d_copy = bl_corner3d.copy()
                # print(bl_corner3d_copy)
                bl_corner3d_copy[0][0] = bl_corner2[0][0]
                bl_corner3d_copy[0][1] = bl_corner2[0][1]
                bl_corner3d_copy[1][0] = bl_corner2[1][0]
                bl_corner3d_copy[1][1] = bl_corner2[1][1]
                bl_corner3d_copy[2][0] = bl_corner2[2][0]
                bl_corner3d_copy[2][1] = bl_corner2[2][1]
                bl_corner3d_copy[3][0] = bl_corner2[3][0]
                bl_corner3d_copy[3][1] = bl_corner2[3][1]
                # depth_buf = []
                # for i in range(4):
                #     depth_buf.append(bl_corner3d_copy[i][2])
                #
                # if (max(depth_buf) - min(depth_buf)) > 500:  # 잘못됨 -> 뒤로
                #     print("approach - deviation is too big")
                #     print("depth = ", depth_buf)
                #     return False, 'b'
                # elif 0 in depth_buf:  # 잘못됨-> 뒤로
                #     print("approach - depth 0 problem")
                #     print("depth = ", depth_buf)
                #     return False, 'b'

                Rot = book_detector.calc_ori(bl_corner3d)
                print('read_estimate', 14)
                self.rot = Rot
                Rot2 = book_detector.calc_ori(bl_corner3d_copy)
                print('read_estimate', 15)
                print("Rot = ", Rot)
                print("Rot2 = ", Rot2)
                if type(Rot2) == np.ndarray and type(
                        Trev) == np.ndarray:  # translation vector와 rotation벡터가 제대로 들어왔을때 연산 시작
                    # print("euler rot=",np.round(book_detector.euler_rot([math.pi, 0, 0])*Rot,4))
                    print('read_estimate', 16)
                    book_detector.visual_coord(book_detector.euler_rot([m.pi, 0, 0]) * Rot,
                                               Trev * 10)  # drawFramesAxes has different coordinate system from what is set before
                    print('read_estimate', 17)
                    book_detector.visual_coord(book_detector.euler_rot([m.pi, 0, 0]) * Rot,
                                               Trev)  # drawFramesAxes has different coordinate system from what is set before
                    print('read_estimate', 18)
                    null_m = np.array([[0, 0, 0, 1]])

                    T = np.around(np.concatenate([Rot, np.reshape(Trev, (3, 1))], axis=1), 4)
                    print('read_estimate', 19)
                    H = np.concatenate([T, null_m], axis=0)
                    print('read_estimate', 20)
                    self.TF = H
                else:
                    self.TF = False
                    return False, None
                if visual:
                    cv2.imshow('depth', book_detector.depth_image)
                    cv2.imshow('image', book_detector.color_image)
                return True, H
            else:
                print("corner = ",bl_corner, b_corner, l_corner)
                return False, None
        else:
            return False, None
        # else:
        #     return False, None
    '''글씨 읽고 트래킹하는 코드'''  # 책의 2차원 위치를 보내줌.
    def read_tracking(self,book_detector, results, visual=False):
        annotated_frame = results[0].plot()  # Visualize the results on the frame
        '''책읽고 그 책에 맞는 마스크를 반환함. +  그 책에 대한 pose를 구함'''
        if results[0].boxes != None and results[0].boxes.id != None:
            label_id = book_detector.label_idx
            book_id = book_detector.book_idx
            annotated_frame = results[0].plot()  # Visualize the results on the frame
            book_idx = results[0].boxes.id.int().cpu().tolist().index(book_id)
            label_idx = results[0].boxes.id.int().cpu().tolist().index(label_id)
            book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
            book_detector.tracking_getter(label_idx, book_idx, visualize=True)

            if visual:  # visualize the mask
                cv2.imshow('trackB', results[0].masks.data[book_idx].cpu().numpy())
                cv2.imshow('trackL', results[0].masks.data[label_idx].cpu().numpy())

                cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            cv2.imshow("YOLOv8 Tracking", results[0].plot())

    '''밀어넣는 과정'''
    def run_push(self):
        pass




    # def run3(self,book_detector, results, visual = False):
    #     book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names

        # return ret
