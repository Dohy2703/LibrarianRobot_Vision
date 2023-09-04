import numpy as np
import cv2
import math as m


class Operator():
    def __init__(self):
        self.rot = np.array([]) # rotation matrix 3x3
        self.trev = np.array([]) # translation vector 1x3
        self.TF = np.array([]) # total transform
        self.num = 0 # book number

        # 상속
        self.BC_point = np.array([])  # 책의 중점(x,y)
        self.LC_point = np.array([])  # 레이블의 중점(x,y)
        self.BLC_point = np.array([])  # 책+레이블의 중점(x,y)

        self.B_corner = np.array([])  # 책의 코너점  _ minareaRect로 뽑은 corner점
        self.L_corner = np.array([])  # 레이블의 코너점_ minareaRect로 뽑은 corner점
        self.BL_corner = np.array([])  # 책과 레이블의 코너점_ minareaRect로 뽑은 corner점

    '''처음 구동할때'''
    def run_approach(self,book_detector,results,visual = True):
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

                color_b, color_l, color_bl = (255, 0, 0), (0, 255, 0), (0, 0, 255)

                '''rearrange corner point'''
                bl_corner = book_detector.rearrange_all(bl_corner0, color_bl, visual=False)
                b_corner = book_detector.rearrange_all(b_corner0, color_b, visual=False)
                l_corner = book_detector.rearrange_all(l_corner0, color_b, visual=False)  #
                '''when error occur in the process of getting corner point'''


                if type(bl_corner) != bool and type(b_corner) != bool and type(l_corner) != bool:
                    # print('coner type before0=',type(b_corner),type(l_corner),type(bl_corner))
                    # '''make inner corner point'''  # corner_in =[P_LD_in, P_LU_in, P_RD_in, P_RU_in]
                    # print('coner before0=', b_corner, l_corner, bl_corner)
                    b_corner, l_corner, bl_corner = book_detector.inner_point(b_corner, color_b,
                                                                              visual=False), book_detector.inner_point(
                        l_corner, color_l, visual=False), book_detector.inner_point(bl_corner, color_bl,
                                                                                    visual=True)  # convert inner point
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

                    if type(Rot) == np.ndarray:
                        H[:3, :3] = Rot
                        H[:3, 3] = Trev

                    '''performance evaluation'''
                    bl_corner3d_copy = bl_corner3d.copy()
                    print(bl_corner3d_copy)
                    bl_corner3d_copy[0][0] = bl_corner2[0][0]
                    bl_corner3d_copy[0][1] = bl_corner2[0][1]
                    bl_corner3d_copy[1][0] = bl_corner2[1][0]
                    bl_corner3d_copy[1][1] = bl_corner2[1][1]
                    bl_corner3d_copy[2][0] = bl_corner2[2][0]
                    bl_corner3d_copy[2][1] = bl_corner2[2][1]
                    bl_corner3d_copy[3][0] = bl_corner2[3][0]
                    bl_corner3d_copy[3][1] = bl_corner2[3][1]

                    Rot = book_detector.calc_ori(bl_corner3d)
                    self.rot = Rot
                    Rot2 = book_detector.calc_ori(bl_corner3d_copy)


                    if type(Rot2) == np.ndarray and type(
                            Trev) == np.ndarray:  # translation vector와 rotation벡터가 제대로 들어왔을때 연산 시작
                        # print("euler rot=",np.round(book_detector.euler_rot([math.pi, 0, 0])*Rot,4))
                        book_detector.visual_coord(book_detector.euler_rot([m.pi,0,0])*Rot,Trev*10) # drawFramesAxes has different coordinate system from what is set before
                        book_detector.visual_coord(book_detector.euler_rot([m.pi, 0, 0]) * Rot, Trev)  # drawFramesAxes has different coordinate system from what is set before
                        null_m = np.array([[0, 0, 0, 1]])
                        # print(Rot,Rot.shape)
                        # print(Trev, Trev.shape)
                        T = np.around(np.concatenate([Rot, np.reshape(Trev,(3,1))], axis=1), 4)
                        H = np.concatenate([T, null_m], axis=0)

                        self.TF = H
                    else:
                        self.TF = False
                        return False
                    if visual:
                        cv2.imshow('depth', book_detector.depth_image)
                        cv2.imshow('image', book_detector.color_image)
                    return True, H
                    # def run1(self,DetectBook, results): #

    '''글씨 읽는코드'''
    def read_ocr(self,book_detector, results, num ,visual = False):  # 중점, 코너점 반환
        book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)
        ret, label_id, book_id = book_detector.ocr(word=num)


        self.BC_point = book_detector.BC_point
        self.LC_point = book_detector.LC_point
        self.BLC_point = book_detector.BLC_point

        self.B_corner = book_detector.B_corner
        self.L_corner = book_detector.L_corner
        self.BL_corner = book_detector.BL_corner

        if ret:
            return True

    # def read_capture(self,book_detector, results): # 한 프레임을 분석해 코너점, 중심점 반환 후 estimation에서 이를 분석 => 그것을 내부 멤버변수에 업데이트하기
    #     book_detector.get_mask(results[0].boxes.cpu(), results[0].masks.data, results[0].names)  # 마스크 검출

    '''글씨 읽고 해당 책의 pose 추정''' # ret이 true일
    def read_estimate(self, book_detector, results,visual=False):  # 검출된 책에 대한 pose 추정
        # 읽은 책에 대한 중점과 코너점을 인자에 저장해 놓음
        # 여기에서 코너 점 추출함
        self.read_capture(book_detector, results)
        book_detector.center_book()
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
        if type(bl_corner) != bool and type(b_corner) != bool and type(l_corner) != bool:
            # print('coner type before0=',type(b_corner),type(l_corner),type(bl_corner))
            # '''make inner corner point'''  # corner_in =[P_LD_in, P_LU_in, P_RD_in, P_RU_in]
            # print('coner before0=', b_corner, l_corner, bl_corner)
            b_corner, l_corner, bl_corner = book_detector.inner_point(b_corner, color_b,
                                                                      visual=False), book_detector.inner_point(
                l_corner, color_l, visual=False), book_detector.inner_point(bl_corner, color_bl,
                                                                            visual=True)  # convert inner point
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

            if type(Rot) == np.ndarray:
                H = np.eye(4)
                H[:3, :3] = Rot
                H[:3, 3] = Trev

            '''performance evaluation'''
            bl_corner3d_copy = bl_corner3d.copy()
            print(bl_corner3d_copy)
            bl_corner3d_copy[0][0] = bl_corner2[0][0]
            bl_corner3d_copy[0][1] = bl_corner2[0][1]
            bl_corner3d_copy[1][0] = bl_corner2[1][0]
            bl_corner3d_copy[1][1] = bl_corner2[1][1]
            bl_corner3d_copy[2][0] = bl_corner2[2][0]
            bl_corner3d_copy[2][1] = bl_corner2[2][1]
            bl_corner3d_copy[3][0] = bl_corner2[3][0]
            bl_corner3d_copy[3][1] = bl_corner2[3][1]

            Rot = book_detector.calc_ori(bl_corner3d)
            self.rot = Rot
            Rot2 = book_detector.calc_ori(bl_corner3d_copy)

            if type(Rot2) == np.ndarray and type(
                    Trev) == np.ndarray:  # translation vector와 rotation벡터가 제대로 들어왔을때 연산 시작
                # print("euler rot=",np.round(book_detector.euler_rot([math.pi, 0, 0])*Rot,4))
                book_detector.visual_coord(book_detector.euler_rot([m.pi, 0, 0]) * Rot,
                                           Trev * 10)  # drawFramesAxes has different coordinate system from what is set before
                book_detector.visual_coord(book_detector.euler_rot([m.pi, 0, 0]) * Rot,
                                           Trev)  # drawFramesAxes has different coordinate system from what is set before
                null_m = np.array([[0, 0, 0, 1]])

                T = np.around(np.concatenate([Rot, np.reshape(Trev, (3, 1))], axis=1), 4)
                H = np.concatenate([T, null_m], axis=0)

                self.TF = H
            else:
                self.TF = False
                return False
            if visual:
                cv2.imshow('depth', book_detector.depth_image)
                cv2.imshow('image', book_detector.color_image)
            return True, H
            # def run1(self,DetectBook, results): #

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
