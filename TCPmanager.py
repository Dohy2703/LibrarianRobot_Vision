import numpy as np
import socket

# 통신프로토콜 :
class TCP:  # 통신 클래스 (파이썬 <-> Matlab) 파이썬 : client
    def __init__(self,ip,port,available = False):
        self.ip = ip
        self.port = port
        self.available = available
        self.recv_data = 0
        if available:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ip, port))
            self.client_socket.setblocking(False)

    def receive(self):  # 서버는 메세지가 올때까지 수신 대기중
        if self.available:
            try:
                data = self.client_socket.recv(100).decode()
                print(f'{data}')
                self.recv_data = int(str(data)[0:-1])
                return self.recv_data
                # print(self.recv_data[-1])
                # print(len('close'))
                # print(len(self.recv_data))
            except:
                return False
                # pass
    def send(self,message, matrix = False):  # string형태의 message]
        '''
        :param message: matrix 또는 Int 또는 Float
        :param matrix: matrix인지 아닌지
        '''
        if self.available:
            if matrix:
                message = np.reshape(message, 16)  # 1x16
                message = str(message[0:12])[2:-1]
                message = str.encode(message)
                self.client_socket.send(message)
                print('sent_mat=', message)
            else:  # int 일때
                message = str(message)
                message = str.encode(message)
                self.client_socket.send(message)
                print("sent_int = ", message)
    def close(self):
        if self.available:
            self.client_socket.close()


#asdf
class Robot_Arm(Node):  # 목적지 도착시 트리거됨. 로봇팔 동작 끝나면 return_origin 동작 트리거. -> 실제에선 목적지 도달시 트리거되어서 callback안에 로봇팔 동작코드 호출.
    def __init__(self):
        super().__init__("Robot_Arm")
        self.subscription = self.create_subscription(Bool,'Robot_Arm',self.listener_callback,5)
        # rate = self.create_rate(3)
        self.done = 1
        self.subscription  # prevent unused variable warning
        self.publisher_return = self.create_publisher(Bool, 'Return_Origin_Position',10)  # 원래라면 창민이꺼 동작 끝나면 return_origin_position 그쪽에서 보내줌

    def listener_callback(self, msg):
        self.get_logger().info('Robot_Arm')
        # rate.sleep()           # 이부분에 이제 로봇팔 동작 코드를 넣거나 이 노드 자체를 밖으로 빼거나 이 노드에서 창민이꺼를 호출하거나.
        self.done -= 1  # 왔다 갔다 하면 총 2번 신호를 받으므로, 이렇게 하면 2번째 마다. 즉 실제로 서가로 출발할 때에만 동작 가능함.
        if (self.done == 0):
            time.sleep(3)  # 실제론 이거 없어야함. 로봇팔 동작을 수행한다는걸 알기 위해 3초간 대기
            msg = Bool()
            self.publisher_return.publish(msg)
            self.done = 2