import socket
import ssl
import cloud
import threading
import ctypes


class Server(threading.Thread):
    def __init__(self):
        self.server_socket = None

    def startSSLListen(self, port):
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.server_socket.bind(('', port))
        self.server_socket.listen(5)
        while True:
            try:
                client_socket, client_addr = self.server_socket.accept()
                print(f"Connection from: {client_addr}")
                sslclient = ssl.wrap_socket(client_socket,
                                            keyfile='./server.key',
                                            certfile='./server.crt',
                                            ca_certs='./ca.crt',
                                            cert_reqs=ssl.CERT_REQUIRED,
                                            server_side=True,
                                            ssl_version=ssl.PROTOCOL_TLS)
                status = True
                while status:
                    status = self.processCloudMessage(sslclient)
            except Exception as e:
                print(e)

    def processCloudMessage(self, sslclient) -> bool:
        try:
            header_var = cloud.Header()
            head_buffer = sslclient.recv(header_var.calc_real_size())
            if len(head_buffer) == 0:
                return False
            if head_buffer[0] != cloud.Header.START_TAG_VALUE:
                raise Exception('bad request')
            print(f"receive msg from client")
            header_var.parse_buffer(head_buffer, 0)
            remainLen = cloud.decodeRemainLength(header_var.m_remain_length)
            # print(f'remainLen:{remainLen}')
            if remainLen > 0:
                command_buffer = sslclient.recv(remainLen)
            if header_var.m_type == cloud.VehicleToCloudRun.TYPE_VALUE:
                print(f'receive VehicleToCloudRun')
                vehicleToCloudRun = cloud.VehicleToCloudRun()
                vehicleToCloudRun.parse_buffer(command_buffer, 0)

                header_var = cloud.Header()
                header_var.m_start_tag = cloud.Header.START_TAG_VALUE
                header_var.m_remain_length = (0x00, 0x00, 0x00,)
                header_var.m_type = 0x00
                header_var.m_version = 0x00
                header_var.m_timestamp_ms = 0x0102
                header_var.m_timestamp_min = 0x01020304
                total_size = header_var.calc_real_size()
                responseBuffer = bytearray(total_size)
                pos = header_var.fill_buffer(responseBuffer, 0)
                sslclient.send(responseBuffer)
            elif header_var.m_type == cloud.VehicleToCloudReq.TYPE_VALUE:
                print(f'receive VehicleToCloudReq')
                vehicleToCloudReq_var = cloud.VehicleToCloudReq()
                vehicleToCloudReq_var.parse_buffer(command_buffer, 0)
                cloudToVehicleReqRes_var = cloud.CloudToVehicleReqRes()
                cloudToVehicleReqRes_var.m_resLen = ctypes.c_int(0)
                header_var = cloud.Header()
                header_var.m_start_tag = cloud.CloudToVehicleReqRes.TYPE_VALUE
                header_var.m_remain_length = cloud.encodeRemainLength(cloudToVehicleReqRes_var.calc_real_size())
                header_var.m_type = cloud.CloudToVehicleReqRes.TYPE_VALUE
                header_var.m_version = 0x00
                header_var.m_timestamp_ms = 0x0102
                header_var.m_timestamp_min = 0x01020304
                total_size = header_var.calc_real_size() + cloudToVehicleReqRes_var.calc_real_size()
                responseBuffer = bytearray(total_size)
                pos = header_var.fill_buffer(responseBuffer, 0)
                pos = cloudToVehicleReqRes_var.fill_buffer(responseBuffer, header_var.calc_real_size())
                sslclient.send(responseBuffer)
                print(f'send CloudToVehicleReqRes to client')
            else:
                print(f'receive unknown msg:{header_var.m_type}')
                header_var = cloud.Header()
                header_var.m_start_tag = cloud.Header.START_TAG_VALUE
                header_var.m_remain_length = (0x00, 0x00, 0x00,)
                header_var.m_type = 0x00
                header_var.m_version = 0x00
                header_var.m_timestamp_ms = 0x0102
                header_var.m_timestamp_min = 0x01020304
                total_size = header_var.calc_real_size()
                responseBuffer = bytearray(total_size)
                pos = header_var.fill_buffer(responseBuffer, 0)
                sslclient.send(responseBuffer)

            return True
        except Exception as e:
            print(e)
            return False


def startListen1():
    server1 = Server()
    server1.startSSLListen(15001)


def startListen2():
    server2 = Server()
    server2.startSSLListen(15004)


if __name__ == '__main__':
    thread1 = threading.Thread(target=startListen1)
    # thread2 = threading.Thread(target=startListen2)
    thread1.start()
    # thread2.start()
    thread1.join()
    # thread2.join()
