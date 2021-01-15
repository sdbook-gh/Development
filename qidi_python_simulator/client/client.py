import socket
import ssl
import cloud
import sys


class Client:

    def startSSLConnect(selfself, serverIp, serverPort):
        # context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # context.load_verify_locations('../cert/ca.crt')
        # context.load_cert_chain('./server.csr', './server_rsa_private.pem.unsecure')
        client_socket = socket.socket()
        client_socket.connect((serverIp, serverPort))
        print('connect to server')
        ssl_client = ssl.wrap_socket(client_socket,
                                     keyfile='./client_rsa_private.pem',
                                     certfile='./client.crt',
                                     # cert_reqs=ssl.CERT_REQUIRED,
                                     ca_certs='./ca.crt',
                                     ssl_version=ssl.PROTOCOL_TLS)
        print('connect ssl to server')
        # header_var = cloud.Header()
        # header_var.m_start_tag = cloud.Header.START_TAG_VALUE
        # header_var.m_remain_length = (0x00, 0x00, 0x00,)
        # header_var.m_type = 0x00
        # header_var.m_version = 0x00
        # header_var.m_timestamp_ms = 0x0102
        # header_var.m_timestamp_min = 0x01020304
        # total_size = header_var.calc_real_size()
        # requestBuffer = bytearray(total_size)
        # pos = header_var.fill_buffer(requestBuffer, 0)
        # ssl_client.send(requestBuffer)
        # print('send head to server')

        # header_var = cloud.Header()
        # head_buffer = ssl_client.recv(header_var.calc_real_size())
        # if len(head_buffer) == 0:
        #     return False
        # if head_buffer[0] != cloud.Header.START_TAG_VALUE:
        #     raise Exception('bad request')
        # print(f"receive msg from server")
        # header_var.parse_buffer(head_buffer, 0)
        # remainLen = cloud.decodeRemainLength(header_var.m_remain_length)
        # # print(f'remainLen:{remainLen}')
        # if remainLen > 0:
        #     command_buffer = ssl_client.recv(remainLen)
        # print(f"receive msg {header_var.m_type} from server")
        ssl_client.close()


if __name__ == '__main__':
    client1 = Client()
    client1.startSSLConnect('172.16.26.211', 51001)
    # client2 = Client()
    # client2.startSSLConnect('120.133.21.14', 51004)
