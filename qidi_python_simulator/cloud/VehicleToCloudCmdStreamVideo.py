import struct
import logging
import ctypes
from .Constant import Constant


class VehicleToCloudCmdStreamVideo:
    __logger = logging.getLogger('VehicleToCloudCmdStreamVideo')

    def __init__(self):
        self.total_size = None
        self.m_cmdType = None
        self.m_uuid = None
        self.m_camIdLen = None
        self.m_camId = None
        self.m_videoQual = None
        self.m_urlAddrLen = None
        self.m_urlAddr = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('36s')
        self.total_size += struct.calcsize('B')

        if self.m_camIdLen.value > 0:
            if not self.m_camId:
                raise Exception('m_camId is not set')
            self.total_size += self.m_camId.calc_real_size()
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')

        if self.m_urlAddrLen.value > 0:
            if not self.m_urlAddr:
                raise Exception('m_urlAddr is not set')
            self.total_size += self.m_urlAddr.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        cmdType_var = struct.unpack_from('B', buffer, pos)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = struct.unpack_from('36s', buffer, pos)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        camIdLen_var = struct.unpack_from('B', buffer, pos)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get camIdLen_var:{camIdLen_var}')
        pos += struct.calcsize('B')

        if self.m_camIdLen.value > 0:
            self.m_camId = camId_VehicleToCloudCmdStreamVideo(self.m_camIdLen)
            if pos + self.m_camId.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_camId > buffer size')
            pos = self.m_camId.parse_buffer(buffer, pos)
        videoQual_var = struct.unpack_from('B', buffer, pos)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get videoQual_var:{videoQual_var}')
        pos += struct.calcsize('B')
        urlAddrLen_var = struct.unpack_from('B', buffer, pos)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get urlAddrLen_var:{urlAddrLen_var}')
        pos += struct.calcsize('B')

        if self.m_urlAddrLen.value > 0:
            self.m_urlAddr = urlAddr_VehicleToCloudCmdStreamVideo(self.m_urlAddrLen)
            if pos + self.m_urlAddr.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_urlAddr > buffer size')
            pos = self.m_urlAddr.parse_buffer(buffer, pos)
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        cmdType_var = self.m_cmdType
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, cmdType_var)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'fill cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = self.m_uuid
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '36s', buffer, pos, uuid_var)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'fill uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        camIdLen_var = self.m_camIdLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, camIdLen_var)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'fill camIdLen_var:{camIdLen_var}')
        pos += struct.calcsize('B')

        if self.m_camIdLen.value > 0:
            if not self.m_camId:
                raise Exception('m_camId is not set')
            pos = self.m_camId.fill_buffer(buffer, pos)
        videoQual_var = self.m_videoQual
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, videoQual_var)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'fill videoQual_var:{videoQual_var}')
        pos += struct.calcsize('B')
        urlAddrLen_var = self.m_urlAddrLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, urlAddrLen_var)
        VehicleToCloudCmdStreamVideo.__logger.debug(f'fill urlAddrLen_var:{urlAddrLen_var}')
        pos += struct.calcsize('B')

        if self.m_urlAddrLen.value > 0:
            if not self.m_urlAddr:
                raise Exception('m_urlAddr is not set')
            pos = self.m_urlAddr.fill_buffer(buffer, pos)
        return pos

class camId_VehicleToCloudCmdStreamVideo:
    __logger = logging.getLogger('camId_VehicleToCloudCmdStreamVideo')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_camId = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('s')
        self.total_size *= self.parentCount.value
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        for i in range(self.parentCount.value):
            camId_var = struct.unpack_from('s', buffer, pos)
            self.m_camId.append(camId_var)
            camId_VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get camId_var:{camId_var}')
            pos += struct.calcsize('s')
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos

        if len(self.m_camId) != self.parentCount.value:
            raise Exception('parentCount mismatch m_camId count')
        for i in range(self.parentCount.value):
            camId_var = self.m_camId[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 's', buffer, pos, camId_var)
            camId_VehicleToCloudCmdStreamVideo.__logger.debug(f'fill camId_var:{camId_var}')
            pos += struct.calcsize('s')
        return pos

class urlAddr_VehicleToCloudCmdStreamVideo:
    __logger = logging.getLogger('urlAddr_VehicleToCloudCmdStreamVideo')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_urlAddr = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('s')
        self.total_size *= self.parentCount.value
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        for i in range(self.parentCount.value):
            urlAddr_var = struct.unpack_from('s', buffer, pos)
            self.m_urlAddr.append(urlAddr_var)
            urlAddr_VehicleToCloudCmdStreamVideo.__logger.debug(f'parse get urlAddr_var:{urlAddr_var}')
            pos += struct.calcsize('s')
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos

        if len(self.m_urlAddr) != self.parentCount.value:
            raise Exception('parentCount mismatch m_urlAddr count')
        for i in range(self.parentCount.value):
            urlAddr_var = self.m_urlAddr[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 's', buffer, pos, urlAddr_var)
            urlAddr_VehicleToCloudCmdStreamVideo.__logger.debug(f'fill urlAddr_var:{urlAddr_var}')
            pos += struct.calcsize('s')
        return pos
