import struct
import logging
import ctypes
from .Constant import Constant


class VehicleToCloudReq:
    TYPE_VALUE = 0x36
    VERSION_VALUE = 0x02

    __logger = logging.getLogger('VehicleToCloudReq')

    def __init__(self):
        self.total_size = None
        self.m_vehicleId = None
        self.m_ctlMode = None
        self.m_reqLen: ctypes.c_int = None
        self.m_funcReq = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('8s')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')

        if self.m_reqLen and self.m_reqLen.value > 0:
            if not self.m_funcReq:
                raise Exception('m_funcReq is not set')
            self.total_size += self.m_funcReq.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        vehicleId_var = struct.unpack_from('8s', buffer, pos)[0]
        VehicleToCloudReq.__logger.debug(f'parse get vehicleId_var:{vehicleId_var}')
        self.m_vehicleId = vehicleId_var
        pos += struct.calcsize('8s')
        ctlMode_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudReq.__logger.debug(f'parse get ctlMode_var:{ctlMode_var}')
        self.m_ctlMode = ctlMode_var
        pos += struct.calcsize('B')
        reqLen_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudReq.__logger.debug(f'parse get reqLen_var:{reqLen_var}')
        self.m_reqLen = reqLen_var
        pos += struct.calcsize('B')

        if self.m_reqLen and self.m_reqLen.value > 0:
            self.m_funcReq = funcReq_VehicleToCloudReq(self.m_reqLen)
            if pos + self.m_funcReq.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_funcReq > buffer size')
            pos = self.m_funcReq.parse_buffer(buffer, pos)
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        vehicleId_var = self.m_vehicleId
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '8s', buffer, pos, vehicleId_var)
        VehicleToCloudReq.__logger.debug(f'fill vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        ctlMode_var = self.m_ctlMode
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, ctlMode_var)
        VehicleToCloudReq.__logger.debug(f'fill ctlMode_var:{ctlMode_var}')
        pos += struct.calcsize('B')
        reqLen_var = self.m_reqLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, reqLen_var)
        VehicleToCloudReq.__logger.debug(f'fill reqLen_var:{reqLen_var}')
        pos += struct.calcsize('B')

        if self.m_reqLen and self.m_reqLen.value > 0:
            if not self.m_funcReq:
                raise Exception('m_funcReq is not set')
            pos = self.m_funcReq.fill_buffer(buffer, pos)
        return pos


class funcReq_VehicleToCloudReq:
    __logger = logging.getLogger('funcReq_VehicleToCloudReq')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_funcReq = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
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
            funcReq_var = struct.unpack_from('B', buffer, pos)[0]
            self.m_funcReq.append(funcReq_var)
            funcReq_VehicleToCloudReq.__logger.debug(f'parse get funcReq_var:{funcReq_var}')
            pos += struct.calcsize('B')
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos

        if len(self.m_funcReq) != self.parentCount.value:
            raise Exception('parentCount mismatch m_funcReq count')
        for i in range(self.parentCount.value):
            funcReq_var = self.m_funcReq[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, funcReq_var)
            funcReq_VehicleToCloudReq.__logger.debug(f'fill funcReq_var:{funcReq_var}')
            pos += struct.calcsize('B')
        return pos
