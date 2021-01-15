import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleCmd:
    TYPE_VALUE = 0x3C
    VERSION_VALUE = 0x02

    __logger = logging.getLogger('CloudToVehicleCmd')

    def __init__(self):
        self.total_size = None
        self.m_vehicleId = None
        self.m_dataLen = None
        self.m_cmdData = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('8s')
        self.total_size += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            if not self.m_cmdData:
                raise Exception('m_cmdData is not set')
            self.total_size += self.m_cmdData.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        vehicleId_var = struct.unpack_from('8s', buffer, pos)
        CloudToVehicleCmd.__logger.debug(f'parse get vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        dataLen_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmd.__logger.debug(f'parse get dataLen_var:{dataLen_var}')
        pos += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            self.m_cmdData = cmdData_CloudToVehicleCmd(self.m_dataLen)
            if pos + self.m_cmdData.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_cmdData > buffer size')
            pos = self.m_cmdData.parse_buffer(buffer, pos)
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
        CloudToVehicleCmd.__logger.debug(f'fill vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        dataLen_var = self.m_dataLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, dataLen_var)
        CloudToVehicleCmd.__logger.debug(f'fill dataLen_var:{dataLen_var}')
        pos += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            if not self.m_cmdData:
                raise Exception('m_cmdData is not set')
            pos = self.m_cmdData.fill_buffer(buffer, pos)
        return pos


class cmdData_CloudToVehicleCmd:
    __logger = logging.getLogger('cmdData_CloudToVehicleCmd')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_cmdData = []

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
            cmdData_var = struct.unpack_from('B', buffer, pos)
            self.m_cmdData.append(cmdData_var)
            cmdData_CloudToVehicleCmd.__logger.debug(f'parse get cmdData_var:{cmdData_var}')
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

        if len(self.m_cmdData) != self.parentCount.value:
            raise Exception('parentCount mismatch m_cmdData count')
        for i in range(self.parentCount.value):
            cmdData_var = self.m_cmdData[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, cmdData_var)
            cmdData_CloudToVehicleCmd.__logger.debug(f'fill cmdData_var:{cmdData_var}')
            pos += struct.calcsize('B')
        return pos
