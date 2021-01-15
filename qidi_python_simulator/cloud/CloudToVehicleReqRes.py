import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleReqRes:
    TYPE_VALUE = 0x37
    VERSION_VALUE = 0x02

    __logger = logging.getLogger('CloudToVehicleReqRes')

    def __init__(self):
        self.total_size = None
        self.m_resLen: ctypes.c_int = None
        self.m_funcRes = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')

        if self.m_resLen and self.m_resLen.value > 0:
            if not self.m_funcRes:
                raise Exception('m_funcRes is not set')
            self.total_size += self.m_funcRes.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        resLen_var = struct.unpack_from('B', buffer, pos)[0]
        CloudToVehicleReqRes.__logger.debug(f'parse get resLen_var:{resLen_var}')
        self.m_resLen = resLen_var
        pos += struct.calcsize('B')

        if self.m_resLen and self.m_resLen.value > 0:
            self.m_funcRes = funcRes_CloudToVehicleReqRes(self.m_resLen)
            if pos + self.m_funcRes.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_funcRes > buffer size')
            pos = self.m_funcRes.parse_buffer(buffer, pos)
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        resLen_var = self.m_resLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, resLen_var)
        CloudToVehicleReqRes.__logger.debug(f'fill resLen_var:{resLen_var}')
        pos += struct.calcsize('B')

        if self.m_resLen and self.m_resLen.value > 0:
            if not self.m_funcRes:
                raise Exception('m_funcRes is not set')
            pos = self.m_funcRes.fill_buffer(buffer, pos)
        return pos


class funcRes_CloudToVehicleReqRes:
    __logger = logging.getLogger('funcRes_CloudToVehicleReqRes')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_funcRes = []

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
            funcRes_var = struct.unpack_from('B', buffer, pos)[0]
            self.m_funcRes.append(funcRes_var)
            funcRes_CloudToVehicleReqRes.__logger.debug(f'parse get funcRes_var:{funcRes_var}')
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

        if len(self.m_funcRes) != self.parentCount.value:
            raise Exception('parentCount mismatch m_funcRes count')
        for i in range(self.parentCount.value):
            funcRes_var = self.m_funcRes[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, funcRes_var)
            funcRes_CloudToVehicleReqRes.__logger.debug(f'fill funcRes_var:{funcRes_var}')
            pos += struct.calcsize('B')
        return pos
