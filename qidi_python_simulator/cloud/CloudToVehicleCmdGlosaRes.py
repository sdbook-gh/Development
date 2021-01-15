import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleCmdGlosaRes:
    __logger = logging.getLogger('CloudToVehicleCmdGlosaRes')

    def __init__(self):
        self.total_size = None
        self.m_cmdType = None
        self.m_uuid = None
        self.m_doFlag = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('36s')
        self.total_size += struct.calcsize('B')
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        cmdType_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'parse get cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = struct.unpack_from('36s', buffer, pos)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'parse get uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        doFlag_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'parse get doFlag_var:{doFlag_var}')
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
        cmdType_var = self.m_cmdType
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, cmdType_var)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'fill cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = self.m_uuid
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '36s', buffer, pos, uuid_var)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'fill uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        doFlag_var = self.m_doFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, doFlag_var)
        CloudToVehicleCmdGlosaRes.__logger.debug(f'fill doFlag_var:{doFlag_var}')
        pos += struct.calcsize('B')
        return pos
