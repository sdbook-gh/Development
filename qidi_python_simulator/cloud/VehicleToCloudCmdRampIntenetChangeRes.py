import struct
import logging
import ctypes
from .Constant import Constant


class VehicleToCloudCmdRampIntenetChangeRes:
    __logger = logging.getLogger('VehicleToCloudCmdRampIntenetChangeRes')

    def __init__(self):
        self.total_size = None
        self.m_cmdType = None
        self.m_seq = None
        self.m_doFlag = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('Q')
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
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'parse get cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        seq_var = struct.unpack_from('Q', buffer, pos)
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'parse get seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        doFlag_var = struct.unpack_from('B', buffer, pos)
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'parse get doFlag_var:{doFlag_var}')
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
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'fill cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        seq_var = self.m_seq
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'Q', buffer, pos, seq_var)
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'fill seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        doFlag_var = self.m_doFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, doFlag_var)
        VehicleToCloudCmdRampIntenetChangeRes.__logger.debug(f'fill doFlag_var:{doFlag_var}')
        pos += struct.calcsize('B')
        return pos
