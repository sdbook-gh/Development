import struct
import logging
from .Constant import Constant


class Header:
    START_TAG_VALUE = 0xF2

    __logger = logging.getLogger('Header')

    def __init__(self):
        self.total_size = None
        self.m_start_tag = None
        self.m_remain_length = None
        self.m_type = None
        self.m_version = None
        self.m_timestamp_ms = None
        self.m_timestamp_min = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('3B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('I')
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        start_tag_var = struct.unpack_from('B', buffer, pos)[0]
        Header.__logger.debug(f'parse get start_tag_var:{start_tag_var}')
        self.m_start_tag = start_tag_var
        pos += struct.calcsize('B')
        remain_length_var = struct.unpack_from('3B', buffer, pos)
        Header.__logger.debug(f'parse get remain_length_var:{remain_length_var}')
        self.m_remain_length = remain_length_var
        pos += struct.calcsize('3B')
        type_var = struct.unpack_from('B', buffer, pos)[0]
        Header.__logger.debug(f'parse get type_var:{type_var}')
        self.m_type = type_var
        pos += struct.calcsize('B')
        version_var = struct.unpack_from('B', buffer, pos)[0]
        Header.__logger.debug(f'parse get version_var:{version_var}')
        self.m_version = version_var
        pos += struct.calcsize('B')
        timestamp_ms_var = struct.unpack_from('H', buffer, pos)[0]
        Header.__logger.debug(f'parse get timestamp_ms_var:{timestamp_ms_var}')
        self.m_timestamp_ms = timestamp_ms_var
        pos += struct.calcsize('H')
        timestamp_min_var = struct.unpack_from('I', buffer, pos)[0]
        Header.__logger.debug(f'parse get timestamp_min_var:{timestamp_min_var}')
        self.m_timestamp_min = timestamp_min_var
        pos += struct.calcsize('I')
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        start_tag_var = self.m_start_tag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, start_tag_var)
        Header.__logger.debug(f'fill start_tag_var:{start_tag_var}')
        pos += struct.calcsize('B')
        remain_length_var = self.m_remain_length
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '3B', buffer, pos, *remain_length_var)
        Header.__logger.debug(f'fill remain_length_var:{remain_length_var}')
        pos += struct.calcsize('3B')
        type_var = self.m_type
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, type_var)
        Header.__logger.debug(f'fill type_var:{type_var}')
        pos += struct.calcsize('B')
        version_var = self.m_version
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, version_var)
        Header.__logger.debug(f'fill version_var:{version_var}')
        pos += struct.calcsize('B')
        timestamp_ms_var = self.m_timestamp_ms
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, timestamp_ms_var)
        Header.__logger.debug(f'fill timestamp_ms_var:{timestamp_ms_var}')
        pos += struct.calcsize('H')
        timestamp_min_var = self.m_timestamp_min
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, timestamp_min_var)
        Header.__logger.debug(f'fill timestamp_min_var:{timestamp_min_var}')
        pos += struct.calcsize('I')
        return pos
