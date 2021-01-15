import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleCmdGlosa:
    __logger = logging.getLogger('CloudToVehicleCmdGlosa')

    def __init__(self):
        self.total_size = None
        self.m_cmdType = None
        self.m_uuid = None
        self.m_seq = None
        self.m_cmdFlag = None
        self.m_spdMax = None
        self.m_spdMin = None
        self.m_spdExp = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('36s')
        self.total_size += struct.calcsize('Q')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        cmdType_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = struct.unpack_from('36s', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        seq_var = struct.unpack_from('Q', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        cmdFlag_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get cmdFlag_var:{cmdFlag_var}')
        pos += struct.calcsize('B')
        spdMax_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get spdMax_var:{spdMax_var}')
        pos += struct.calcsize('H')
        spdMin_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get spdMin_var:{spdMin_var}')
        pos += struct.calcsize('H')
        spdExp_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmdGlosa.__logger.debug(f'parse get spdExp_var:{spdExp_var}')
        pos += struct.calcsize('H')
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
        CloudToVehicleCmdGlosa.__logger.debug(f'fill cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = self.m_uuid
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '36s', buffer, pos, uuid_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        seq_var = self.m_seq
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'Q', buffer, pos, seq_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        cmdFlag_var = self.m_cmdFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, cmdFlag_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill cmdFlag_var:{cmdFlag_var}')
        pos += struct.calcsize('B')
        spdMax_var = self.m_spdMax
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, spdMax_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill spdMax_var:{spdMax_var}')
        pos += struct.calcsize('H')
        spdMin_var = self.m_spdMin
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, spdMin_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill spdMin_var:{spdMin_var}')
        pos += struct.calcsize('H')
        spdExp_var = self.m_spdExp
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, spdExp_var)
        CloudToVehicleCmdGlosa.__logger.debug(f'fill spdExp_var:{spdExp_var}')
        pos += struct.calcsize('H')
        return pos
