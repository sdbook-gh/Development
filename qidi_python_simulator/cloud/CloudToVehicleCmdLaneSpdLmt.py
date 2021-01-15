import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleCmdLaneSpdLmt:
    __logger = logging.getLogger('CloudToVehicleCmdLaneSpdLmt')

    def __init__(self):
        self.total_size = None
        self.m_cmdType = None
        self.m_uuid = None
        self.m_seq = None
        self.m_alertType = None
        self.m_alertRadius = None
        self.m_pointNum = None
        self.m_path = None
        self.m_laneNum = None
        self.m_speed = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('36s')
        self.total_size += struct.calcsize('Q')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('B')

        if self.m_pointNum.value > 0:
            if not self.m_path:
                raise Exception('m_path is not set')
            self.total_size += self.m_path.calc_real_size()
        self.total_size += struct.calcsize('B')

        if self.m_laneNum.value > 0:
            if not self.m_speed:
                raise Exception('m_speed is not set')
            self.total_size += self.m_speed.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        cmdType_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = struct.unpack_from('36s', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        seq_var = struct.unpack_from('Q', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        alertType_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get alertType_var:{alertType_var}')
        pos += struct.calcsize('H')
        alertRadius_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get alertRadius_var:{alertRadius_var}')
        pos += struct.calcsize('H')
        pointNum_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get pointNum_var:{pointNum_var}')
        pos += struct.calcsize('B')

        if self.m_pointNum.value > 0:
            self.m_path = path_CloudToVehicleCmdLaneSpdLmt(self.m_pointNum)
            if pos + self.m_path.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_path > buffer size')
            pos = self.m_path.parse_buffer(buffer, pos)
        laneNum_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get laneNum_var:{laneNum_var}')
        pos += struct.calcsize('B')

        if self.m_laneNum.value > 0:
            self.m_speed = speed_CloudToVehicleCmdLaneSpdLmt(self.m_laneNum)
            if pos + self.m_speed.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_speed > buffer size')
            pos = self.m_speed.parse_buffer(buffer, pos)
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
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill cmdType_var:{cmdType_var}')
        pos += struct.calcsize('B')
        uuid_var = self.m_uuid
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '36s', buffer, pos, uuid_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill uuid_var:{uuid_var}')
        pos += struct.calcsize('36s')
        seq_var = self.m_seq
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'Q', buffer, pos, seq_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill seq_var:{seq_var}')
        pos += struct.calcsize('Q')
        alertType_var = self.m_alertType
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, alertType_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill alertType_var:{alertType_var}')
        pos += struct.calcsize('H')
        alertRadius_var = self.m_alertRadius
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, alertRadius_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill alertRadius_var:{alertRadius_var}')
        pos += struct.calcsize('H')
        pointNum_var = self.m_pointNum.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, pointNum_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill pointNum_var:{pointNum_var}')
        pos += struct.calcsize('B')

        if self.m_pointNum.value > 0:
            if not self.m_path:
                raise Exception('m_path is not set')
            pos = self.m_path.fill_buffer(buffer, pos)
        laneNum_var = self.m_laneNum.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, laneNum_var)
        CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill laneNum_var:{laneNum_var}')
        pos += struct.calcsize('B')

        if self.m_laneNum.value > 0:
            if not self.m_speed:
                raise Exception('m_speed is not set')
            pos = self.m_speed.fill_buffer(buffer, pos)
        return pos

class path_CloudToVehicleCmdLaneSpdLmt:
    __logger = logging.getLogger('path_CloudToVehicleCmdLaneSpdLmt')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_longitude = []
        self.m_latitude = []
        self.m_elevation = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('I')
        self.total_size += struct.calcsize('I')
        self.total_size += struct.calcsize('i')
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
            longitude_var = struct.unpack_from('I', buffer, pos)
            self.m_longitude.append(longitude_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get longitude_var:{longitude_var}')
            pos += struct.calcsize('I')
            latitude_var = struct.unpack_from('I', buffer, pos)
            self.m_latitude.append(latitude_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get latitude_var:{latitude_var}')
            pos += struct.calcsize('I')
            elevation_var = struct.unpack_from('i', buffer, pos)
            self.m_elevation.append(elevation_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get elevation_var:{elevation_var}')
            pos += struct.calcsize('i')
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos

        if len(self.m_longitude) != self.parentCount.value:
            raise Exception('parentCount mismatch m_longitude count')

        if len(self.m_latitude) != self.parentCount.value:
            raise Exception('parentCount mismatch m_latitude count')

        if len(self.m_elevation) != self.parentCount.value:
            raise Exception('parentCount mismatch m_elevation count')
        for i in range(self.parentCount.value):
            longitude_var = self.m_longitude[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, longitude_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill longitude_var:{longitude_var}')
            pos += struct.calcsize('I')
            latitude_var = self.m_latitude[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, latitude_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill latitude_var:{latitude_var}')
            pos += struct.calcsize('I')
            elevation_var = self.m_elevation[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'i', buffer, pos, elevation_var)
            path_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill elevation_var:{elevation_var}')
            pos += struct.calcsize('i')
        return pos

class speed_CloudToVehicleCmdLaneSpdLmt:
    __logger = logging.getLogger('speed_CloudToVehicleCmdLaneSpdLmt')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_laneId = []
        self.m_speedLimit = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('B')
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
            laneId_var = struct.unpack_from('B', buffer, pos)
            self.m_laneId.append(laneId_var)
            speed_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get laneId_var:{laneId_var}')
            pos += struct.calcsize('B')
            speedLimit_var = struct.unpack_from('B', buffer, pos)
            self.m_speedLimit.append(speedLimit_var)
            speed_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'parse get speedLimit_var:{speedLimit_var}')
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

        if len(self.m_laneId) != self.parentCount.value:
            raise Exception('parentCount mismatch m_laneId count')

        if len(self.m_speedLimit) != self.parentCount.value:
            raise Exception('parentCount mismatch m_speedLimit count')
        for i in range(self.parentCount.value):
            laneId_var = self.m_laneId[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, laneId_var)
            speed_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill laneId_var:{laneId_var}')
            pos += struct.calcsize('B')
            speedLimit_var = self.m_speedLimit[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, speedLimit_var)
            speed_CloudToVehicleCmdLaneSpdLmt.__logger.debug(f'fill speedLimit_var:{speedLimit_var}')
            pos += struct.calcsize('B')
        return pos
