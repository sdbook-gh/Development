import struct
import logging
import ctypes
from .Constant import Constant


class CloudToVehicleCtl:
    TYPE_VALUE = 0x36
    VERSION_VALUE = 0x02

    __logger = logging.getLogger('CloudToVehicleCtl')

    def __init__(self):
        self.total_size = None
        self.m_vehicleId = None
        self.m_ctlMode = None
        self.m_dataLen = None
        self.m_ctlData = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('8s')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            if not self.m_ctlData:
                raise Exception('m_ctlData is not set')
            self.total_size += self.m_ctlData.calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        vehicleId_var = struct.unpack_from('8s', buffer, pos)
        CloudToVehicleCtl.__logger.debug(f'parse get vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        ctlMode_var = struct.unpack_from('B', buffer, pos)
        CloudToVehicleCtl.__logger.debug(f'parse get ctlMode_var:{ctlMode_var}')
        pos += struct.calcsize('B')
        dataLen_var = struct.unpack_from('H', buffer, pos)
        CloudToVehicleCtl.__logger.debug(f'parse get dataLen_var:{dataLen_var}')
        pos += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            self.m_ctlData = ctlData_CloudToVehicleCtl(self.m_dataLen)
            if pos + self.m_ctlData.calc_real_size() > len(buffer):
                raise Exception('start_pos + size of m_ctlData > buffer size')
            pos = self.m_ctlData.parse_buffer(buffer, pos)
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
        CloudToVehicleCtl.__logger.debug(f'fill vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        ctlMode_var = self.m_ctlMode
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, ctlMode_var)
        CloudToVehicleCtl.__logger.debug(f'fill ctlMode_var:{ctlMode_var}')
        pos += struct.calcsize('B')
        dataLen_var = self.m_dataLen.value
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, dataLen_var)
        CloudToVehicleCtl.__logger.debug(f'fill dataLen_var:{dataLen_var}')
        pos += struct.calcsize('H')

        if self.m_dataLen.value > 0:
            if not self.m_ctlData:
                raise Exception('m_ctlData is not set')
            pos = self.m_ctlData.fill_buffer(buffer, pos)
        return pos


class ctlData_CloudToVehicleCtl:
    __logger = logging.getLogger('ctlData_CloudToVehicleCtl')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_expSpeed = []
        self.m_equationNum = []
        self.m_formula = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('B')
        self.total_size *= self.parentCount.value

        for i in range(len(self.m_equationNum)):
            if self.m_equationNum[i].value > 0:
                if not self.m_formula[i]:
                    raise Exception('m_formula{i} is not set')
                self.total_size += self.m_formula[i].calc_real_size()
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        for i in range(self.parentCount.value):
            expSpeed_var = struct.unpack_from('H', buffer, pos)
            self.m_expSpeed.append(expSpeed_var)
            ctlData_CloudToVehicleCtl.__logger.debug(f'parse get expSpeed_var:{expSpeed_var}')
            pos += struct.calcsize('H')
            equationNum_var = struct.unpack_from('B', buffer, pos)
            self.m_equationNum.append(equationNum_var)
            ctlData_CloudToVehicleCtl.__logger.debug(f'parse get equationNum_var:{equationNum_var}')
            pos += struct.calcsize('B')

            if self.m_equationNum[-1].value > 0:
                self.m_formula.append(formula_ctlData_CloudToVehicleCtl(self.m_equationNum[-1]))
                if pos + self.m_formula[-1].calc_real_size() > len(buffer):
                    raise Exception('start_pos + size of m_formula > buffer size')
                pos = self.m_formula[-1].parse_buffer(buffer, pos)
        return pos

    def fill_buffer(self, buffer: bytes, start_pos: int) -> int:
        if not self.total_size:
            raise Exception('calc_real_size is not invoked')
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos

        if len(self.m_expSpeed) != self.parentCount.value:
            raise Exception('parentCount mismatch m_expSpeed count')

        if len(self.m_equationNum) != self.parentCount.value:
            raise Exception('parentCount mismatch m_equationNum count')
        for i in range(self.parentCount.value):
            expSpeed_var = self.m_expSpeed[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, expSpeed_var)
            ctlData_CloudToVehicleCtl.__logger.debug(f'fill expSpeed_var:{expSpeed_var}')
            pos += struct.calcsize('H')
            equationNum_var = self.m_equationNum[i].value
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, equationNum_var)
            ctlData_CloudToVehicleCtl.__logger.debug(f'fill equationNum_var:{equationNum_var}')
            pos += struct.calcsize('B')

            if self.m_equationNum[i].value > 0:
                if not self.m_formula[i]:
                    raise Exception('m_formula is not set')
                pos = self.m_formula[i].fill_buffer(buffer, pos)
        return pos


class formula_ctlData_CloudToVehicleCtl:
    __logger = logging.getLogger('formula_ctlData_CloudToVehicleCtl')

    def __init__(self, parentCount: ctypes.c_int):
        if parentCount is None:
            raise Exception('parentCount is None')
        self.parentCount = parentCount
        self.total_size = None
        self.m_factor3 = []
        self.m_factor2 = []
        self.m_factor1 = []
        self.m_factorC = []
        self.m_min = []
        self.m_max = []

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('d')
        self.total_size += struct.calcsize('d')
        self.total_size += struct.calcsize('d')
        self.total_size += struct.calcsize('d')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
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
            factor3_var = struct.unpack_from('d', buffer, pos)
            self.m_factor3.append(factor3_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get factor3_var:{factor3_var}')
            pos += struct.calcsize('d')
            factor2_var = struct.unpack_from('d', buffer, pos)
            self.m_factor2.append(factor2_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get factor2_var:{factor2_var}')
            pos += struct.calcsize('d')
            factor1_var = struct.unpack_from('d', buffer, pos)
            self.m_factor1.append(factor1_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get factor1_var:{factor1_var}')
            pos += struct.calcsize('d')
            factorC_var = struct.unpack_from('d', buffer, pos)
            self.m_factorC.append(factorC_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get factorC_var:{factorC_var}')
            pos += struct.calcsize('d')
            min_var = struct.unpack_from('H', buffer, pos)
            self.m_min.append(min_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get min_var:{min_var}')
            pos += struct.calcsize('H')
            max_var = struct.unpack_from('H', buffer, pos)
            self.m_max.append(max_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'parse get max_var:{max_var}')
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

        if len(self.m_factor3) != self.parentCount.value:
            raise Exception('parentCount mismatch m_factor3 count')

        if len(self.m_factor2) != self.parentCount.value:
            raise Exception('parentCount mismatch m_factor2 count')

        if len(self.m_factor1) != self.parentCount.value:
            raise Exception('parentCount mismatch m_factor1 count')

        if len(self.m_factorC) != self.parentCount.value:
            raise Exception('parentCount mismatch m_factorC count')

        if len(self.m_min) != self.parentCount.value:
            raise Exception('parentCount mismatch m_min count')

        if len(self.m_max) != self.parentCount.value:
            raise Exception('parentCount mismatch m_max count')
        for i in range(self.parentCount.value):
            factor3_var = self.m_factor3[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'd', buffer, pos, factor3_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill factor3_var:{factor3_var}')
            pos += struct.calcsize('d')
            factor2_var = self.m_factor2[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'd', buffer, pos, factor2_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill factor2_var:{factor2_var}')
            pos += struct.calcsize('d')
            factor1_var = self.m_factor1[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'd', buffer, pos, factor1_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill factor1_var:{factor1_var}')
            pos += struct.calcsize('d')
            factorC_var = self.m_factorC[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'd', buffer, pos, factorC_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill factorC_var:{factorC_var}')
            pos += struct.calcsize('d')
            min_var = self.m_min[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, min_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill min_var:{min_var}')
            pos += struct.calcsize('H')
            max_var = self.m_max[i]
            struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, max_var)
            formula_ctlData_CloudToVehicleCtl.__logger.debug(f'fill max_var:{max_var}')
            pos += struct.calcsize('H')
        return pos
