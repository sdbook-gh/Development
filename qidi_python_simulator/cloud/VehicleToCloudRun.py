import struct
import logging
from .Constant import Constant


class VehicleToCloudRun:
    TYPE_VALUE = 0x15
    VERSION_VALUE = 0x02

    __logger = logging.getLogger('VehicleToCloudRun')

    def __init__(self):
        self.total_size = None
        self.m_vehicleId = None
        self.m_timestampGNSS = None
        self.m_velocityGNSS = None
        self.m_longitude = None
        self.m_latitude = None
        self.m_elevation = None
        self.m_heading = None
        self.m_hdop = None
        self.m_vdop = None
        self.m_tapPos = None
        self.m_steeringAngle = None
        self.m_lights = None
        self.m_velocityCAN = None
        self.m_acceleration_V = None
        self.m_acceleration_H = None
        self.m_accelPos = None
        self.m_engineSpeed = None
        self.m_engineTorque = None
        self.m_brakeFlag = None
        self.m_brakePos = None
        self.m_brakePressure = None
        self.m_yawRate = None
        self.m_wheelVelocity_FL = None
        self.m_wheelVelocity_RL = None
        self.m_wheelVelocity_RR = None
        self.m_absFlag = None
        self.m_tcsFlag = None
        self.m_espFlag = None
        self.m_lkaFlag = None
        self.m_accMode = None

    def calc_real_size(self) -> int:
        self.total_size = 0
        self.total_size += struct.calcsize('8s')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('I')
        self.total_size += struct.calcsize('I')
        self.total_size += struct.calcsize('i')
        self.total_size += struct.calcsize('I')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('i')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('i')
        self.total_size += struct.calcsize('i')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('H')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        self.total_size += struct.calcsize('B')
        return self.total_size

    def parse_buffer(self, buffer, start_pos: int) -> int:
        if len(buffer) - start_pos < 0:
            raise Exception('start_pos > buffer size')
        self.calc_real_size()
        if start_pos + self.total_size > len(buffer):
            raise Exception('start_pos + total_size > buffer size')
        pos = start_pos
        vehicleId_var = struct.unpack_from('8s', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get vehicleId_var:{vehicleId_var}')
        self.m_vehicleId = vehicleId_var
        pos += struct.calcsize('8s')
        timestampGNSS_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get timestampGNSS_var:{timestampGNSS_var}')
        self.m_timestampGNSS = timestampGNSS_var
        pos += struct.calcsize('H')
        velocityGNSS_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get velocityGNSS_var:{velocityGNSS_var}')
        self.m_velocityGNSS = velocityGNSS_var
        pos += struct.calcsize('H')
        longitude_var = struct.unpack_from('I', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get longitude_var:{longitude_var}')
        self.m_longitude = longitude_var
        pos += struct.calcsize('I')
        latitude_var = struct.unpack_from('I', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get latitude_var:{latitude_var}')
        self.m_latitude = latitude_var
        pos += struct.calcsize('I')
        elevation_var = struct.unpack_from('i', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get elevation_var:{elevation_var}')
        self.m_elevation = elevation_var
        pos += struct.calcsize('i')
        heading_var = struct.unpack_from('I', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get heading_var:{heading_var}')
        self.m_heading = heading_var
        pos += struct.calcsize('I')
        hdop_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get hdop_var:{hdop_var}')
        self.m_hdop = hdop_var
        pos += struct.calcsize('B')
        vdop_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get vdop_var:{vdop_var}')
        self.m_vdop = vdop_var
        pos += struct.calcsize('B')
        tapPos_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get tapPos_var:{tapPos_var}')
        self.m_tapPos = tapPos_var
        pos += struct.calcsize('B')
        steeringAngle_var = struct.unpack_from('i', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get steeringAngle_var:{steeringAngle_var}')
        self.m_steeringAngle = steeringAngle_var
        pos += struct.calcsize('i')
        lights_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get lights_var:{lights_var}')
        self.m_lights = lights_var
        pos += struct.calcsize('H')
        velocityCAN_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get velocityCAN_var:{velocityCAN_var}')
        self.m_velocityCAN = velocityCAN_var
        pos += struct.calcsize('H')
        acceleration_V_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get acceleration_V_var:{acceleration_V_var}')
        self.m_acceleration_V = acceleration_V_var
        pos += struct.calcsize('H')
        acceleration_H_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get acceleration_H_var:{acceleration_H_var}')
        self.m_acceleration_H = acceleration_H_var
        pos += struct.calcsize('H')
        accelPos_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get accelPos_var:{accelPos_var}')
        self.m_accelPos = accelPos_var
        pos += struct.calcsize('B')
        engineSpeed_var = struct.unpack_from('i', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get engineSpeed_var:{engineSpeed_var}')
        self.m_engineSpeed = engineSpeed_var
        pos += struct.calcsize('i')
        engineTorque_var = struct.unpack_from('i', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get engineTorque_var:{engineTorque_var}')
        self.m_engineTorque = engineTorque_var
        pos += struct.calcsize('i')
        brakeFlag_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get brakeFlag_var:{brakeFlag_var}')
        self.m_brakeFlag = brakeFlag_var
        pos += struct.calcsize('B')
        brakePos_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get brakePos_var:{brakePos_var}')
        self.m_brakePos = brakePos_var
        pos += struct.calcsize('B')
        brakePressure_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get brakePressure_var:{brakePressure_var}')
        self.m_brakePressure = brakePressure_var
        pos += struct.calcsize('H')
        yawRate_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get yawRate_var:{yawRate_var}')
        self.m_yawRate = yawRate_var
        pos += struct.calcsize('H')
        wheelVelocity_FL_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get wheelVelocity_FL_var:{wheelVelocity_FL_var}')
        self.m_wheelVelocity_FL = wheelVelocity_FL_var
        pos += struct.calcsize('H')
        wheelVelocity_RL_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get wheelVelocity_RL_var:{wheelVelocity_RL_var}')
        self.m_wheelVelocity_RL = wheelVelocity_RL_var
        pos += struct.calcsize('H')
        wheelVelocity_RR_var = struct.unpack_from('H', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get wheelVelocity_RR_var:{wheelVelocity_RR_var}')
        self.m_wheelVelocity_RR = wheelVelocity_RR_var
        pos += struct.calcsize('H')
        absFlag_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get absFlag_var:{absFlag_var}')
        self.m_absFlag = absFlag_var
        pos += struct.calcsize('B')
        tcsFlag_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get tcsFlag_var:{tcsFlag_var}')
        self.m_tcsFlag = tcsFlag_var
        pos += struct.calcsize('B')
        espFlag_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get espFlag_var:{espFlag_var}')
        self.m_espFlag = espFlag_var
        pos += struct.calcsize('B')
        lkaFlag_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get lkaFlag_var:{lkaFlag_var}')
        self.m_lkaFlag = lkaFlag_var
        pos += struct.calcsize('B')
        accMode_var = struct.unpack_from('B', buffer, pos)[0]
        VehicleToCloudRun.__logger.debug(f'parse get accMode_var:{accMode_var}')
        self.m_accMode = accMode_var
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
        vehicleId_var = self.m_vehicleId
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + '8s', buffer, pos, vehicleId_var)
        VehicleToCloudRun.__logger.debug(f'fill vehicleId_var:{vehicleId_var}')
        pos += struct.calcsize('8s')
        timestampGNSS_var = self.m_timestampGNSS
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, timestampGNSS_var)
        VehicleToCloudRun.__logger.debug(f'fill timestampGNSS_var:{timestampGNSS_var}')
        pos += struct.calcsize('H')
        velocityGNSS_var = self.m_velocityGNSS
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, velocityGNSS_var)
        VehicleToCloudRun.__logger.debug(f'fill velocityGNSS_var:{velocityGNSS_var}')
        pos += struct.calcsize('H')
        longitude_var = self.m_longitude
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, longitude_var)
        VehicleToCloudRun.__logger.debug(f'fill longitude_var:{longitude_var}')
        pos += struct.calcsize('I')
        latitude_var = self.m_latitude
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, latitude_var)
        VehicleToCloudRun.__logger.debug(f'fill latitude_var:{latitude_var}')
        pos += struct.calcsize('I')
        elevation_var = self.m_elevation
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'i', buffer, pos, elevation_var)
        VehicleToCloudRun.__logger.debug(f'fill elevation_var:{elevation_var}')
        pos += struct.calcsize('i')
        heading_var = self.m_heading
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'I', buffer, pos, heading_var)
        VehicleToCloudRun.__logger.debug(f'fill heading_var:{heading_var}')
        pos += struct.calcsize('I')
        hdop_var = self.m_hdop
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, hdop_var)
        VehicleToCloudRun.__logger.debug(f'fill hdop_var:{hdop_var}')
        pos += struct.calcsize('B')
        vdop_var = self.m_vdop
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, vdop_var)
        VehicleToCloudRun.__logger.debug(f'fill vdop_var:{vdop_var}')
        pos += struct.calcsize('B')
        tapPos_var = self.m_tapPos
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, tapPos_var)
        VehicleToCloudRun.__logger.debug(f'fill tapPos_var:{tapPos_var}')
        pos += struct.calcsize('B')
        steeringAngle_var = self.m_steeringAngle
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'i', buffer, pos, steeringAngle_var)
        VehicleToCloudRun.__logger.debug(f'fill steeringAngle_var:{steeringAngle_var}')
        pos += struct.calcsize('i')
        lights_var = self.m_lights
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, lights_var)
        VehicleToCloudRun.__logger.debug(f'fill lights_var:{lights_var}')
        pos += struct.calcsize('H')
        velocityCAN_var = self.m_velocityCAN
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, velocityCAN_var)
        VehicleToCloudRun.__logger.debug(f'fill velocityCAN_var:{velocityCAN_var}')
        pos += struct.calcsize('H')
        acceleration_V_var = self.m_acceleration_V
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, acceleration_V_var)
        VehicleToCloudRun.__logger.debug(f'fill acceleration_V_var:{acceleration_V_var}')
        pos += struct.calcsize('H')
        acceleration_H_var = self.m_acceleration_H
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, acceleration_H_var)
        VehicleToCloudRun.__logger.debug(f'fill acceleration_H_var:{acceleration_H_var}')
        pos += struct.calcsize('H')
        accelPos_var = self.m_accelPos
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, accelPos_var)
        VehicleToCloudRun.__logger.debug(f'fill accelPos_var:{accelPos_var}')
        pos += struct.calcsize('B')
        engineSpeed_var = self.m_engineSpeed
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'i', buffer, pos, engineSpeed_var)
        VehicleToCloudRun.__logger.debug(f'fill engineSpeed_var:{engineSpeed_var}')
        pos += struct.calcsize('i')
        engineTorque_var = self.m_engineTorque
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'i', buffer, pos, engineTorque_var)
        VehicleToCloudRun.__logger.debug(f'fill engineTorque_var:{engineTorque_var}')
        pos += struct.calcsize('i')
        brakeFlag_var = self.m_brakeFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, brakeFlag_var)
        VehicleToCloudRun.__logger.debug(f'fill brakeFlag_var:{brakeFlag_var}')
        pos += struct.calcsize('B')
        brakePos_var = self.m_brakePos
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, brakePos_var)
        VehicleToCloudRun.__logger.debug(f'fill brakePos_var:{brakePos_var}')
        pos += struct.calcsize('B')
        brakePressure_var = self.m_brakePressure
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, brakePressure_var)
        VehicleToCloudRun.__logger.debug(f'fill brakePressure_var:{brakePressure_var}')
        pos += struct.calcsize('H')
        yawRate_var = self.m_yawRate
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, yawRate_var)
        VehicleToCloudRun.__logger.debug(f'fill yawRate_var:{yawRate_var}')
        pos += struct.calcsize('H')
        wheelVelocity_FL_var = self.m_wheelVelocity_FL
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, wheelVelocity_FL_var)
        VehicleToCloudRun.__logger.debug(f'fill wheelVelocity_FL_var:{wheelVelocity_FL_var}')
        pos += struct.calcsize('H')
        wheelVelocity_RL_var = self.m_wheelVelocity_RL
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, wheelVelocity_RL_var)
        VehicleToCloudRun.__logger.debug(f'fill wheelVelocity_RL_var:{wheelVelocity_RL_var}')
        pos += struct.calcsize('H')
        wheelVelocity_RR_var = self.m_wheelVelocity_RR
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'H', buffer, pos, wheelVelocity_RR_var)
        VehicleToCloudRun.__logger.debug(f'fill wheelVelocity_RR_var:{wheelVelocity_RR_var}')
        pos += struct.calcsize('H')
        absFlag_var = self.m_absFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, absFlag_var)
        VehicleToCloudRun.__logger.debug(f'fill absFlag_var:{absFlag_var}')
        pos += struct.calcsize('B')
        tcsFlag_var = self.m_tcsFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, tcsFlag_var)
        VehicleToCloudRun.__logger.debug(f'fill tcsFlag_var:{tcsFlag_var}')
        pos += struct.calcsize('B')
        espFlag_var = self.m_espFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, espFlag_var)
        VehicleToCloudRun.__logger.debug(f'fill espFlag_var:{espFlag_var}')
        pos += struct.calcsize('B')
        lkaFlag_var = self.m_lkaFlag
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, lkaFlag_var)
        VehicleToCloudRun.__logger.debug(f'fill lkaFlag_var:{lkaFlag_var}')
        pos += struct.calcsize('B')
        accMode_var = self.m_accMode
        struct.pack_into(Constant.CLOUD_PROTOCOL_ENDIAN_SIGN + 'B', buffer, pos, accMode_var)
        VehicleToCloudRun.__logger.debug(f'fill accMode_var:{accMode_var}')
        pos += struct.calcsize('B')
        return pos
