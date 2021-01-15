yamlTextList = []

# yamltext='''
# VehicleToCloudCmdFcwRes:
#     cmdType: '{"_struct": "B"}'
#     seq: '{"_struct": "Q"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext='''
# CloudToVehicleCmdFcw:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     seq: '{"_struct": "Q"}'
#     alertType: '{"_struct": "H"}'
#     alertRadius: '{"_struct": "H"}'
#     pointNum: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     path:
#         longitude: '{"_struct": "I"}'
#         latitude: '{"_struct": "I"}'
#         elevation: '{"_struct": "i"}'
#     objType: '{"_struct": "B"}'
#     fcwLevel: '{"_struct": "B"}'
# '''

# yamltext='''
# VehicleToCloudCmdRampIntenetChangeRes:
#     cmdType: '{"_struct": "B"}'
#     seq: '{"_struct": "Q"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext='''
# CloudToVehicleCmdRampIntenetChange:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     seq: '{"_struct": "Q"}'
#     alertType: '{"_struct": "H"}'
#     alertRadius: '{"_struct": "H"}'
#     pointNum: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     path:
#         longitude: '{"_struct": "I"}'
#         latitude: '{"_struct": "I"}'
#         elevation: '{"_struct": "i"}'
#         dtc: '{"_struct": "H"}'
#         ttc: '{"_struct": "H"}'
# '''

# yamltext = '''
# CloudToVehicleCmdLaneSpdLmtRes:
#     cmdType: '{"_struct": "B"}'
#     seq: '{"_struct": "Q"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext='''
# CloudToVehicleCmdLaneSpdLmt:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     seq: '{"_struct": "Q"}'
#     alertType: '{"_struct": "H"}'
#     alertRadius: '{"_struct": "H"}'
#     pointNum: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     path:
#         longitude: '{"_struct": "I"}'
#         latitude: '{"_struct": "I"}'
#         elevation: '{"_struct": "i"}'
#     laneNum: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     speed:
#         laneId: '{"_struct": "B"}'
#         speedLimit: '{"_struct": "B"}'
# '''

# yamltext = '''
# CloudToVehicleCmdNtlarRes:
#     cmdType: '{"_struct": "B"}'
#     seq: '{"_struct": "Q"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext = '''
# CloudToVehicleCmdNtlar:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     seq: '{"_struct": "Q"}'
#     cmdFlag: '{"_struct": "B"}'
#     level: '{"_struct": "B"}'
# '''

# yamltext = '''
# CloudToVehicleCmdGlosaRes:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext = '''
# CloudToVehicleCmdGlosa:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     seq: '{"_struct": "Q"}'
#     cmdFlag: '{"_struct": "B"}'
#     spdMax: '{"_struct": "H"}'
#     spdMin: '{"_struct": "H"}'
#     spdExp: '{"_struct": "H"}'
# '''

# yamltext = '''
# VehicleToCloudCmdHistvideoRes:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext = '''
# VehicleToCloudCmdHistvideo:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     videoType: '{"_struct": "B"}'
#     camIdLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     camId:
#         camId: '{"_struct": "s"}'
#     startTime: '{"_struct": "I"}'
#     endTime: '{"_struct": "I"}'
#     urlAddrLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     urlAddr:
#         urlAddr: '{"_struct": "s"}'
# '''

# yamltext = '''
# VehicleToCloudCmdStreamVideoRes:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     doFlag: '{"_struct": "B"}'
# '''

# yamltext = '''
# VehicleToCloudCmdStreamVideo:
#     cmdType: '{"_struct": "B"}'
#     uuid: '{"_struct": "36s"}'
#     camIdLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     camId:
#         camId: '{"_struct": "s"}'
#     videoQual: '{"_struct": "B"}'
#     urlAddrLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#     urlAddr:
#         urlAddr: '{"_struct": "s"}'
# '''

# yamltext = '''
# CloudToVehicleCmd:
#     vehicleId: '{"_struct": "8s"}'
#     dataLen: '{"_struct": "H", "_repeat": "ctypes.c_int"}'
#     cmdData:
#         cmdData: '{"_struct": "B"}'
# '''

# yamltext = '''
# CloudToVehicleCtl:
#     vehicleId: '{"_struct": "8s"}'
#     ctlMode: '{"_struct": "B"}'
#     dataLen: '{"_struct": "H", "_repeat": "ctypes.c_int"}'
#     ctlData:
#         expSpeed: '{"_struct": "H"}'
#         equationNum: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
#         formula:
#             factor3: '{"_struct": "d"}'
#             factor2: '{"_struct": "d"}'
#             factor1: '{"_struct": "d"}'
#             factorC: '{"_struct": "d"}'
#             min: '{"_struct": "H"}'
#             max: '{"_struct": "H"}'
# '''

yamlTextList.append('''
CloudToVehicleReqRes:
    resLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
    funcRes:
        funcRes: '{"_struct": "B"}'
''')

yamlTextList.append('''
VehicleToCloudReq:
    vehicleId: '{"_struct": "8s"}'
    ctlMode: '{"_struct": "B"}'
    reqLen: '{"_struct": "B", "_repeat": "ctypes.c_int"}'
    funcReq:
        funcReq: '{"_struct": "B"}'
''')

yamlTextList.append('''
VehicleToCloudRun:
    vehicleId: '{"_struct": "8s"}'
    timestampGNSS: '{"_struct": "H"}'
    velocityGNSS: '{"_struct": "H"}'
    longitude: '{"_struct": "I"}'
    latitude: '{"_struct": "I"}'
    elevation: '{"_struct": "i"}'
    heading: '{"_struct": "I"}'
    hdop: '{"_struct": "B"}'
    vdop: '{"_struct": "B"}'
    tapPos: '{"_struct": "B"}'
    steeringAngle: '{"_struct": "i"}'
    lights: '{"_struct": "H"}'
    velocityCAN: '{"_struct": "H"}'
    acceleration_V: '{"_struct": "H"}'
    acceleration_H: '{"_struct": "H"}'
    accelPos: '{"_struct": "B"}'
    engineSpeed: '{"_struct": "i"}'
    engineTorque: '{"_struct": "i"}'
    brakeFlag: '{"_struct": "B"}'
    brakePos: '{"_struct": "B"}'
    brakePressure: '{"_struct": "H"}'
    yawRate: '{"_struct": "H"}'
    wheelVelocity_FL: '{"_struct": "H"}'
    wheelVelocity_RL: '{"_struct": "H"}'
    wheelVelocity_RR: '{"_struct": "H"}'
    absFlag: '{"_struct": "B"}'
    tcsFlag: '{"_struct": "B"}'
    espFlag: '{"_struct": "B"}'
    lkaFlag: '{"_struct": "B"}'
    accMode: '{"_struct": "B"}'
''')

yamlTextList.append('''
Header:
    start_tag: '{"_struct": "B"}'
    remain_length: '{"_struct": "3B"}'
    type: '{"_struct": "B"}'
    version: '{"_struct": "B"}'
    timestamp_ms: '{"_struct": "H"}'
    timestamp_min: '{"_struct": "I"}'
''')
