import cloud
import logging
import ctypes

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    header_var = cloud.Header()
    header_var.m_start_tag = 0x01
    header_var.m_remain_length = (0x01, 0x01, 0x01,)
    header_var.m_type = 0x01
    header_var.m_version = 0x01
    header_var.m_timestamp_ms = 0x0102
    header_var.m_timestamp_min = 0x01020304
    total_size = header_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = header_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudrun_var = cloud.VehicleToCloudRun()
    vehicletocloudrun_var.m_vehicleId = b'________'
    vehicletocloudrun_var.m_timestampGNSS = 0x0102
    vehicletocloudrun_var.m_velocityGNSS = 0x0102
    vehicletocloudrun_var.m_longitude = 0x01020304
    vehicletocloudrun_var.m_latitude = 0x01020304
    vehicletocloudrun_var.m_elevation = 0x01020304
    vehicletocloudrun_var.m_heading = 0x01020304
    vehicletocloudrun_var.m_hdop = 0x01
    vehicletocloudrun_var.m_vdop = 0x01
    vehicletocloudrun_var.m_tapPos = 0x01
    vehicletocloudrun_var.m_steeringAngle = 0x01020304
    vehicletocloudrun_var.m_lights = 0x0102
    vehicletocloudrun_var.m_velocityCAN = 0x0102
    vehicletocloudrun_var.m_acceleration_V = 0x0102
    vehicletocloudrun_var.m_acceleration_H = 0x0102
    vehicletocloudrun_var.m_accelPos = 0x01
    vehicletocloudrun_var.m_engineSpeed = 0x01020304
    vehicletocloudrun_var.m_engineTorque = 0x01020304
    vehicletocloudrun_var.m_brakeFlag = 0x01
    vehicletocloudrun_var.m_brakePos = 0x01
    vehicletocloudrun_var.m_brakePressure = 0x0102
    vehicletocloudrun_var.m_yawRate = 0x0102
    vehicletocloudrun_var.m_wheelVelocity_FL = 0x0102
    vehicletocloudrun_var.m_wheelVelocity_RL = 0x0102
    vehicletocloudrun_var.m_wheelVelocity_RR = 0x0102
    vehicletocloudrun_var.m_absFlag = 0x01
    vehicletocloudrun_var.m_tcsFlag = 0x01
    vehicletocloudrun_var.m_espFlag = 0x01
    vehicletocloudrun_var.m_lkaFlag = 0x01
    vehicletocloudrun_var.m_accMode = 0x01
    buffer = bytearray(vehicletocloudrun_var.calc_real_size())
    vehicletocloudrun_var.fill_buffer(buffer, 0)
    print(buffer)
    print(len(buffer))

    vehicletocloudreq_var = cloud.VehicleToCloudReq()
    vehicletocloudreq_var.m_vehicleId = b'________'
    vehicletocloudreq_var.m_ctlMode = 0x01
    vehicletocloudreq_var.m_reqLen = ctypes.c_int(0)
    funcreq_vehicletocloudreq_var = cloud.funcReq_VehicleToCloudReq(vehicletocloudreq_var.m_reqLen)
    vehicletocloudreq_var.m_funcReq = funcreq_vehicletocloudreq_var
    funcreq_vehicletocloudreq_var.m_funcReq.append(0x01)
    vehicletocloudreq_var.m_reqLen.value = len(funcreq_vehicletocloudreq_var.m_funcReq)
    total_size = vehicletocloudreq_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudreq_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclereqres_var = cloud.CloudToVehicleReqRes()
    cloudtovehiclereqres_var.m_resLen = ctypes.c_int(0)
    funcres_cloudtovehiclereqres_var = cloud.funcRes_CloudToVehicleReqRes(cloudtovehiclereqres_var.m_resLen)
    cloudtovehiclereqres_var.m_funcRes = funcres_cloudtovehiclereqres_var
    funcres_cloudtovehiclereqres_var.m_funcRes.append(0x01)
    cloudtovehiclereqres_var.m_resLen.value = len(funcres_cloudtovehiclereqres_var.m_funcRes)
    total_size = cloudtovehiclereqres_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclereqres_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclectl_var = cloud.CloudToVehicleCtl()
    cloudtovehiclectl_var.m_vehicleId = b'________'
    cloudtovehiclectl_var.m_ctlMode = 0x01
    cloudtovehiclectl_var.m_dataLen = ctypes.c_int(0)
    ctldata_cloudtovehiclectl_var = cloud.ctlData_CloudToVehicleCtl(cloudtovehiclectl_var.m_dataLen)
    cloudtovehiclectl_var.m_ctlData = ctldata_cloudtovehiclectl_var
    ctldata_cloudtovehiclectl_var.m_expSpeed.append(0x0102)
    ctldata_cloudtovehiclectl_var.m_expSpeed.append(0x0102)
    ctldata_cloudtovehiclectl_var.m_equationNum.append(ctypes.c_int(0))
    formula_ctldata_cloudtovehiclectl_var = cloud.formula_ctlData_CloudToVehicleCtl(
        ctldata_cloudtovehiclectl_var.m_equationNum[0])
    ctldata_cloudtovehiclectl_var.m_formula.append(formula_ctldata_cloudtovehiclectl_var)
    formula_ctldata_cloudtovehiclectl_var.m_factor3.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor2.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor1.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factorC.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_min.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_max.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_factor3.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor2.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor1.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factorC.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_min.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_max.append(0x0102)
    ctldata_cloudtovehiclectl_var.m_equationNum[0].value = len(formula_ctldata_cloudtovehiclectl_var.m_factor3)
    ctldata_cloudtovehiclectl_var.m_equationNum.append(ctypes.c_int(0))
    formula_ctldata_cloudtovehiclectl_var = cloud.formula_ctlData_CloudToVehicleCtl(
        ctldata_cloudtovehiclectl_var.m_equationNum[1])
    ctldata_cloudtovehiclectl_var.m_formula.append(formula_ctldata_cloudtovehiclectl_var)
    formula_ctldata_cloudtovehiclectl_var.m_factor3.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor2.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor1.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factorC.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_min.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_max.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_factor3.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor2.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor1.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factorC.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_min.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_max.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_factor3.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor2.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factor1.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_factorC.append(0x0102030405060708)
    formula_ctldata_cloudtovehiclectl_var.m_min.append(0x0102)
    formula_ctldata_cloudtovehiclectl_var.m_max.append(0x0102)
    ctldata_cloudtovehiclectl_var.m_equationNum[1].value = len(formula_ctldata_cloudtovehiclectl_var.m_factor3)
    cloudtovehiclectl_var.m_dataLen.value = len(ctldata_cloudtovehiclectl_var.m_expSpeed)
    total_size = cloudtovehiclectl_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclectl_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmd_var = cloud.CloudToVehicleCmd()
    cloudtovehiclecmd_var.m_vehicleId = b'________'
    cloudtovehiclecmd_var.m_dataLen = ctypes.c_int(0)
    cmddata_cloudtovehiclecmd_var = cloud.cmdData_CloudToVehicleCmd(cloudtovehiclecmd_var.m_dataLen)
    cloudtovehiclecmd_var.m_cmdData = cmddata_cloudtovehiclecmd_var
    cmddata_cloudtovehiclecmd_var.m_cmdData.append(0x01)
    cloudtovehiclecmd_var.m_dataLen.value = len(cmddata_cloudtovehiclecmd_var.m_cmdData)
    total_size = cloudtovehiclecmd_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmd_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdstreamvideo_var = cloud.VehicleToCloudCmdStreamVideo()
    vehicletocloudcmdstreamvideo_var.m_cmdType = 0x01
    vehicletocloudcmdstreamvideo_var.m_uuid = b'____________________________________'
    vehicletocloudcmdstreamvideo_var.m_camIdLen = ctypes.c_int(0)
    camid_vehicletocloudcmdstreamvideo_var = cloud.camId_VehicleToCloudCmdStreamVideo(
        vehicletocloudcmdstreamvideo_var.m_camIdLen)
    vehicletocloudcmdstreamvideo_var.m_camId = camid_vehicletocloudcmdstreamvideo_var
    camid_vehicletocloudcmdstreamvideo_var.m_camId.append(b'_')
    vehicletocloudcmdstreamvideo_var.m_camIdLen.value = len(camid_vehicletocloudcmdstreamvideo_var.m_camId)
    vehicletocloudcmdstreamvideo_var.m_videoQual = 0x01
    vehicletocloudcmdstreamvideo_var.m_urlAddrLen = ctypes.c_int(0)
    urladdr_vehicletocloudcmdstreamvideo_var = cloud.urlAddr_VehicleToCloudCmdStreamVideo(
        vehicletocloudcmdstreamvideo_var.m_urlAddrLen)
    vehicletocloudcmdstreamvideo_var.m_urlAddr = urladdr_vehicletocloudcmdstreamvideo_var
    urladdr_vehicletocloudcmdstreamvideo_var.m_urlAddr.append(b'_')
    vehicletocloudcmdstreamvideo_var.m_urlAddrLen.value = len(urladdr_vehicletocloudcmdstreamvideo_var.m_urlAddr)
    total_size = vehicletocloudcmdstreamvideo_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdstreamvideo_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdstreamvideores_var = cloud.VehicleToCloudCmdStreamVideoRes()
    vehicletocloudcmdstreamvideores_var.m_cmdType = 0x01
    vehicletocloudcmdstreamvideores_var.m_uuid = b'____________________________________'
    vehicletocloudcmdstreamvideores_var.m_doFlag = 0x01
    total_size = vehicletocloudcmdstreamvideores_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdstreamvideores_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdhistvideo_var = cloud.VehicleToCloudCmdHistvideo()
    vehicletocloudcmdhistvideo_var.m_cmdType = 0x01
    vehicletocloudcmdhistvideo_var.m_uuid = b'____________________________________'
    vehicletocloudcmdhistvideo_var.m_videoType = 0x01
    vehicletocloudcmdhistvideo_var.m_camIdLen = ctypes.c_int(0)
    camid_vehicletocloudcmdhistvideo_var = cloud.camId_VehicleToCloudCmdHistvideo(
        vehicletocloudcmdhistvideo_var.m_camIdLen)
    vehicletocloudcmdhistvideo_var.m_camId = camid_vehicletocloudcmdhistvideo_var
    camid_vehicletocloudcmdhistvideo_var.m_camId.append(b'_')
    vehicletocloudcmdhistvideo_var.m_camIdLen.value = len(camid_vehicletocloudcmdhistvideo_var.m_camId)
    vehicletocloudcmdhistvideo_var.m_startTime = 0x01020304
    vehicletocloudcmdhistvideo_var.m_endTime = 0x01020304
    vehicletocloudcmdhistvideo_var.m_urlAddrLen = ctypes.c_int(0)
    urladdr_vehicletocloudcmdhistvideo_var = cloud.urlAddr_VehicleToCloudCmdHistvideo(
        vehicletocloudcmdhistvideo_var.m_urlAddrLen)
    vehicletocloudcmdhistvideo_var.m_urlAddr = urladdr_vehicletocloudcmdhistvideo_var
    urladdr_vehicletocloudcmdhistvideo_var.m_urlAddr.append(b'_')
    vehicletocloudcmdhistvideo_var.m_urlAddrLen.value = len(urladdr_vehicletocloudcmdhistvideo_var.m_urlAddr)
    total_size = vehicletocloudcmdhistvideo_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdhistvideo_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdhistvideores_var = cloud.VehicleToCloudCmdHistvideoRes()
    vehicletocloudcmdhistvideores_var.m_cmdType = 0x01
    vehicletocloudcmdhistvideores_var.m_uuid = b'____________________________________'
    vehicletocloudcmdhistvideores_var.m_doFlag = 0x01
    total_size = vehicletocloudcmdhistvideores_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdhistvideores_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdglosa_var = cloud.CloudToVehicleCmdGlosa()
    cloudtovehiclecmdglosa_var.m_cmdType = 0x01
    cloudtovehiclecmdglosa_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdglosa_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdglosa_var.m_cmdFlag = 0x01
    cloudtovehiclecmdglosa_var.m_spdMax = 0x0102
    cloudtovehiclecmdglosa_var.m_spdMin = 0x0102
    cloudtovehiclecmdglosa_var.m_spdExp = 0x0102
    total_size = cloudtovehiclecmdglosa_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdglosa_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdglosares_var = cloud.CloudToVehicleCmdGlosaRes()
    cloudtovehiclecmdglosares_var.m_cmdType = 0x01
    cloudtovehiclecmdglosares_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdglosares_var.m_doFlag = 0x01
    total_size = cloudtovehiclecmdglosares_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdglosares_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdntlar_var = cloud.CloudToVehicleCmdNtlar()
    cloudtovehiclecmdntlar_var.m_cmdType = 0x01
    cloudtovehiclecmdntlar_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdntlar_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdntlar_var.m_cmdFlag = 0x01
    cloudtovehiclecmdntlar_var.m_level = 0x01
    total_size = cloudtovehiclecmdntlar_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdntlar_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdntlarres_var = cloud.CloudToVehicleCmdNtlarRes()
    cloudtovehiclecmdntlarres_var.m_cmdType = 0x01
    cloudtovehiclecmdntlarres_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdntlarres_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdntlarres_var.m_cmdFlag = 0x01
    cloudtovehiclecmdntlarres_var.m_level = 0x01
    cloudtovehiclecmdntlarres_var.m_doFlag = 0x01
    total_size = cloudtovehiclecmdntlarres_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdntlarres_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdlanespdlmt_var = cloud.CloudToVehicleCmdLaneSpdLmt()
    cloudtovehiclecmdlanespdlmt_var.m_cmdType = 0x01
    cloudtovehiclecmdlanespdlmt_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdlanespdlmt_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdlanespdlmt_var.m_alertType = 0x0102
    cloudtovehiclecmdlanespdlmt_var.m_alertRadius = 0x0102
    cloudtovehiclecmdlanespdlmt_var.m_pointNum = ctypes.c_int(0)
    path_cloudtovehiclecmdlanespdlmt_var = cloud.path_CloudToVehicleCmdLaneSpdLmt(
        cloudtovehiclecmdlanespdlmt_var.m_pointNum)
    cloudtovehiclecmdlanespdlmt_var.m_path = path_cloudtovehiclecmdlanespdlmt_var
    path_cloudtovehiclecmdlanespdlmt_var.m_longitude.append(0x01020304)
    path_cloudtovehiclecmdlanespdlmt_var.m_latitude.append(0x01020304)
    path_cloudtovehiclecmdlanespdlmt_var.m_elevation.append(0x01020304)
    cloudtovehiclecmdlanespdlmt_var.m_pointNum.value = len(path_cloudtovehiclecmdlanespdlmt_var.m_longitude)
    cloudtovehiclecmdlanespdlmt_var.m_laneNum = ctypes.c_int(0)
    speed_cloudtovehiclecmdlanespdlmt_var = cloud.speed_CloudToVehicleCmdLaneSpdLmt(
        cloudtovehiclecmdlanespdlmt_var.m_laneNum)
    cloudtovehiclecmdlanespdlmt_var.m_speed = speed_cloudtovehiclecmdlanespdlmt_var
    speed_cloudtovehiclecmdlanespdlmt_var.m_laneId.append(0x01)
    speed_cloudtovehiclecmdlanespdlmt_var.m_speedLimit.append(0x01)
    cloudtovehiclecmdlanespdlmt_var.m_laneNum.value = len(speed_cloudtovehiclecmdlanespdlmt_var.m_laneId)
    total_size = cloudtovehiclecmdlanespdlmt_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdlanespdlmt_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdlanespdlmtres_var = cloud.CloudToVehicleCmdLaneSpdLmtRes()
    cloudtovehiclecmdlanespdlmtres_var.m_cmdType = 0x01
    cloudtovehiclecmdlanespdlmtres_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdlanespdlmtres_var.m_doFlag = 0x01
    total_size = cloudtovehiclecmdlanespdlmtres_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdlanespdlmtres_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdrampintenetchange_var = cloud.CloudToVehicleCmdRampIntenetChange()
    cloudtovehiclecmdrampintenetchange_var.m_cmdType = 0x01
    cloudtovehiclecmdrampintenetchange_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdrampintenetchange_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdrampintenetchange_var.m_alertType = 0x0102
    cloudtovehiclecmdrampintenetchange_var.m_alertRadius = 0x0102
    cloudtovehiclecmdrampintenetchange_var.m_pointNum = ctypes.c_int(0)
    path_cloudtovehiclecmdrampintenetchange_var = cloud.path_CloudToVehicleCmdRampIntenetChange(
        cloudtovehiclecmdrampintenetchange_var.m_pointNum)
    cloudtovehiclecmdrampintenetchange_var.m_path = path_cloudtovehiclecmdrampintenetchange_var
    path_cloudtovehiclecmdrampintenetchange_var.m_longitude.append(0x01020304)
    path_cloudtovehiclecmdrampintenetchange_var.m_latitude.append(0x01020304)
    path_cloudtovehiclecmdrampintenetchange_var.m_elevation.append(0x01020304)
    path_cloudtovehiclecmdrampintenetchange_var.m_dtc.append(0x0102)
    path_cloudtovehiclecmdrampintenetchange_var.m_ttc.append(0x0102)
    cloudtovehiclecmdrampintenetchange_var.m_pointNum.value = len(
        path_cloudtovehiclecmdrampintenetchange_var.m_longitude)
    total_size = cloudtovehiclecmdrampintenetchange_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdrampintenetchange_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdrampintenetchangeres_var = cloud.VehicleToCloudCmdRampIntenetChangeRes()
    vehicletocloudcmdrampintenetchangeres_var.m_cmdType = 0x01
    vehicletocloudcmdrampintenetchangeres_var.m_seq = 0x0102030405060708
    vehicletocloudcmdrampintenetchangeres_var.m_doFlag = 0x01
    total_size = vehicletocloudcmdrampintenetchangeres_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdrampintenetchangeres_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    cloudtovehiclecmdfcw_var = cloud.CloudToVehicleCmdFcw()
    cloudtovehiclecmdfcw_var.m_cmdType = 0x01
    cloudtovehiclecmdfcw_var.m_uuid = b'____________________________________'
    cloudtovehiclecmdfcw_var.m_seq = 0x0102030405060708
    cloudtovehiclecmdfcw_var.m_alertType = 0x0102
    cloudtovehiclecmdfcw_var.m_alertRadius = 0x0102
    cloudtovehiclecmdfcw_var.m_pointNum = ctypes.c_int(0)
    path_cloudtovehiclecmdfcw_var = cloud.path_CloudToVehicleCmdFcw(cloudtovehiclecmdfcw_var.m_pointNum)
    cloudtovehiclecmdfcw_var.m_path = path_cloudtovehiclecmdfcw_var
    path_cloudtovehiclecmdfcw_var.m_longitude.append(0x01020304)
    path_cloudtovehiclecmdfcw_var.m_latitude.append(0x01020304)
    path_cloudtovehiclecmdfcw_var.m_elevation.append(0x01020304)
    cloudtovehiclecmdfcw_var.m_pointNum.value = len(path_cloudtovehiclecmdfcw_var.m_longitude)
    cloudtovehiclecmdfcw_var.m_objType = 0x01
    cloudtovehiclecmdfcw_var.m_fcwLevel = 0x01
    total_size = cloudtovehiclecmdfcw_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = cloudtovehiclecmdfcw_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)

    vehicletocloudcmdfcwres_var = cloud.VehicleToCloudCmdFcwRes()
    vehicletocloudcmdfcwres_var.m_cmdType = 0x01
    vehicletocloudcmdfcwres_var.m_seq = 0x0102030405060708
    vehicletocloudcmdfcwres_var.m_doFlag = 0x01
    total_size = vehicletocloudcmdfcwres_var.calc_real_size()
    buffer = bytearray(total_size)
    pos = vehicletocloudcmdfcwres_var.fill_buffer(buffer, 0)
    print(len(buffer))
    print(buffer)
