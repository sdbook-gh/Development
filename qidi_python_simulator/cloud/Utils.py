import sys

def decodeRemainLength(remainLenTuple:tuple) -> int:
    deodeRemainLen = remainLenTuple[0] << 16 | remainLenTuple[1] << 8 | remainLenTuple[2]
    return deodeRemainLen

def encodeRemainLength(remainLen:int) -> tuple:
    remainLenBytes = remainLen.to_bytes(3, sys.byteorder)
    encodeRemainLenTuple = (remainLenBytes[0], remainLenBytes[1], remainLenBytes[2])
    return encodeRemainLenTuple
