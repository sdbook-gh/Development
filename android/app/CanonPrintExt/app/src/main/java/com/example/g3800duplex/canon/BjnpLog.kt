package com.example.g3800duplex.canon

import android.util.Log

/**
 * BJNP / CLSS transport diagnostics. Filter logcat: `adb logcat -s G3800Bjnp`
 */
object BjnpLog {
    const val TAG = "G3800Bjnp"

    fun i(stage: String, msg: String) {
        Log.i(TAG, "[$stage] $msg")
    }

    fun w(stage: String, msg: String, t: Throwable? = null) {
        if (t != null) Log.w(TAG, "[$stage] $msg", t) else Log.w(TAG, "[$stage] $msg")
    }

    fun e(stage: String, msg: String, t: Throwable? = null) {
        if (t != null) Log.e(TAG, "[$stage] $msg", t) else Log.e(TAG, "[$stage] $msg")
    }

    fun d(stage: String, msg: String) {
        Log.d(TAG, "[$stage] $msg")
    }

    fun hex(bytes: ByteArray?, max: Int = 48): String {
        if (bytes == null || bytes.isEmpty()) return "(empty)"
        val n = minOf(bytes.size, max)
        val sb = StringBuilder(n * 3)
        for (i in 0 until n) {
            if (i > 0) sb.append(' ')
            sb.append(String.format("%02X", bytes[i].toInt() and 0xFF))
        }
        if (bytes.size > max) sb.append(" …(+${bytes.size - max})")
        return sb.toString()
    }

    fun openRcMessage(rc: Int): String = when (rc) {
        0 -> "OK"
        -1 -> "UDP 握手失败/忙/无 session（无应答、seq 不匹配、payloadLen≠0 忙、或 sessionId=0）"
        -2 -> "非法状态（ip 空或 socket 已打开）"
        -3 -> "TCP:8611 连接或 idle 配置失败"
        else -> "未知 rc=$rc"
    }
}
