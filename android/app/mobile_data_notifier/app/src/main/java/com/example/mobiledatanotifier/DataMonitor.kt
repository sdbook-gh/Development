package com.example.mobiledatanotifier

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.TrafficStats
import android.telephony.TelephonyManager
import android.util.Log

/** 读取移动数据开关状态与流量消耗。 */
object DataMonitor {

    /** 移动数据开关是否开启（需 READ_PHONE_STATE，API 26+）。 */
    fun isMobileDataEnabled(ctx: Context): Boolean {
        return try {
            val tm = ctx.getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
            tm.isDataEnabled
        } catch (e: Exception) {
            // 回退：判断是否存在活动的蜂窝网络
            try {
                val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
                val net = cm.activeNetwork
                val caps = cm.getNetworkCapabilities(net)
                caps != null && caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)
            } catch (e2: Exception) {
                false
            }
        }
    }

    /** 移动流量：开机以来累计收发字节。 */
    fun mobileBytesSinceBoot(): Long {
        return try {
            TrafficStats.getMobileRxBytes() + TrafficStats.getMobileTxBytes()
        } catch (e: Exception) {
            0L
        }
    }

    /** 本次统计：自上次重置以来的流量。 */
    fun mobileBytesThisSession(ctx: Context): Long {
        val cur = mobileBytesSinceBoot()
        val base = Prefs.getUsageBaseline(ctx)
        return (cur - base).coerceAtLeast(0L)
    }

    /** 自动把本次统计的增量累加到累计流量中并保存。
     * 每次 UI 刷新时调用，保证累计值随实际使用持续更新。
     */
    fun autoUpdateCumulative(ctx: Context) {
        try {
            val cur = mobileBytesSinceBoot()
            val base = Prefs.getUsageBaseline(ctx)
            val delta = (cur - base).coerceAtLeast(0L)
            if (delta > 0) {
                Prefs.setCumulativeUsage(ctx, Prefs.getCumulativeUsage(ctx) + delta)
                Prefs.setUsageBaseline(ctx, cur)
            }
        } catch (e: Exception) {
            Log.w("DataMonitor", "autoUpdateCumulative failed", e)
        }
    }

    /** 解析用户输入的流量值，支持：
     * - 纯数字 → 字节
     * - 末尾带 M/m → MB（支持小数，如 1.5m）
     * - 末尾带 G/g → GB（支持小数，如 2.3g）
     * - 末尾带 K/k → KB
     * 不合法输入返回 -1。
     */
    fun parseCumulativeInput(input: String): Long {
        val trimmed = input.trim()
        if (trimmed.isEmpty()) return -1
        return try {
            when {
                trimmed.endsWith("G", true) || trimmed.endsWith("g", true) -> {
                    val v = trimmed.dropLast(1).trim().toDouble()
                    (v * 1024 * 1024 * 1024).toLong()
                }
                trimmed.endsWith("M", true) || trimmed.endsWith("m", true) -> {
                    val v = trimmed.dropLast(1).trim().toDouble()
                    (v * 1024 * 1024).toLong()
                }
                trimmed.endsWith("K", true) || trimmed.endsWith("k", true) -> {
                    val v = trimmed.dropLast(1).trim().toDouble()
                    (v * 1024).toLong()
                }
                else -> trimmed.toLong()
            }
        } catch (_: NumberFormatException) {
            -1
        }
    }

    fun formatBytes(bytes: Long): String {
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        return when {
            gb >= 1.0 -> String.format("%.2f GB", gb)
            mb >= 1.0 -> String.format("%.2f MB", mb)
            kb >= 1.0 -> String.format("%.1f KB", kb)
            else -> "$bytes B"
        }
    }

    /** 悬浮窗用的精简格式：仅 K 或 M。 */
    fun formatCompact(bytes: Long): String {
        val mb = bytes / 1048576.0
        val kb = bytes / 1024.0
        return if (mb >= 1.0) String.format("%.2f M", mb) else String.format("%.1f K", kb)
    }
}
