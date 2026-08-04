package com.example.mobiledatanotifier

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.TrafficStats
import android.telephony.TelephonyManager

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
