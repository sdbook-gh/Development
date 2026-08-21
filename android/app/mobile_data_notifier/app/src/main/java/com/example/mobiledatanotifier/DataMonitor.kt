package com.example.mobiledatanotifier

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.TrafficStats
import android.telephony.TelephonyManager
import android.app.usage.NetworkStatsManager
import android.os.SystemClock
import android.util.Log

enum class TrafficScope(val prefValue: String, val overlayPrefix: String) {
    MOBILE("mobile", "移"),
    WIFI("wifi", "WiFi"),
    ALL("all", "全");

    companion object {
        fun fromPref(value: String?): TrafficScope {
            for (s in values()) {
                if (s.prefValue == value) return s
            }
            return MOBILE
        }
    }
}

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

    /**
     * 开机以来累计收发字节。优先 TrafficStats（内核计数器，实时）；
     * UNSUPPORTED / 无效时再降级 NetworkStatsManager（按时间桶聚合，会滞后）。
     */
    fun bytesSinceBoot(ctx: Context, scope: TrafficScope = Prefs.getTrafficScope(ctx)): Long {
        val fromTraffic = trafficBytes(scope)
        if (fromTraffic >= 0L) return fromTraffic
        return nsmBytes(ctx, scope)
    }

    /** 移动流量：开机以来累计收发字节（TrafficStats，可能为 UNSUPPORTED）。 */
    fun mobileBytesSinceBoot(): Long = trafficBytes(TrafficScope.MOBILE).coerceAtLeast(0L)

    /** 探测 PACKAGE_USAGE_STATS：能读到 NetworkStats 则 >= 0。 */
    fun mobileBytesSinceBootV2(ctx: Context): Long = nsmQuery(ctx, ConnectivityManager.TYPE_MOBILE)

    /** 本次统计：自上次重置以来的流量。 */
    fun bytesThisSession(ctx: Context): Long {
        val cur = bytesSinceBoot(ctx)
        if (cur < 0L) return 0L
        val base = Prefs.getUsageBaseline(ctx)
        return (cur - base).coerceAtLeast(0L)
    }

    /** 切换统计范围：先把旧范围增量写入累计，再重设基线（累计值保留，避免跳变）。 */
    fun switchTrafficScope(ctx: Context, scope: TrafficScope) {
        autoUpdateCumulative(ctx)
        Prefs.setTrafficScope(ctx, scope)
        val cur = bytesSinceBoot(ctx, scope)
        if (cur >= 0L) Prefs.setUsageBaseline(ctx, cur)
    }

    /**
     * 自动把本次统计的增量累加到累计流量中并保存。
     * 基线与当前值必须来自同一数据源。重启后 TrafficStats 归零，需重设基线、不把负增量写入累计。
     */
    fun autoUpdateCumulative(ctx: Context) {
        try {
            val cur = bytesSinceBoot(ctx)
            if (cur < 0L) return
            val elapsed = SystemClock.elapsedRealtime()
            val baseElapsed = Prefs.getUsageBaselineElapsed(ctx)
            val base = Prefs.getUsageBaseline(ctx)
            if (baseElapsed == 0L || elapsed < baseElapsed || cur < base) {
                Prefs.setUsageBaseline(ctx, cur)
                return
            }
            val delta = cur - base
            if (delta > 0L) {
                Prefs.setCumulativeUsage(ctx, Prefs.getCumulativeUsage(ctx) + delta)
                Prefs.setUsageBaseline(ctx, cur)
            }
        } catch (e: Exception) {
            Log.w("DataMonitor", "autoUpdateCumulative failed", e)
        }
    }

    private fun trafficBytes(scope: TrafficScope): Long {
        return try {
            val mobile = sumTraffic(TrafficStats.getMobileRxBytes(), TrafficStats.getMobileTxBytes())
            val total = sumTraffic(TrafficStats.getTotalRxBytes(), TrafficStats.getTotalTxBytes())
            when (scope) {
                TrafficScope.MOBILE -> mobile
                TrafficScope.ALL -> total
                TrafficScope.WIFI -> {
                    if (total < 0L || mobile < 0L) -1L
                    else (total - mobile).coerceAtLeast(0L)
                }
            }
        } catch (e: Exception) {
            -1L
        }
    }

    private fun sumTraffic(rx: Long, tx: Long): Long {
        if (rx < 0L || tx < 0L) return -1L
        return rx + tx
    }

    private fun nsmBytes(ctx: Context, scope: TrafficScope): Long {
        return when (scope) {
            TrafficScope.MOBILE -> nsmQuery(ctx, ConnectivityManager.TYPE_MOBILE)
            TrafficScope.WIFI -> nsmQuery(ctx, ConnectivityManager.TYPE_WIFI)
            TrafficScope.ALL -> {
                val mobile = nsmQuery(ctx, ConnectivityManager.TYPE_MOBILE)
                val wifi = nsmQuery(ctx, ConnectivityManager.TYPE_WIFI)
                if (mobile < 0L && wifi < 0L) -1L
                else mobile.coerceAtLeast(0L) + wifi.coerceAtLeast(0L)
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun nsmQuery(ctx: Context, networkType: Int): Long {
        return try {
            val ns = ctx.getSystemService(Context.NETWORK_STATS_SERVICE) as NetworkStatsManager
            val bootTime = System.currentTimeMillis() - SystemClock.elapsedRealtime()
            val now = System.currentTimeMillis()
            val bucket = ns.querySummaryForDevice(networkType, null, bootTime, now)
            val sum = bucket.rxBytes + bucket.txBytes
            if (sum < 0L) -1L else sum
        } catch (e: SecurityException) {
            -1L
        } catch (e: Exception) {
            -1L
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
