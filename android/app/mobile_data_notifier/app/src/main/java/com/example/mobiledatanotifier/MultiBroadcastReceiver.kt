package com.example.mobiledatanotifier

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.wifi.WifiManager
import android.telephony.TelephonyManager

/**
 * 多广播监听兜底：在系统广播发生时检查 OverlayService 是否存活，若死则拉起。
 *
 * 监听事件（覆盖尽可能多的系统触发点）：
 * - 亮屏 / 灭屏 / 用户解锁
 * - 网络状态变化（蜂窝 + WiFi）
 * - SIM 卡状态变化（移动网络注册状态的强信号）
 * - 电量变化、充电插入
 * - 耳机插拔、充电底座
 * - 系统每分钟心跳（ACTION_TIME_TICK）
 * - 精确闹钟（AlarmKeeper 触发）
 */
class MultiBroadcastReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent?) {
        val action = intent?.action
        if (!OverlayService.isRunning) {
            when (action) {
                Intent.ACTION_SCREEN_ON,
                Intent.ACTION_SCREEN_OFF,
                Intent.ACTION_USER_PRESENT,
                "android.net.conn.NETWORK_CONNECTED",
                "android.net.conn.NETWORK_DISCONNECTED",
                WifiManager.WIFI_STATE_CHANGED_ACTION,
                "android.intent.action.SIM_STATE_CHANGED",
                Intent.ACTION_BATTERY_LOW,
                Intent.ACTION_BATTERY_OKAY,
                Intent.ACTION_POWER_CONNECTED,
                Intent.ACTION_HEADSET_PLUG,
                Intent.ACTION_DOCK_EVENT,
                Intent.ACTION_TIME_TICK,
                "com.example.mobiledatanotifier.KEEP_ALIVE" -> {
                    try { OverlayService.start(context) } catch (_: Exception) {}
                    try { GuardService.start(context) } catch (_: Exception) {}
                    try { KeepAliveJob.schedule(context) } catch (_: Exception) {}
                    try { WatchdogJob.schedule(context) } catch (_: Exception) {}
                    try { AlarmKeeper.register(context) } catch (_: Exception) {}
                    try { ScheduleManager.rescheduleAll(context) } catch (_: Exception) {}
                }
            }
        }
        if (action == "com.example.mobiledatanotifier.KEEP_ALIVE") {
            try { AlarmKeeper.register(context) } catch (_: Exception) {}
        }
    }
}
