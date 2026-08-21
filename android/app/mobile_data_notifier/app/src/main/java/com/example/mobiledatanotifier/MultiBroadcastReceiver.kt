package com.example.mobiledatanotifier

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * 多广播监听兜底：在系统广播发生时检查 OverlayService 是否存活，若死则拉起。
 *
 * 监听事件（仅保留 Android 8.0+ 允许静态注册接收的系统广播）：
 * - 电量低 / 电量恢复 / 接入电源
 * - 闹钟（AlarmKeeper 触发：周期心跳、划掉重启、滚动短延迟）
 * 注：SCREEN_ON/OFF、USER_PRESENT、TIME_TICK 等广播已改为
 * OverlayService 内动态注册（系统限制无法静态接收）。
 */
class MultiBroadcastReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent?) {
        val action = intent?.action ?: return
        if (!Prefs.isServiceEnabled(context)) return
        when (action) {
            Intent.ACTION_BATTERY_LOW,
            Intent.ACTION_BATTERY_OKAY,
            Intent.ACTION_POWER_CONNECTED,
            "com.example.mobiledatanotifier.KEEP_ALIVE" -> {
                try { OverlayService.startIfEnabled(context) } catch (_: Exception) {}
                try { GuardService.startIfEnabled(context) } catch (_: Exception) {}
                try { KeepAliveJob.schedule(context) } catch (_: Exception) {}
                try { WatchdogJob.schedule(context) } catch (_: Exception) {}
                try { AlarmKeeper.register(context) } catch (_: Exception) {}
                try { AlarmKeeper.scheduleRolling(context) } catch (_: Exception) {}
                try { ScheduleManager.rescheduleAll(context) } catch (_: Exception) {}
            }
        }
    }
}
