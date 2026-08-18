package com.example.mobiledatanotifier

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build

/**
 * 精确闹钟兜底：使用 AlarmManager.setExactAndAllowWhileIdle 保证即使在 Doze 模式下
 * 也能按时唤醒，触发守护接收器检查并重启服务。
 */
object AlarmKeeper {

    private const val INTERVAL_MS = 15 * 60 * 1000L
    private const val EXTRA_KEEP_ALIVE = "com.example.mobiledatanotifier.KEEP_ALIVE"

    /** 注册精确闹钟。设备重启后由 BootReceiver / WatchdogJob 重新注册。 */
    fun register(ctx: Context) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(ctx, MultiBroadcastReceiver::class.java).apply {
            action = EXTRA_KEEP_ALIVE
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        val pi = PendingIntent.getBroadcast(ctx, 0, intent, flags)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP,
                System.currentTimeMillis() + INTERVAL_MS, pi)
        } else {
            am.setExact(AlarmManager.RTC_WAKEUP,
                System.currentTimeMillis() + INTERVAL_MS, pi)
        }
    }

    fun isRegistered(ctx: Context): Boolean {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(ctx, MultiBroadcastReceiver::class.java).apply {
            action = EXTRA_KEEP_ALIVE
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return am.canScheduleExactAlarms() &&
            am.getNextAlarmClock() != null
    }
}
