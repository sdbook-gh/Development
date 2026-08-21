package com.example.mobiledatanotifier

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build

/**
 * 精确闹钟兜底：使用 AlarmManager.setExactAndAllowWhileIdle 保证即使在 Doze 模式下
 * 也能按时唤醒，触发守护接收器检查并重启服务。
 *
 * 滚动短延迟闹钟由系统持有：进程被杀但未被强制停止时，约 3 秒内仍会被拉回。
 */
object AlarmKeeper {

    private const val INTERVAL_MS = 5 * 60 * 1000L
    private const val ROLLING_MS = 3000L
    private const val RC_PERIODIC = 0
    private const val RC_RESTART = 1
    private const val RC_ROLLING = 2
    private const val EXTRA_KEEP_ALIVE = "com.example.mobiledatanotifier.KEEP_ALIVE"

    /**
     * 注册周期心跳闹钟。Android 14+ 默认不授予 SCHEDULE_EXACT_ALARM，
     * 无权限时降级为 setAndAllowWhileIdle（inexact），保证闹钟链不断。
     * 设备重启后由 BootReceiver / MultiBroadcastReceiver 重新注册。
     */
    fun register(ctx: Context) {
        schedule(ctx, INTERVAL_MS, RC_PERIODIC)
    }

    /**
     * 任务被从最近列表划掉后的延迟重启：闹钟由系统持有，
     * 不随进程死亡而丢失，触发后由 MultiBroadcastReceiver 拉起全部保活组件。
     */
    fun scheduleRestart(ctx: Context, delayMs: Long = 1000L) {
        if (!Prefs.isServiceEnabled(ctx)) return
        schedule(ctx, delayMs, RC_RESTART)
    }

    /** 服务运行期间持续刷新「约 3 秒后」的精确闹钟。 */
    fun scheduleRolling(ctx: Context, delayMs: Long = ROLLING_MS) {
        if (!Prefs.isServiceEnabled(ctx)) return
        schedule(ctx, delayMs, RC_ROLLING)
    }

    /** 用户主动停止保活时取消全部闹钟，避免停了又被拉起。 */
    fun cancel(ctx: Context) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        listOf(RC_PERIODIC, RC_RESTART, RC_ROLLING).forEach { rc ->
            val pi = pendingIntent(ctx, rc)
            try { am.cancel(pi) } catch (_: Exception) {}
            try { pi.cancel() } catch (_: Exception) {}
        }
    }

    private fun pendingIntent(ctx: Context, requestCode: Int): PendingIntent {
        val intent = Intent(ctx, MultiBroadcastReceiver::class.java).apply {
            action = EXTRA_KEEP_ALIVE
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getBroadcast(ctx, requestCode, intent, flags)
    }

    private fun schedule(ctx: Context, delayMs: Long, requestCode: Int) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val pi = pendingIntent(ctx, requestCode)
        val triggerAt = System.currentTimeMillis() + delayMs
        val canExact = Build.VERSION.SDK_INT < Build.VERSION_CODES.S || am.canScheduleExactAlarms()
        if (canExact) {
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
                } else {
                    am.setExact(AlarmManager.RTC_WAKEUP, triggerAt, pi)
                }
                return
            } catch (_: SecurityException) {
                // 精确闹钟权限被收回，降级为 inexact
            }
        }
        am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
    }
}
