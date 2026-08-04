package com.example.mobiledatanotifier

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import java.util.Calendar

/** 根据 Prefs 中的时段，用 AlarmManager 排程下一次"关闭/开启"提醒。 */
object ScheduleManager {

    private const val RC_BASE = 1000
    private const val TYPE_CLOSE = 0  // 到关闭时间
    private const val TYPE_OPEN = 1   // 到开启时间

    fun rescheduleAll(ctx: Context) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val periods = Prefs.getPeriods(ctx)
        // 先全部取消
        for (p in periods) {
            cancelAlarm(ctx, am, p.id, TYPE_CLOSE)
            cancelAlarm(ctx, am, p.id, TYPE_OPEN)
        }
        // 重新排程启用的时段
        for (p in periods) {
            if (!p.enabled) continue
            scheduleNext(ctx, am, p, TYPE_CLOSE)
            scheduleNext(ctx, am, p, TYPE_OPEN)
        }
    }

    private fun nextTrigger(hour: Int, min: Int): Long {
        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, hour)
            set(Calendar.MINUTE, min)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
            if (timeInMillis <= System.currentTimeMillis()) add(Calendar.DAY_OF_YEAR, 1)
        }
        return cal.timeInMillis
    }

    private fun scheduleNext(ctx: Context, am: AlarmManager, p: TimePeriod, type: Int) {
        val triggerAt = if (type == TYPE_CLOSE)
            nextTrigger(p.startHour, p.startMin)
        else
            nextTrigger(p.endHour, p.endMin)
        val pi = makePendingIntent(ctx, p.id, type)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !am.canScheduleExactAlarms()) {
                am.setWindow(AlarmManager.RTC_WAKEUP, triggerAt, 60_000L, pi)
            } else {
                am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
            }
        } catch (e: SecurityException) {
            am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
        }
    }

    private fun cancelAlarm(ctx: Context, am: AlarmManager, periodId: Long, type: Int) {
        am.cancel(makePendingIntent(ctx, periodId, type))
    }

    private fun makePendingIntent(ctx: Context, periodId: Long, type: Int): PendingIntent {
        val i = Intent(ctx, ScheduleReceiver::class.java).apply {
            action = ScheduleReceiver.ACTION_REMIND
            putExtra(ScheduleReceiver.EXTRA_PERIOD_ID, periodId)
            putExtra(ScheduleReceiver.EXTRA_TYPE, type)
        }
        val rc = RC_BASE + (periodId.toInt() and 0x3FFF) * 2 + type
        return PendingIntent.getBroadcast(ctx, rc, i,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }
}
