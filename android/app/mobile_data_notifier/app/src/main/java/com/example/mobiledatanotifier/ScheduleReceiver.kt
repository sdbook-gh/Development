package com.example.mobiledatanotifier

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Settings

/**
 * 到点提醒接收器：发送高优先级通知（声音+震动），并重新排程次日。
 */
class ScheduleReceiver : BroadcastReceiver() {

    companion object {
        const val ACTION_REMIND = "com.example.mobiledatanotifier.REMIND"
        const val EXTRA_PERIOD_ID = "period_id"
        const val EXTRA_TYPE = "type"  // 0=关闭提醒, 1=开启提醒
    }

    override fun onReceive(context: Context, intent: Intent) {
        val type = intent.getIntExtra(EXTRA_TYPE, 0)
        val periodId = intent.getLongExtra(EXTRA_PERIOD_ID, -1L)
        showReminder(context, type, periodId)
        // 重新排程（含本时段次日）
        try { ScheduleManager.rescheduleAll(context) } catch (_: Exception) {}
    }

    private fun showReminder(ctx: Context, type: Int, periodId: Long) {
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val title: String
        val text: String
        if (type == 0) {
            title = ctx.getString(R.string.reminder_close_title)
            text = ctx.getString(R.string.reminder_close_text)
        } else {
            title = ctx.getString(R.string.reminder_open_title)
            text = ctx.getString(R.string.reminder_open_text)
        }

        val dataIntent = Intent(Settings.ACTION_DATA_USAGE_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val settingsPi = PendingIntent.getActivity(ctx, 1, dataIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

        // 走高重要性渠道（CH_REMINDER 已配置声音+震动）
        val notif = Notification.Builder(ctx, OverlayService.CH_REMINDER)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notify)
            .setContentIntent(settingsPi)
            .addAction(0, ctx.getString(R.string.action_close_data), settingsPi)
            .setAutoCancel(true)
            .setPriority(Notification.PRIORITY_HIGH)
            .setCategory(Notification.CATEGORY_REMINDER)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .build()

        val id = 3000 + (periodId.toInt() and 0x3FFF) * 2 + type
        nm.notify(id, notif)
    }
}
