package com.example.mobiledatanotifier

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder

/**
 * 第二前台服务：与 OverlayService 互为"双保险"。
 * 当 OverlayService 被系统杀掉时，GuardService 仍然存活，可借机上位、拉起 OverlayService
 * 并重新注册所有保活组件。反之，OverlayService 拉起时也会同步启动本服务。
 */
class GuardService : Service() {

    companion object {
        private const val CH = "ch_guard"
        private const val FG_ID = 7001

        fun start(ctx: Context) {
            val i = Intent(ctx, GuardService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }

        @Volatile
        var isRunning: Boolean = false
            private set
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(FG_ID, buildNotification())
        isRunning = true
        ensureKeepAlive()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        ensureKeepAlive()
        return START_STICKY
    }

    override fun onDestroy() {
        isRunning = false
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CH, "保活守护", NotificationManager.IMPORTANCE_LOW).apply {
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(): Notification {
        val pi = PendingIntent.getActivity(this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CH)
            .setContentTitle("移动数据通知器守护运行中")
            .setContentText("双前台保活服务")
            .setSmallIcon(R.drawable.ic_notify)
            .setOngoing(true)
            .setContentIntent(pi)
            .build()
    }

    private fun ensureKeepAlive() {
        if (!OverlayService.isRunning) {
            try { OverlayService.start(this) } catch (_: Exception) {}
        }
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
    }
}
