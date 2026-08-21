package com.example.mobiledatanotifier

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper

/**
 * 第二前台服务：跑在独立 `:guard` 进程，与 OverlayService 互为双保险。
 * 主进程被杀时本进程仍可能存活，可立刻 startForegroundService 拉回悬浮窗。
 * 若系统对整个 UID 强制停止，两个进程仍会一起死（系统限制）。
 */
class GuardService : Service() {

    companion object {
        private const val CH = "ch_guard"
        private const val FG_ID = 7001
        private const val HEARTBEAT_MS = 2500L

        fun start(ctx: Context) {
            val i = Intent(ctx, GuardService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }

        fun startIfEnabled(ctx: Context) {
            if (!Prefs.isServiceEnabled(ctx)) return
            start(ctx)
        }

        fun stop(ctx: Context) {
            try { ctx.stopService(Intent(ctx, GuardService::class.java)) } catch (_: Exception) {}
        }
    }

    private val handler = Handler(Looper.getMainLooper())
    private val heartbeat = object : Runnable {
        override fun run() {
            if (!Prefs.isServiceEnabled(this@GuardService)) return
            ensureKeepAlive()
            try { AlarmKeeper.scheduleRolling(this@GuardService) } catch (_: Exception) {}
            handler.postDelayed(this, HEARTBEAT_MS)
        }
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(FG_ID, buildNotification())
        ensureKeepAlive()
        handler.post(heartbeat)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!Prefs.isServiceEnabled(this)) {
            try { AlarmKeeper.cancel(this) } catch (_: Exception) {}
            handler.removeCallbacks(heartbeat)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        ensureKeepAlive()
        return START_STICKY
    }

    override fun onDestroy() {
        handler.removeCallbacks(heartbeat)
        if (Prefs.isServiceEnabled(this)) {
            try { AlarmKeeper.scheduleRestart(this, 1000L) } catch (_: Exception) {}
            try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
            try { WatchdogJob.scheduleImmediate(this) } catch (_: Exception) {}
        }
        super.onDestroy()
    }

    /** 任务被划掉时通过系统持有的闹钟延迟重启，双保险。 */
    override fun onTaskRemoved(rootIntent: Intent?) {
        if (Prefs.isServiceEnabled(this)) {
            try { AlarmKeeper.scheduleRestart(this, 1000L) } catch (_: Exception) {}
            try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
            try { WatchdogJob.scheduleImmediate(this) } catch (_: Exception) {}
            try { if (!ProcessUtil.isOverlayAlive(this)) OverlayService.start(this) } catch (_: Exception) {}
        }
        super.onTaskRemoved(rootIntent)
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
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
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
        if (!Prefs.isServiceEnabled(this)) {
            try { AlarmKeeper.cancel(this) } catch (_: Exception) {}
            handler.removeCallbacks(heartbeat)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }
        if (!ProcessUtil.isOverlayAlive(this)) {
            try { OverlayService.start(this) } catch (_: Exception) {}
        }
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
    }
}
