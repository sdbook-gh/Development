package com.example.mobiledatanotifier

import android.animation.ValueAnimator
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.media.AudioAttributes
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.telephony.PhoneStateListener
import android.telephony.ServiceState
import android.telephony.TelephonyManager
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView

/**
 * 保活前台服务 + 顶部悬浮窗。
 * - 前台服务通知走低优先级渠道（静音、不震动）。
 * - 悬浮窗显示移动数据开关状态、已用流量，含"关闭流量"按钮（跳转系统设置）。
 * - START_STICKY：被杀后系统尝试重建。
 */
class OverlayService : Service() {

    companion object {
        const val CH_FG = "ch_fg"
        const val CH_REMINDER = "ch_reminder"
        const val ACTION_SHOW_OVERLAY = "com.example.mobiledatanotifier.SHOW_OVERLAY"
        const val ACTION_HIDE_OVERLAY = "com.example.mobiledatanotifier.HIDE_OVERLAY"
        const val ACTION_STOP = "com.example.mobiledatanotifier.STOP"

        /** 保活服务是否在运行（供 MainActivity / KeepAliveJob 按需判断） */
        @Volatile
        var isRunning: Boolean = false
            private set

        /** 悬浮窗是否在显示（供 MainActivity 按需判断） */
        @Volatile
        var isOverlayShown: Boolean = false
            private set

        fun start(ctx: Context) {
            val i = Intent(ctx, OverlayService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }
    }

    private lateinit var wm: WindowManager
    private var overlayView: View? = null
    private lateinit var handler: Handler
    private var connectivityCallback: ConnectivityManager.NetworkCallback? = null
    private var phoneStateListener: PhoneStateListener? = null
    private var colorAnimator: ValueAnimator? = null
    // 用户是否主动隐藏了悬浮窗；为 false 时开机/被杀重建后可自愈重新挂载
    private var overlayHiddenByUser = false

    override fun onCreate() {
        super.onCreate()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        handler = Handler(Looper.getMainLooper())
        createChannels()
        startForeground(1, buildFgNotification())
        registerConnectivity()
        registerPhoneState()
        addOverlay()
        handler.post(updateRunnable)
        isRunning = true
        // 调度兜底心跳：被系统杀后由 JobScheduler 拉起
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_HIDE_OVERLAY -> { overlayHiddenByUser = true; removeOverlay() }
            ACTION_SHOW_OVERLAY -> addOverlay()
            ACTION_STOP -> {
                shutdown()
                return START_NOT_STICKY
            }
        }
        // 无论以何种方式被拉起，都重新确保所有保活组件生效
        ensureKeepAlive()
        return START_STICKY
    }

    /** 确保所有保活组件已注册。失败时通过 AlarmKeeper 兜底。 */
    private fun ensureKeepAlive() {
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
        try { if (!GuardService.isRunning) GuardService.start(this) } catch (_: Exception) {}
    }

    private fun shutdown() {
        unregisterPhoneState()
        handler.removeCallbacks(updateRunnable)
        unregisterConnectivity()
        removeOverlay()
        stopForeground(STOP_FOREGROUND_REMOVE)
        isRunning = false
        stopSelf()
    }

    override fun onDestroy() {
        unregisterPhoneState()
        handler.removeCallbacks(updateRunnable)
        unregisterConnectivity()
        removeOverlay()
        isRunning = false
        super.onDestroy()
    }

    /** 任务被从最近列表划掉时，立即重启保活服务。 */
    override fun onTaskRemoved(rootIntent: Intent?) {
        try { OverlayService.start(this) } catch (_: Exception) {}
        super.onTaskRemoved(rootIntent)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ---------- 通知渠道 ----------
    private fun createChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val fg = NotificationChannel(CH_FG, getString(R.string.channel_fg_name),
            NotificationManager.IMPORTANCE_LOW).apply {
            description = getString(R.string.channel_fg_desc)
            setShowBadge(false)
        }
        nm.createNotificationChannel(fg)

        // 高重要性提醒渠道：带声音 + 震动
        val reminder = NotificationChannel(CH_REMINDER, getString(R.string.channel_reminder_name),
            NotificationManager.IMPORTANCE_HIGH).apply {
            description = getString(R.string.channel_reminder_desc)
            enableVibration(true)
            vibrationPattern = longArrayOf(0, 300, 200, 300)
            setSound(
                Settings.System.DEFAULT_NOTIFICATION_URI,
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_NOTIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            lockscreenVisibility = Notification.VISIBILITY_PUBLIC
        }
        nm.createNotificationChannel(reminder)
    }

    private fun buildFgNotification(): Notification {
        val showIntent = Intent(this, OverlayService::class.java).setAction(ACTION_SHOW_OVERLAY)
        val showPi = PendingIntent.getService(this, 0, showIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val mainIntent = Intent(this, MainActivity::class.java)
        val mainPi = PendingIntent.getActivity(this, 0, mainIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CH_FG)
            .setContentTitle(getString(R.string.fg_notification_title))
            .setContentText(getString(R.string.fg_notification_text))
            .setSmallIcon(R.drawable.ic_notify)
            .setOngoing(true)
            .setContentIntent(mainPi)
            .addAction(0, getString(R.string.action_show_overlay), showPi)
            .build()
    }

    // ---------- 悬浮窗 ----------
    private fun overlayType(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else
            @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE

    private fun addOverlay() {
        overlayHiddenByUser = false
        if (overlayView != null) return
        if (!PermUtil.canDrawOverlays(this)) return
        val v = LayoutInflater.from(this).inflate(R.layout.view_overlay, null, false)
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            overlayType(),
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        )
        params.gravity = Gravity.TOP or Gravity.START
        params.x = 0
        params.y = 0
        v.findViewById<Button>(R.id.btn_close_data).setOnClickListener { openDataSettings() }
        enableDrag(v, params)
        try {
            wm.addView(v, params)
            overlayView = v
            isOverlayShown = true
            refreshOverlay()
        } catch (e: Exception) {
            // 权限缺失或异常
        }
    }

    private fun removeOverlay() {
        colorAnimator?.cancel()
        colorAnimator = null
        overlayView?.let {
            try { wm.removeView(it) } catch (_: Exception) {}
        }
        overlayView = null
        isOverlayShown = false
    }

    private fun enableDrag(v: View, params: WindowManager.LayoutParams) {
        var initX = 0; var initY = 0; var touchX = 0f; var touchY = 0f; var moved = false
        v.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> {
                    initX = params.x; initY = params.y
                    touchX = e.rawX; touchY = e.rawY; moved = false
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (e.rawX - touchX).toInt()
                    val dy = (e.rawY - touchY).toInt()
                    if (dx * dx + dy * dy > 25) moved = true
                    params.x = initX + dx
                    params.y = initY + dy
                    try { wm.updateViewLayout(v, params) } catch (_: Exception) {}
                }
                MotionEvent.ACTION_UP -> if (moved) return@setOnTouchListener true
            }
            false
        }
    }

    private fun openDataSettings() {
        val i = Intent(Settings.ACTION_DATA_USAGE_SETTINGS)
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        try {
            startActivity(i)
        } catch (e: Exception) {
            try {
                val fallback = Intent(Settings.ACTION_SETTINGS)
                fallback.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(fallback)
            } catch (_: Exception) {}
        }
    }

    private val updateRunnable = object : Runnable {
        override fun run() {
            // 自愈：开机后/被杀重建后若悬浮窗未挂载且未被用户主动隐藏，则重新添加
            if (overlayView == null && !overlayHiddenByUser && PermUtil.canDrawOverlays(this@OverlayService)) {
                addOverlay()
            }
            refreshOverlay()
            handler.postDelayed(this, 3000)
        }
    }

    private fun refreshOverlay() {
        val v = overlayView ?: return
        DataMonitor.autoUpdateCumulative(this)
        val on = DataMonitor.isMobileDataEnabled(this)
        val tvState = v.findViewById<TextView>(R.id.tv_state)
        tvState.text = getString(if (on) R.string.overlay_data_on else R.string.overlay_data_off)
        if (on) startColorAnimation(tvState) else stopColorAnimation(tvState)
        v.findViewById<TextView>(R.id.tv_usage).text =
            getString(R.string.overlay_traffic_fmt, DataMonitor.formatCompact(Prefs.getCumulativeUsage(this)))
    }

    /** 数据开启时：状态文字不断变色（醒目提醒）。 */
    private fun startColorAnimation(tv: TextView) {
        if (colorAnimator != null) return
        val colors = intArrayOf(
            0xFFFF0000.toInt(), 0xFFFF6600.toInt(), 0xFFFFFF00.toInt(),
            0xFF00FF00.toInt(), 0xFF00FFFF.toInt(), 0xFF0000FF.toInt(), 0xFFFF00FF.toInt()
        )
        val anim = ValueAnimator.ofArgb(*colors).apply {
            duration = 2000
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            addUpdateListener { tv.setTextColor(it.animatedValue as Int) }
        }
        anim.start()
        colorAnimator = anim
    }

    private fun stopColorAnimation(tv: TextView) {
        colorAnimator?.cancel()
        colorAnimator = null
        tv.setTextColor(0xFFCCCCCC.toInt())
    }

    // ---------- 监听网络变化 ----------
    private fun registerConnectivity() {
        try {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val req = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_CELLULAR)
                .build()
            val cb = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) { handler.post { refreshOverlay() } }
                override fun onLost(network: Network) { handler.post { refreshOverlay() } }
                override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) { handler.post { refreshOverlay() } }
            }
            cm.registerNetworkCallback(req, cb)
            connectivityCallback = cb
        } catch (_: Exception) {}
    }

    private fun unregisterConnectivity() {
        connectivityCallback?.let {
            try {
                val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
                cm.unregisterNetworkCallback(it)
            } catch (_: Exception) {}
        }
        connectivityCallback = null
    }

    private fun registerPhoneState() {
        try {
            val tm = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
            val listener = object : PhoneStateListener() {
                override fun onServiceStateChanged(state: ServiceState?) {
                    ensureKeepAlive()
                    handler.post { refreshOverlay() }
                }

                override fun onDataActivity(direction: Int) {
                    ensureKeepAlive()
                }

                override fun onDataConnectionStateChanged(state: Int, reason: Int) {
                    ensureKeepAlive()
                    handler.post { refreshOverlay() }
                }
            }
            tm.listen(listener,
                PhoneStateListener.LISTEN_SERVICE_STATE or
                    PhoneStateListener.LISTEN_DATA_ACTIVITY or
                    PhoneStateListener.LISTEN_DATA_CONNECTION_STATE)
            phoneStateListener = listener
        } catch (_: Exception) {}
    }

    private fun unregisterPhoneState() {
        phoneStateListener?.let {
            try {
                val tm = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
                tm.listen(it, PhoneStateListener.LISTEN_NONE)
            } catch (_: Exception) {}
        }
        phoneStateListener = null
    }
}
