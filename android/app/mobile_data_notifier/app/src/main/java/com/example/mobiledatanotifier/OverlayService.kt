package com.example.mobiledatanotifier

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
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
import android.app.AlertDialog
import android.media.AudioManager
import android.util.Log
import android.view.ContextThemeWrapper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.SeekBar
import android.widget.TextView

/**
 * 保活前台服务 + 顶部悬浮窗。
 * - 前台服务通知走低优先级渠道（静音、不震动）。
 * - 悬浮窗显示移动数据开关状态、已用流量，含"关闭流量"按钮（打开 OPPO SIM 设置页）。
 * - START_STICKY：被杀后系统尝试重建。
 */
class OverlayService : Service() {

    companion object {
        const val CH_FG = "ch_fg"
        const val CH_REMINDER = "ch_reminder"
        const val ACTION_SHOW_OVERLAY = "com.example.mobiledatanotifier.SHOW_OVERLAY"
        const val ACTION_HIDE_OVERLAY = "com.example.mobiledatanotifier.HIDE_OVERLAY"
        const val ACTION_REFRESH_OVERLAY = "com.example.mobiledatanotifier.REFRESH_OVERLAY"
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

        fun startIfEnabled(ctx: Context) {
            if (!Prefs.isServiceEnabled(ctx)) return
            start(ctx)
        }

        /**
         * 打开 OPPO SIM/双卡设置（可关移动数据）。
         * 第三方通常无法直接启动 OplusSimSettingsActivity（resolve 成功但 start 会抛），
         * 因此优先返回系统公开的 NETWORK_OPERATOR_SETTINGS / GEMINI_MANAGEMENT。
         */
        fun createMobileDataSettingsIntent(ctx: Context): Intent {
            val pm = ctx.packageManager
            for (intent in simSettingsIntents(includePrivilegedExplicit = false)) {
                if (pm.resolveActivity(intent, 0) != null) return intent
            }
            return Intent(Settings.ACTION_DATA_USAGE_SETTINGS)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }

        /** ColorOS 设置首页进入 SIM 页时用的 identifier。 */
        private const val OPLUS_FROM_SETTINGS = "oplus.intent.category.START_FROM_SETTINGS_MAIN_PAGE"

        private fun simSettingsIntents(includePrivilegedExplicit: Boolean): List<Intent> {
            val flags = Intent.FLAG_ACTIVITY_NEW_TASK
            val list = mutableListOf<Intent>()
            if (includePrivilegedExplicit) {
                val explicit = Intent(Intent.ACTION_MAIN)
                    .setClassName(
                        "com.android.phone",
                        "com.android.simsettings.activity.OplusSimSettingsActivity"
                    )
                    .addCategory(Intent.CATEGORY_DEFAULT)
                    .addFlags(flags)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    explicit.identifier = OPLUS_FROM_SETTINGS
                }
                list.add(explicit)
            }
            list.add(
                Intent(Settings.ACTION_NETWORK_OPERATOR_SETTINGS)
                    .addCategory(Intent.CATEGORY_DEFAULT)
                    .addFlags(flags)
            )
            list.add(
                Intent("android.settings.MANAGE_ALL_SIM_PROFILES_SETTINGS")
                    .addCategory(Intent.CATEGORY_DEFAULT)
                    .addFlags(flags)
            )
            list.add(
                Intent("android.settings.GEMINI_MANAGEMENT")
                    .addCategory(Intent.CATEGORY_DEFAULT)
                    .addFlags(flags)
            )
            list.add(
                Intent("com.android.settings.MULTI_SIM_SETTINGS")
                    .addCategory(Intent.CATEGORY_DEFAULT)
                    .addFlags(flags)
            )
            return list
        }
    }

    private lateinit var wm: WindowManager
    private var overlayView: View? = null
    private lateinit var handler: Handler
    private var connectivityCallback: ConnectivityManager.NetworkCallback? = null
    private var phoneStateListener: PhoneStateListener? = null
    private var colorFlashing = false
    private var flashOnRed = true
    private val flashRunnable = object : Runnable {
        override fun run() {
            val tv = overlayView?.findViewById<TextView>(R.id.tv_state)
            if (tv == null) {
                colorFlashing = false
                return
            }
            flashOnRed = !flashOnRed
            tv.setTextColor(if (flashOnRed) 0xFFFF0000.toInt() else 0xFF00FF00.toInt())
            tv.invalidate()
            handler.postDelayed(this, 400L)
        }
    }
    // 动态注册的系统事件接收器（SCREEN_ON/OFF、USER_PRESENT、TIME_TICK 无法 Manifest 静态注册）
    private var systemEventReceiver: BroadcastReceiver? = null
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
        overlayHiddenByUser = Prefs.isOverlayHiddenByUser(this)
        if (!overlayHiddenByUser) addOverlay()
        registerSystemEvents()
        handler.post(updateRunnable)
        isRunning = true
        // 调度兜底心跳：被系统杀后由 JobScheduler 拉起
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_HIDE_OVERLAY -> {
                overlayHiddenByUser = true
                Prefs.setOverlayHiddenByUser(this, true)
                removeOverlay()
            }
            ACTION_SHOW_OVERLAY -> {
                overlayHiddenByUser = false
                Prefs.setOverlayHiddenByUser(this, false)
                addOverlay()
            }
            ACTION_REFRESH_OVERLAY -> handler.post { refreshOverlay() }
            ACTION_STOP -> {
                Prefs.setServiceEnabled(this, false)
                try { AlarmKeeper.cancel(this) } catch (_: Exception) {}
                shutdown()
                try { GuardService.stop(this) } catch (_: Exception) {}
                return START_NOT_STICKY
            }
        }
        // 无论以何种方式被拉起，都重新确保所有保活组件生效
        ensureKeepAlive()
        return START_STICKY
    }

    /** 确保所有保活组件已注册。失败时通过 AlarmKeeper 兜底。用户主动停止后不再拉起。 */
    private fun ensureKeepAlive() {
        if (!Prefs.isServiceEnabled(this)) return
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
        try { if (!ProcessUtil.isGuardAlive(this)) GuardService.start(this) } catch (_: Exception) {}
    }

    private fun shutdown() {
        unregisterSystemEvents()
        unregisterPhoneState()
        handler.removeCallbacks(updateRunnable)
        handler.removeCallbacks(flashRunnable)
        colorFlashing = false
        unregisterConnectivity()
        removeOverlay()
        stopForeground(STOP_FOREGROUND_REMOVE)
        isRunning = false
        stopSelf()
    }

    override fun onDestroy() {
        unregisterSystemEvents()
        unregisterPhoneState()
        handler.removeCallbacks(updateRunnable)
        handler.removeCallbacks(flashRunnable)
        colorFlashing = false
        unregisterConnectivity()
        removeOverlay()
        isRunning = false
        scheduleRestartIfEnabled()
        super.onDestroy()
    }

    /**
     * 任务被从最近列表划掉时：直接 start 自己会随进程一起消亡，无效。
     * 改为通过 AlarmManager 安排 1 秒后延迟重启（闹钟由系统持有），并调度一次性 Job。
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        scheduleRestartIfEnabled()
        try { if (Prefs.isServiceEnabled(this) && !ProcessUtil.isGuardAlive(this)) GuardService.start(this) } catch (_: Exception) {}
        super.onTaskRemoved(rootIntent)
    }

    private fun scheduleRestartIfEnabled() {
        if (!Prefs.isServiceEnabled(this)) return
        try { AlarmKeeper.scheduleRestart(this, 1000L) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        try { WatchdogJob.scheduleImmediate(this) } catch (_: Exception) {}
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
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
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
        // 关闭流量按钮已隐藏至长按菜单
        // v.findViewById<Button>(R.id.btn_close_data).setOnClickListener { openDataSettings() }
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
        handler.removeCallbacks(flashRunnable)
        colorFlashing = false
        overlayView?.let {
            try { wm.removeView(it) } catch (_: Exception) {}
        }
        overlayView = null
        isOverlayShown = false
    }

    private fun enableDrag(v: View, params: WindowManager.LayoutParams) {
        var initX = 0; var initY = 0; var touchX = 0f; var touchY = 0f; var moved = false
        val longPressRunnable = Runnable { showPopupMenu(v) }
        v.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> {
                    initX = params.x; initY = params.y
                    touchX = e.rawX; touchY = e.rawY; moved = false
                    // 启动长按计时器，未移动且满 4 秒则弹菜单
                    handler.removeCallbacks(longPressRunnable)
                    handler.postDelayed(longPressRunnable, 4000L)
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (e.rawX - touchX).toInt()
                    val dy = (e.rawY - touchY).toInt()
                    if (dx * dx + dy * dy > 25) {
                        moved = true
                        handler.removeCallbacks(longPressRunnable)
                    }
                    params.x = initX + dx
                    params.y = initY + dy
                    try { wm.updateViewLayout(v, params) } catch (_: Exception) {}
                }
                MotionEvent.ACTION_UP -> {
                    handler.removeCallbacks(longPressRunnable)
                    if (moved) return@setOnTouchListener true
                }
            }
            false
        }
    }

    /** 长按浮窗满 4 秒后弹出操作菜单。 */
    private fun showPopupMenu(anchor: View) {
        val idClose = 0x0001_0001
        val idVolume = 0x0001_0002
        val menu = PopupMenu(this, anchor)
        menu.menu.add(0, idClose, 0, getString(R.string.overlay_close_data))
        menu.menu.add(0, idVolume, 1, "调节音量")
        menu.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                idClose -> {
                    openDataSettings()
                    true
                }
                idVolume -> {
                    showVolumeDialog()
                    true
                }
                else -> false
            }
        }
        try { menu.show() } catch (_: Exception) {}
    }

    /** 弹出音量调节对话框（SeekBar 实时控制媒体音量）。 */
    private fun showVolumeDialog() {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val max = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val cur = am.getStreamVolume(AudioManager.STREAM_MUSIC)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(64, 24, 64, 24)
        }
        val seekBar = SeekBar(this).apply {
            this.max = if (max > 0) max else 1
            progress = cur.coerceIn(0, this.max)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        val tvCur = TextView(this).apply {
            text = "当前音量：$cur / $max"
            textSize = 14f
        }
        layout.addView(seekBar)
        layout.addView(tvCur)
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                if (fromUser) {
                    setMusicVolume(am, p)
                    tvCur.text = "当前音量：$p / $max"
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {
                // 兜底：部分系统仅在松手时才提交音量更改
                setMusicVolume(am, seekBar.progress)
            }
        })
        val builder = AlertDialog.Builder(ContextThemeWrapper(this, android.R.style.Theme_Material_Light_Dialog_Alert))
            .setTitle("调节音量")
            .setView(layout)
            .setNegativeButton("关闭", null)
            .setOnDismissListener { handler.post { refreshOverlay() } }
        val dialog = builder.create()
        try {
            dialog.window?.setType(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY)
        } catch (_: Exception) {}
        dialog.show()
    }

    /** 设置媒体音量，携带 FLAG_SHOW_UI 确保更改生效并回显系统音量面板。 */
    private fun setMusicVolume(am: AudioManager, vol: Int) {
        val safe = vol.coerceIn(0, am.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
        try {
            am.setStreamVolume(
                AudioManager.STREAM_MUSIC,
                safe,
                AudioManager.FLAG_SHOW_UI or AudioManager.FLAG_PLAY_SOUND
            )
        } catch (e: Exception) {
            Log.w("OverlayService", "setStreamVolume failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    private fun openDataSettings() {
        val fallback = Intent(Settings.ACTION_DATA_USAGE_SETTINGS)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        for (intent in simSettingsIntents(includePrivilegedExplicit = true) + fallback) {
            try {
                startActivity(intent)
                return
            } catch (_: Exception) {
            }
        }
    }

    private val updateRunnable = object : Runnable {
        override fun run() {
            // 自愈：开机后/被杀重建后若悬浮窗未挂载且未被用户主动隐藏，则重新添加
            if (overlayView == null && !overlayHiddenByUser && PermUtil.canDrawOverlays(this@OverlayService)) {
                addOverlay()
            }
            refreshOverlay()
            try { AlarmKeeper.scheduleRolling(this@OverlayService) } catch (_: Exception) {}
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
        val scope = Prefs.getTrafficScope(this)
        v.findViewById<TextView>(R.id.tv_usage).text =
            getString(
                R.string.overlay_traffic_fmt,
                scope.overlayPrefix,
                DataMonitor.formatCompact(Prefs.getCumulativeUsage(this))
            )
    }

    /** 数据开启时：状态文字红绿硬切闪烁。用 Handler 定时切色，不受系统动画缩放影响。 */
    private fun startColorAnimation(tv: TextView) {
        if (colorFlashing) return
        colorFlashing = true
        flashOnRed = true
        tv.setTextColor(0xFFFF0000.toInt())
        tv.invalidate()
        handler.postDelayed(flashRunnable, 400L)
    }

    private fun stopColorAnimation(tv: TextView) {
        handler.removeCallbacks(flashRunnable)
        colorFlashing = false
        tv.setTextColor(0xFFCCCCCC.toInt())
    }

    // ---------- 动态注册系统事件（Android 8.0+ 这些广播无法 Manifest 静态接收） ----------
    private fun registerSystemEvents() {
        if (systemEventReceiver != null) return
        val r = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                when (intent?.action) {
                    Intent.ACTION_SCREEN_OFF -> ensureKeepAlive()
                    Intent.ACTION_SCREEN_ON,
                    Intent.ACTION_USER_PRESENT,
                    Intent.ACTION_TIME_TICK -> {
                        ensureKeepAlive()
                        handler.post { refreshOverlay() }
                    }
                }
            }
        }
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
            addAction(Intent.ACTION_TIME_TICK)
        }
        try {
            registerReceiver(r, filter)
            systemEventReceiver = r
        } catch (_: Exception) {}
    }

    private fun unregisterSystemEvents() {
        systemEventReceiver?.let {
            try { unregisterReceiver(it) } catch (_: Exception) {}
        }
        systemEventReceiver = null
    }

    // ---------- 监听网络变化 ----------
    private fun registerConnectivity() {
        try {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val req = NetworkRequest.Builder().build()
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
                    if (direction != TelephonyManager.DATA_ACTIVITY_NONE) {
                        handler.post { refreshOverlay() }
                    }
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
