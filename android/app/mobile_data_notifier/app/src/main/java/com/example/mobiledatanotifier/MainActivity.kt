package com.example.mobiledatanotifier

import android.Manifest
import android.app.Activity
import android.app.AlarmManager
import android.app.TimePickerDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.app.AlertDialog
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 主界面：权限授予、状态展示、悬浮窗/保活控制、定时时段管理。
 * 仅使用 Android 框架 API（不引入 AndroidX，避免 Maven 依赖）。
 */
class MainActivity : Activity() {

    private lateinit var tvDataState: TextView
    private lateinit var rgTrafficScope: RadioGroup
    private lateinit var tvUsageBoot: TextView
    private lateinit var tvUsageCumulative: TextView
    private lateinit var btnEditCumulative: Button
    private lateinit var btnResetCumulative: Button
    private lateinit var btnResetUsage: Button
    private lateinit var btnStartService: Button
    private lateinit var btnStopService: Button
    private lateinit var btnShowOverlay: Button
    private lateinit var btnHideOverlay: Button
    private lateinit var btnGrantOverlay: Button
    private lateinit var btnBatteryWhitelist: Button
    private lateinit var btnAutostart: Button
    private lateinit var btnBackground: Button
    private lateinit var btnBgPopup: Button
    private lateinit var tvBatteryStatus: TextView
    private lateinit var tvUsageAccess: TextView
    // ---- 定时关闭部分已隐藏 ----
    // private lateinit var tvNoPeriods: TextView
    // private lateinit var periodsContainer: LinearLayout
    // private lateinit var btnAddPeriod: Button
    // private var periods: MutableList<TimePeriod> = mutableListOf()
    // ---- 定时关闭部分已隐藏 ----
    private val handler = Handler(Looper.getMainLooper())
    private val refreshRunnable = object : Runnable {
        override fun run() { refreshStatus(); handler.postDelayed(this, 3000) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvDataState = findViewById(R.id.tv_data_state)
        rgTrafficScope = findViewById(R.id.rg_traffic_scope)
        tvUsageBoot = findViewById(R.id.tv_usage_boot)
        tvUsageCumulative = findViewById(R.id.tv_usage_cumulative)
        btnEditCumulative = findViewById(R.id.btn_edit_cumulative)
        btnResetCumulative = findViewById(R.id.btn_reset_cumulative)
        btnResetUsage = findViewById(R.id.btn_reset_usage)
        btnStartService = findViewById(R.id.btn_start_service)
        btnStopService = findViewById(R.id.btn_stop_service)
        btnShowOverlay = findViewById(R.id.btn_show_overlay)
        btnHideOverlay = findViewById(R.id.btn_hide_overlay)
        btnGrantOverlay = findViewById(R.id.btn_grant_overlay)
        btnBatteryWhitelist = findViewById(R.id.btn_battery_whitelist)
        btnAutostart = findViewById(R.id.btn_autostart)
        btnBackground = findViewById(R.id.btn_background)
        btnBgPopup = findViewById(R.id.btn_bg_popup)
        tvBatteryStatus = findViewById(R.id.tv_battery_status)
        tvUsageAccess = findViewById(R.id.tv_usage_access)
        tvUsageAccess.setOnClickListener {
            PermUtil.openUsageAccessSettings(this)
            toast(getString(R.string.usage_access_open_toast))
        }
        // ---- 定时关闭部分已隐藏 ----
        // tvNoPeriods = findViewById(R.id.tv_no_periods)
        // periodsContainer = findViewById(R.id.periods_container)
        // btnAddPeriod = findViewById(R.id.btn_add_period)
        // ---- 定时关闭部分已隐藏 ----

        btnEditCumulative.setOnClickListener {
            showEditCumulativeDialog()
        }
        btnResetCumulative.setOnClickListener {
            Prefs.resetCumulativeUsage(this)
            refreshStatus()
            toast("累计流量已重置")
        }
        btnResetUsage.setOnClickListener {
            val cur = DataMonitor.bytesSinceBoot(this)
            if (cur >= 0L) Prefs.setUsageBaseline(this, cur)
            refreshStatus()
            toast("已重置流量统计")
        }
        bindTrafficScopeRadios()
        btnStartService.setOnClickListener {
            Prefs.setServiceEnabled(this, true)
            try { OverlayService.start(this); toast("已启动保活服务") } catch (e: Exception) { toast("启动失败：$e") }
            try { GuardService.start(this) } catch (_: Exception) {}
        }
        btnStopService.setOnClickListener {
            Prefs.setServiceEnabled(this, false)
            try { AlarmKeeper.cancel(this) } catch (_: Exception) {}
            sendServiceAction(OverlayService.ACTION_STOP)
            GuardService.stop(this)
            toast("已停止服务")
        }
        btnShowOverlay.setOnClickListener {
            Prefs.setServiceEnabled(this, true)
            Prefs.setOverlayHiddenByUser(this, false)
            OverlayService.start(this)
            sendServiceAction(OverlayService.ACTION_SHOW_OVERLAY)
        }
        btnHideOverlay.setOnClickListener {
            Prefs.setOverlayHiddenByUser(this, true)
            sendServiceAction(OverlayService.ACTION_HIDE_OVERLAY)
        }
        btnGrantOverlay.setOnClickListener {
            PermUtil.openOverlaySettings(this)
        }
        btnBatteryWhitelist.setOnClickListener {
            PermUtil.requestIgnoreBatteryOptimizations(this)
        }
        btnAutostart.setOnClickListener {
            PermUtil.openAutoStartSettings(this)
        }
        btnBackground.setOnClickListener {
            PermUtil.openBackgroundRunSettings(this)
        }
        btnBgPopup.setOnClickListener {
            PermUtil.openBackgroundPopupSettings(this)
        }
        // ---- 定时关闭部分已隐藏 ----
        // btnAddPeriod.setOnClickListener {
        //     val p = TimePeriod(Prefs.nextPeriodId(this), 23, 0, 7, 0, enabled = true)
        //     periods.add(p)
        //     Prefs.savePeriods(this, periods)
        //     renderPeriods()
        //     ScheduleManager.rescheduleAll(this)
        //     toast("已添加时段")
        // }
        // periods = Prefs.getPeriods(this)
        // renderPeriods()
        // ---- 定时关闭部分已隐藏 ----
        requestRuntimePermissions()
        Prefs.setServiceEnabled(this, true)
        // 调度兜底保活心跳（首次打开即注册，设备重启后依然生效）
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
        try { if (!ProcessUtil.isGuardAlive(this)) GuardService.start(this) } catch (_: Exception) {}
    }

    private fun sendServiceAction(action: String) {
        val i = Intent(this, OverlayService::class.java).setAction(action)
        try { startService(i) } catch (_: Exception) {}
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
        refreshPermStatus()
        // 打开 App 界面时按需激活：保活服务未运行则启动；悬浮窗未显示则显示
        val justStarted = !OverlayService.isRunning
        if (justStarted) {
            Prefs.setServiceEnabled(this, true)
            try { OverlayService.start(this) } catch (_: Exception) {}
            try { GuardService.start(this) } catch (_: Exception) {}
            toast("服务已自动恢复")
        }
        // 服务已在运行但悬浮窗未显示，且用户未主动隐藏、具备权限时，补发显示指令
        if (!justStarted && OverlayService.isRunning &&
            !OverlayService.isOverlayShown &&
            !Prefs.isOverlayHiddenByUser(this) &&
            PermUtil.canDrawOverlays(this)) {
            sendServiceAction(OverlayService.ACTION_SHOW_OVERLAY)
        }
        // 无论如何都重新注册所有保活组件，确保无遗漏
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { WatchdogJob.schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        try { ScheduleManager.rescheduleAll(this) } catch (_: Exception) {}
        handler.post(refreshRunnable)
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacks(refreshRunnable)
    }

    // ---------- 状态刷新 ----------
    private fun refreshStatus() {
        val on = try { DataMonitor.isMobileDataEnabled(this) } catch (_: Exception) { false }
        tvDataState.text = getString(
            if (on) R.string.data_state_on else R.string.data_state_off
        )
        tvDataState.setTextColor(
            getColor(if (on) R.color.danger else R.color.accent)
        )
        DataMonitor.autoUpdateCumulative(this)
        val scope = Prefs.getTrafficScope(this)
        val boot = DataMonitor.bytesSinceBoot(this, scope).coerceAtLeast(0L)
        tvUsageBoot.text = getString(
            R.string.usage_since_boot_fmt,
            scopeLabel(scope),
            DataMonitor.formatBytes(boot)
        )
        tvUsageCumulative.text = getString(
            R.string.usage_cumulative_fmt,
            DataMonitor.formatBytes(Prefs.getCumulativeUsage(this))
        )
    }

    private fun bindTrafficScopeRadios() {
        val checkedId = when (Prefs.getTrafficScope(this)) {
            TrafficScope.WIFI -> R.id.rb_scope_wifi
            TrafficScope.ALL -> R.id.rb_scope_all
            TrafficScope.MOBILE -> R.id.rb_scope_mobile
        }
        rgTrafficScope.check(checkedId)
        rgTrafficScope.setOnCheckedChangeListener { _, id ->
            val scope = when (id) {
                R.id.rb_scope_wifi -> TrafficScope.WIFI
                R.id.rb_scope_all -> TrafficScope.ALL
                else -> TrafficScope.MOBILE
            }
            DataMonitor.switchTrafficScope(this, scope)
            refreshStatus()
            sendServiceAction(OverlayService.ACTION_REFRESH_OVERLAY)
        }
    }

    private fun scopeLabel(scope: TrafficScope): String = when (scope) {
        TrafficScope.MOBILE -> getString(R.string.traffic_scope_mobile)
        TrafficScope.WIFI -> getString(R.string.traffic_scope_wifi)
        TrafficScope.ALL -> getString(R.string.traffic_scope_all)
    }

    private fun refreshPermStatus() {
        btnGrantOverlay.visibility =
            if (PermUtil.canDrawOverlays(this)) View.GONE else View.VISIBLE
        val batteryOk = PermUtil.isIgnoringBatteryOptimizations(this)
        tvBatteryStatus.text = "电池优化白名单：" + if (batteryOk) "已加入" else "未加入"
        val alarmOk = PermUtil.canScheduleExactAlarms(this)
        if (!alarmOk) {
            tvBatteryStatus.append("  |  精确闹钟：未授权")
            tvBatteryStatus.setOnClickListener { PermUtil.openExactAlarmSettings(this) }
        }
        // 使用情况访问权限状态（AppOps GET_USAGE_STATS）
        val usageAccessOk = PermUtil.hasUsageAccess(this)
        tvUsageAccess.text = getString(
            if (usageAccessOk) R.string.usage_access_granted else R.string.usage_access_denied
        )
        tvUsageAccess.setTextColor(
            getColor(if (usageAccessOk) R.color.accent else R.color.danger)
        )
    }

    // ---------- 权限 ----------
    private fun requestRuntimePermissions() {
        val need = mutableListOf<String>()
        if (checkSelfPermission(Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.READ_PHONE_STATE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (need.isNotEmpty()) {
            requestPermissions(need.toTypedArray(), 100)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        refreshStatus()
        if (grantResults.isNotEmpty() && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            toast("部分权限未授予，相关功能可能受限")
        }
    }

    // ---------- 时段列表（已隐藏）----------
    // private fun renderPeriods() {
    //     periodsContainer.removeAllViews()
    //     tvNoPeriods.visibility = if (periods.isEmpty()) View.VISIBLE else View.GONE
    //     for (p in periods) {
    //         addPeriodRow(p)
    //     }
    // }
    // private fun addPeriodRow(p: TimePeriod) {
    //     val row = LayoutInflater.from(this).inflate(R.layout.item_time_period, periodsContainer, false)
    //     val tvStart = row.findViewById<TextView>(R.id.tv_start)
    //     val tvEnd = row.findViewById<TextView>(R.id.tv_end)
    //     val sw = row.findViewById<Switch>(R.id.sw_enabled)
    //     val btnDel = row.findViewById<Button>(R.id.btn_delete)
    //     tvStart.text = "开始 " + p.startLabel()
    //     tvEnd.text = "结束 " + p.endLabel()
    //     sw.isChecked = p.enabled
    //     tvStart.setOnClickListener {
    //         show24hTimePicker(p.startHour, p.startMin) { h, m ->
    //             p.startHour = h; p.startMin = m
    //             tvStart.text = "开始 " + p.startLabel()
    //             Prefs.savePeriods(this, periods)
    //             ScheduleManager.rescheduleAll(this)
    //         }
    //     }
    //     tvEnd.setOnClickListener {
    //         show24hTimePicker(p.endHour, p.endMin) { h, m ->
    //             p.endHour = h; p.endMin = m
    //             tvEnd.text = "结束 " + p.endLabel()
    //             Prefs.savePeriods(this, periods)
    //             ScheduleManager.rescheduleAll(this)
    //         }
    //     }
    //     sw.setOnCheckedChangeListener { _, isChecked ->
    //         p.enabled = isChecked
    //         Prefs.savePeriods(this, periods)
    //         ScheduleManager.rescheduleAll(this)
    //     }
    //     btnDel.setOnClickListener {
    //         periods.remove(p)
    //         Prefs.savePeriods(this, periods)
    //         renderPeriods()
    //         ScheduleManager.rescheduleAll(this)
    //     }
    //     periodsContainer.addView(row)
    // }
    // /** 24 小时制时间选择器（is24HourView=true 强制 24h）。 */
    // private fun show24hTimePicker(hour: Int, minute: Int, onSet: (Int, Int) -> Unit) {
    //     TimePickerDialog(this, { _, h, m -> onSet(h, m) }, hour, minute, true).show()
    // }

    private fun showEditCumulativeDialog() {
        val current = DataMonitor.formatBytes(Prefs.getCumulativeUsage(this))
        val input = EditText(this).apply {
            hint = getString(R.string.edit_cumulative_hint)
            setRawInputType(android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL or
                android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS)
            setText(current)
            setSelection(text.length)
        }
        AlertDialog.Builder(this)
            .setTitle("修改累计流量")
            .setMessage("当前累计：$current\n支持输入：纯数字=字节，或带 m/M、k/K、g/G（支持小数）")
            .setView(input)
            .setPositiveButton("确认") { _, _ ->
                val text = input.text.toString().trim()
                val bytes = DataMonitor.parseCumulativeInput(text)
                if (bytes < 0) {
                    toast("输入不合法，请检查单位")
                } else {
                    Prefs.setCumulativeUsage(this, bytes)
                    refreshStatus()
                    toast("累计流量已更新：" + DataMonitor.formatBytes(bytes))
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }
}
