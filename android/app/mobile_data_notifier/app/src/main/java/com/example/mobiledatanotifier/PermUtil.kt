package com.example.mobiledatanotifier

import android.app.AlarmManager
import android.app.AppOpsManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.provider.Settings

/** 各类特殊权限的检测与跳转。 */
object PermUtil {

    fun canDrawOverlays(ctx: Context): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) Settings.canDrawOverlays(ctx) else true

    fun openOverlaySettings(ctx: Context) {
        val i = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + ctx.packageName))
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(i)
    }

    fun isIgnoringBatteryOptimizations(ctx: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return true
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(ctx.packageName)
    }

    fun requestIgnoreBatteryOptimizations(ctx: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return
        try {
            val i = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
            i.data = Uri.parse("package:" + ctx.packageName)
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            ctx.startActivity(i)
        } catch (e: Exception) {
            try {
                val i = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                ctx.startActivity(i)
            } catch (_: Exception) {
            }
        }
    }

    fun canScheduleExactAlarms(ctx: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return true
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        return am.canScheduleExactAlarms()
    }

    fun openExactAlarmSettings(ctx: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return
        val i = Intent(Settings.ACTION_REQUEST_SCHEDULE_EXACT_ALARM)
        i.data = Uri.parse("package:" + ctx.packageName)
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(i)
    }

    /** ColorOS / OPPO 自启动管理。打不开则回退到应用详情。 */
    fun openAutoStartSettings(ctx: Context) {
        val pkg = ctx.packageName
        val intents = listOf(
            component("com.oplus.safecenter", "com.oplus.safecenter.startupapp.StartupAppListActivity"),
            component("com.coloros.safecenter", "com.coloros.safecenter.startupapp.StartupAppListActivity"),
            component("com.coloros.safecenter", "com.coloros.safecenter.permission.startup.StartupAppListActivity"),
            component("com.coloros.safecenter", "com.coloros.safecenter.startupapp.view.StartupAppListActivity"),
            component("com.oplusos.startup", "com.oplusos.startup.ui.StartupAppListActivity"),
            component("com.coloros.phonemanager", "com.coloros.phonemanager.startupapp.StartupAppListActivity"),
            Intent("android.settings.APP_STARTUP_SETTINGS").putExtra("packageName", pkg)
        )
        if (!tryStart(ctx, intents)) openAppDetails(ctx)
    }

    /** ColorOS 后台运行 / 耗电管理。 */
    fun openBackgroundRunSettings(ctx: Context) {
        val pkg = ctx.packageName
        val intents = listOf(
            component("com.oplus.battery", "com.oplus.powermanager.fuelui.PowerControlActivity")
                .putExtra("packageName", pkg),
            component("com.coloros.oppoguardelf", "com.coloros.powermanager.fuelui.PowerUsageModelActivity")
                .putExtra("packageName", pkg),
            component("com.oplus.safecenter", "com.oplus.safecenter.permission.PermissionAppAllActivity"),
            component("com.coloros.safecenter", "com.coloros.privacypermissionsentry.PermissionTopActivity")
        )
        if (!tryStart(ctx, intents)) openAppDetails(ctx)
    }

    /** ColorOS 后台弹出界面。被杀后重新挂悬浮窗常需要此权限。 */
    fun openBackgroundPopupSettings(ctx: Context) {
        val pkg = ctx.packageName
        val intents = listOf(
            component("com.oplus.safecenter", "com.oplus.safecenter.permission.PermissionTopActivity"),
            component("com.coloros.safecenter", "com.coloros.safecenter.permission.PermissionTopActivity"),
            component("com.oplus.safecenter", "com.oplus.safecenter.permission.floatwindow.FloatWindowListActivity"),
            Intent("oplus.intent.action.PERMISSION_APP_DETAIL").putExtra("packageName", pkg),
            Intent("com.coloros.safecenter.permission.PermissionTopActivity")
        )
        if (!tryStart(ctx, intents)) openAppDetails(ctx)
    }

    private fun component(pkg: String, cls: String): Intent =
        Intent().setComponent(ComponentName(pkg, cls))

    private fun tryStart(ctx: Context, intents: List<Intent>): Boolean {
        for (raw in intents) {
            try {
                raw.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                if (raw.resolveActivity(ctx.packageManager) != null) {
                    ctx.startActivity(raw)
                    return true
                }
            } catch (_: Exception) {
            }
        }
        return false
    }

    /** 是否已授予使用情况访问（AppOps GET_USAGE_STATS）。 */
    fun hasUsageAccess(ctx: Context): Boolean {
        return try {
            val appOps = ctx.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
            val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                appOps.unsafeCheckOpNoThrow(
                    AppOpsManager.OPSTR_GET_USAGE_STATS,
                    Process.myUid(),
                    ctx.packageName
                )
            } else {
                @Suppress("DEPRECATION")
                appOps.checkOpNoThrow(
                    AppOpsManager.OPSTR_GET_USAGE_STATS,
                    Process.myUid(),
                    ctx.packageName
                )
            }
            mode == AppOpsManager.MODE_ALLOWED
        } catch (_: Exception) {
            false
        }
    }

    /**
     * 打开使用情况访问设置。优先带包名跳转；ColorOS 再试隐私/权限页；
     * 最后回退通用列表或应用详情。
     */
    fun openUsageAccessSettings(ctx: Context) {
        val pkg = ctx.packageName
        val withPkg = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS).apply {
            data = Uri.parse("package:$pkg")
            putExtra(Settings.EXTRA_APP_PACKAGE, pkg)
            putExtra(Intent.EXTRA_PACKAGE_NAME, pkg)
            putExtra("packageName", pkg)
            putExtra("package", pkg)
        }
        val intents = listOf(
            withPkg,
            component("com.android.settings", "com.android.settings.Settings\$UsageAccessSettingsActivity")
                .putExtra(Settings.EXTRA_APP_PACKAGE, pkg)
                .putExtra("packageName", pkg),
            component("com.oplus.safecenter", "com.oplus.safecenter.permission.PermissionTopActivity"),
            component("com.coloros.safecenter", "com.coloros.safecenter.permission.PermissionTopActivity"),
            component("com.coloros.safecenter", "com.coloros.privacypermissionsentry.PermissionTopActivity"),
            component("com.oplus.safecenter", "com.oplus.privacypermissionsentry.PermissionTopActivity"),
            Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        )
        if (!tryStart(ctx, intents)) openAppDetails(ctx)
    }

    fun openAppDetails(ctx: Context) {
        val i = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
        i.data = Uri.parse("package:" + ctx.packageName)
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(i)
    }
}
