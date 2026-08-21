package com.example.mobiledatanotifier

import android.app.ActivityManager
import android.content.Context

/**
 * 跨进程探活。GuardService 跑在 `:guard` 后，内存里的 isRunning 对另一进程无效。
 * getRunningServices 在 Android 8+ 仍会返回本应用（含其它进程）的服务。
 */
object ProcessUtil {

    fun isOverlayAlive(ctx: Context): Boolean =
        isServiceRunning(ctx, OverlayService::class.java.name)

    fun isGuardAlive(ctx: Context): Boolean =
        isServiceRunning(ctx, GuardService::class.java.name) ||
            isProcessRunning(ctx, "${ctx.packageName}:guard")

    private fun isProcessRunning(ctx: Context, processName: String): Boolean {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return try {
            am.runningAppProcesses?.any { it.processName == processName } == true
        } catch (_: Exception) {
            false
        }
    }

    @Suppress("DEPRECATION")
    private fun isServiceRunning(ctx: Context, className: String): Boolean {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return try {
            am.getRunningServices(Integer.MAX_VALUE)
                ?.any { it.service.className == className } == true
        } catch (_: Exception) {
            false
        }
    }
}
