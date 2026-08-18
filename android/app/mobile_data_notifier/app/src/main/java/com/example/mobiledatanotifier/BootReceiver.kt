package com.example.mobiledatanotifier

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** 开机自启：启动保活服务并重新排程定时提醒。 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_LOCKED_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            "android.intent.action.QUICKBOOT_POWERON" -> {
                try { OverlayService.start(context) } catch (_: Exception) {}
                try { GuardService.start(context) } catch (_: Exception) {}
                try { KeepAliveJob.schedule(context) } catch (_: Exception) {}
                try { WatchdogJob.schedule(context) } catch (_: Exception) {}
                try { AlarmKeeper.register(context) } catch (_: Exception) {}
                try { ScheduleManager.rescheduleAll(context) } catch (_: Exception) {}
            }
        }
    }
}
