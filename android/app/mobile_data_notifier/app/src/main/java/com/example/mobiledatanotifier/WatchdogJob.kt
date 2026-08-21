package com.example.mobiledatanotifier

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context

/**
 * 第二道兜底守护：与 KeepAliveJob 互为"双保险"。
 * 当 OverlayService 被杀后，KeepAliveJob 可能因进程消亡一并丢失调度，
 * 此时 WatchdogJob 独立存在，负责再次拉起服务并重建所有保活任务。
 */
class WatchdogJob : JobService() {

    companion object {
        private const val JOB_ID = 7002
        private const val JOB_ID_ONESHOT = 7003
        private const val INTERVAL_MS = 15 * 60 * 1000L

        fun schedule(ctx: Context) {
            val js = ctx.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler
            val job = JobInfo.Builder(JOB_ID, ComponentName(ctx, WatchdogJob::class.java))
                .setPeriodic(INTERVAL_MS)
                .setPersisted(true)
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_NONE)
                .build()
            js.schedule(job)
        }

        /** 划掉任务/进程死亡后的一次性短延迟拉起，不走 15 分钟周期。 */
        fun scheduleImmediate(ctx: Context, delayMs: Long = 1500L) {
            if (!Prefs.isServiceEnabled(ctx)) return
            val js = ctx.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler
            val job = JobInfo.Builder(JOB_ID_ONESHOT, ComponentName(ctx, WatchdogJob::class.java))
                .setMinimumLatency(delayMs)
                .setOverrideDeadline(delayMs + 2000L)
                .setPersisted(true)
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_NONE)
                .build()
            js.schedule(job)
        }
    }

    override fun onStartJob(params: JobParameters?): Boolean {
        if (!Prefs.isServiceEnabled(this)) return false
        if (!ProcessUtil.isOverlayAlive(this)) {
            try { OverlayService.start(this) } catch (_: Exception) {}
        }
        if (!ProcessUtil.isGuardAlive(this)) {
            try { GuardService.start(this) } catch (_: Exception) {}
        }
        // 确保另外两道保活也在运行
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        try { schedule(this) } catch (_: Exception) {}
        try { AlarmKeeper.register(this) } catch (_: Exception) {}
        try { AlarmKeeper.scheduleRolling(this) } catch (_: Exception) {}
        return false
    }

    override fun onStopJob(params: JobParameters?): Boolean = true
}
