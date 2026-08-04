package com.example.mobiledatanotifier

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context

/**
 * 兜底保活：周期性检查 OverlayService 是否存活，若被杀则重新拉起。
 * - 基于 JobScheduler 周期任务，最小间隔 15 分钟（系统限制）。
 * - setPersisted(true)：设备重启后任务依然生效。
 * - Doze 维护窗口也会触发，作为"被系统杀后"的兜底拉起。
 */
class KeepAliveJob : JobService() {

    companion object {
        private const val JOB_ID = 7001
        private const val INTERVAL_MS = 15 * 60 * 1000L  // 15 分钟

        fun schedule(ctx: Context) {
            val js = ctx.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler
            val job = JobInfo.Builder(JOB_ID, ComponentName(ctx, KeepAliveJob::class.java))
                .setPeriodic(INTERVAL_MS)
                .setPersisted(true)
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_NONE)
                .build()
            js.schedule(job)
        }
    }

    override fun onStartJob(params: JobParameters?): Boolean {
        if (!OverlayService.isRunning) {
            try { OverlayService.start(this) } catch (_: Exception) {}
        }
        return false  // 任务已同步完成，无需后台线程
    }

    override fun onStopJob(params: JobParameters?): Boolean = true  // 被系统中断则重新调度
}
