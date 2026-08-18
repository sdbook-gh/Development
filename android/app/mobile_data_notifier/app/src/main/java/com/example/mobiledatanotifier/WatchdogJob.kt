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
    }

    override fun onStartJob(params: JobParameters?): Boolean {
        if (!OverlayService.isRunning) {
            try { OverlayService.start(this) } catch (_: Exception) {}
        }
        // 确保另外两道保活也在运行
        try { KeepAliveJob.schedule(this) } catch (_: Exception) {}
        return false
    }

    override fun onStopJob(params: JobParameters?): Boolean = true
}
