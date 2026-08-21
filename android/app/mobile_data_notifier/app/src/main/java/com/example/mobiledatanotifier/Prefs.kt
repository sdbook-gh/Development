package com.example.mobiledatanotifier

import android.content.Context
import android.content.SharedPreferences
import android.os.SystemClock
import org.json.JSONArray

/** SharedPreferences 封装：定时时段、流量统计基线、保活/悬浮窗开关。 */
object Prefs {
    private const val NAME = "mdn_prefs"
    private const val KEY_PERIODS = "periods"
    private const val KEY_BASELINE = "usage_baseline"
    private const val KEY_BASELINE_TIME = "usage_baseline_time"
    private const val KEY_BASELINE_ELAPSED = "usage_baseline_elapsed"
    private const val KEY_TRAFFIC_SCOPE = "traffic_scope"
    private const val KEY_NEXT_PERIOD_ID = "next_period_id"
    private const val KEY_CUMULATIVE_USAGE = "cumulative_usage"
    private const val KEY_SERVICE_ENABLED = "service_enabled"
    private const val KEY_OVERLAY_HIDDEN = "overlay_hidden_by_user"

    @Suppress("DEPRECATION")
    private fun sp(ctx: Context): SharedPreferences =
        ctx.applicationContext.getSharedPreferences(NAME, Context.MODE_MULTI_PROCESS)

    fun getPeriods(ctx: Context): MutableList<TimePeriod> {
        val raw = sp(ctx).getString(KEY_PERIODS, null) ?: return mutableListOf()
        return try {
            TimePeriod.fromJsonArray(JSONArray(raw))
        } catch (e: Exception) {
            mutableListOf()
        }
    }

    fun savePeriods(ctx: Context, periods: List<TimePeriod>) {
        val arr = JSONArray()
        periods.forEach { arr.put(it.toJson()) }
        sp(ctx).edit().putString(KEY_PERIODS, arr.toString()).apply()
    }

    fun nextPeriodId(ctx: Context): Long {
        val cur = sp(ctx).getLong(KEY_NEXT_PERIOD_ID, 1L)
        sp(ctx).edit().putLong(KEY_NEXT_PERIOD_ID, cur + 1).apply()
        return cur
    }

    fun setUsageBaseline(ctx: Context, bytes: Long) {
        sp(ctx).edit()
            .putLong(KEY_BASELINE, bytes)
            .putLong(KEY_BASELINE_TIME, System.currentTimeMillis())
            .putLong(KEY_BASELINE_ELAPSED, SystemClock.elapsedRealtime())
            .apply()
    }

    fun getUsageBaseline(ctx: Context): Long = sp(ctx).getLong(KEY_BASELINE, 0L)

    fun getUsageBaselineTime(ctx: Context): Long = sp(ctx).getLong(KEY_BASELINE_TIME, 0L)

    fun getUsageBaselineElapsed(ctx: Context): Long = sp(ctx).getLong(KEY_BASELINE_ELAPSED, 0L)

    fun getTrafficScope(ctx: Context): TrafficScope =
        TrafficScope.fromPref(sp(ctx).getString(KEY_TRAFFIC_SCOPE, TrafficScope.MOBILE.prefValue))

    fun setTrafficScope(ctx: Context, scope: TrafficScope) {
        sp(ctx).edit().putString(KEY_TRAFFIC_SCOPE, scope.prefValue).apply()
    }

    fun getCumulativeUsage(ctx: Context): Long = sp(ctx).getLong(KEY_CUMULATIVE_USAGE, 0L)

    fun setCumulativeUsage(ctx: Context, bytes: Long) {
        sp(ctx).edit().putLong(KEY_CUMULATIVE_USAGE, bytes).apply()
    }

    fun resetCumulativeUsage(ctx: Context) {
        sp(ctx).edit().remove(KEY_CUMULATIVE_USAGE).apply()
    }

    /** 用户是否要保活服务。划掉/进程死亡后的重启路径必须尊重此标记。commit 以免进程马上被杀时丢失。 */
    fun isServiceEnabled(ctx: Context): Boolean =
        sp(ctx).getBoolean(KEY_SERVICE_ENABLED, false)

    fun setServiceEnabled(ctx: Context, enabled: Boolean) {
        sp(ctx).edit().putBoolean(KEY_SERVICE_ENABLED, enabled).commit()
    }

    /** 用户是否主动隐藏了悬浮窗。服务重建后按此决定是否重新挂载。 */
    fun isOverlayHiddenByUser(ctx: Context): Boolean =
        sp(ctx).getBoolean(KEY_OVERLAY_HIDDEN, false)

    fun setOverlayHiddenByUser(ctx: Context, hidden: Boolean) {
        sp(ctx).edit().putBoolean(KEY_OVERLAY_HIDDEN, hidden).commit()
    }
}
