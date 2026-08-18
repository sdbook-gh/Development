package com.example.mobiledatanotifier

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray

/** SharedPreferences 封装：定时时段、流量统计基线、提醒铃声开关。 */
object Prefs {
    private const val NAME = "mdn_prefs"
    private const val KEY_PERIODS = "periods"
    private const val KEY_BASELINE = "usage_baseline"
    private const val KEY_BASELINE_TIME = "usage_baseline_time"
    private const val KEY_NEXT_PERIOD_ID = "next_period_id"
    private const val KEY_CUMULATIVE_USAGE = "cumulative_usage"

    private fun sp(ctx: Context): SharedPreferences =
        ctx.applicationContext.getSharedPreferences(NAME, Context.MODE_PRIVATE)

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
            .apply()
    }

    fun getUsageBaseline(ctx: Context): Long = sp(ctx).getLong(KEY_BASELINE, 0L)

    fun getUsageBaselineTime(ctx: Context): Long = sp(ctx).getLong(KEY_BASELINE_TIME, 0L)

    fun getCumulativeUsage(ctx: Context): Long = sp(ctx).getLong(KEY_CUMULATIVE_USAGE, 0L)

    fun setCumulativeUsage(ctx: Context, bytes: Long) {
        sp(ctx).edit().putLong(KEY_CUMULATIVE_USAGE, bytes).apply()
    }

    fun resetCumulativeUsage(ctx: Context) {
        sp(ctx).edit().remove(KEY_CUMULATIVE_USAGE).apply()
    }
}
