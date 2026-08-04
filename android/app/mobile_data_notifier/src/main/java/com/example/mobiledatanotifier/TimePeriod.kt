package com.example.mobiledatanotifier

import org.json.JSONArray
import org.json.JSONObject

/**
 * 定时时段数据模型。
 * 例如 23:00 关闭、07:00 开启（跨天）。
 */
data class TimePeriod(
    val id: Long,
    var startHour: Int,
    var startMin: Int,
    var endHour: Int,
    var endMin: Int,
    var enabled: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("sh", startHour); put("sm", startMin)
        put("eh", endHour); put("em", endMin)
        put("en", enabled)
    }

    /** 是否跨天（如 23:00 -> 07:00） */
    val crossesMidnight: Boolean
        get() = (endHour < startHour) || (endHour == startHour && endMin <= startMin)

    fun startLabel(): String = String.format("%02d:%02d", startHour, startMin)
    fun endLabel(): String = String.format("%02d:%02d", endHour, endMin)

    companion object {
        fun fromJson(o: JSONObject): TimePeriod = TimePeriod(
            id = o.getLong("id"),
            startHour = o.getInt("sh"), startMin = o.getInt("sm"),
            endHour = o.getInt("eh"), endMin = o.getInt("em"),
            enabled = o.optBoolean("en", true)
        )

        fun fromJsonArray(arr: JSONArray): MutableList<TimePeriod> {
            val list = mutableListOf<TimePeriod>()
            for (i in 0 until arr.length()) {
                list.add(fromJson(arr.getJSONObject(i)))
            }
            return list
        }
    }
}
