package com.example.g3800duplex.cloud

import android.content.Context

/** Matches official Canon PRINT pref key for cloud convert ToS. */
object DocConvertAcceptance {
    private const val PREFS = "docconvert"
    const val KEY = "docconvert.accepted.v230"

    fun isAccepted(context: Context): Boolean =
        context.applicationContext
            .getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY, false)

    fun setAccepted(context: Context, accepted: Boolean = true) {
        context.applicationContext
            .getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY, accepted)
            .apply()
    }
}
