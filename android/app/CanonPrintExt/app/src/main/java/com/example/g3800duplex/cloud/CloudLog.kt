package com.example.g3800duplex.cloud

import android.util.Log

/**
 * Word / CNPS cloud convert logging. Filter logcat by tag [TAG].
 */
object CloudLog {
    const val TAG = "G3800CloudConvert"

    fun i(stage: String, msg: String) {
        Log.i(TAG, "[$stage] $msg")
    }

    fun w(stage: String, msg: String, t: Throwable? = null) {
        if (t != null) Log.w(TAG, "[$stage] $msg", t) else Log.w(TAG, "[$stage] $msg")
    }

    fun e(stage: String, msg: String, t: Throwable? = null) {
        if (t != null) Log.e(TAG, "[$stage] $msg", t) else Log.e(TAG, "[$stage] $msg")
        t?.let { logCauseChain(stage, it) }
    }

    fun http(stage: String, method: String, url: String, code: Int, body: String?) {
        val snippet = body?.replace("\n", " ")?.take(800).orEmpty()
        val line = "$method $url → HTTP $code" +
            if (snippet.isNotBlank()) " body=$snippet" else ""
        if (code in 200..299) {
            Log.i(TAG, "[$stage] $line")
        } else {
            Log.e(TAG, "[$stage] $line")
        }
    }

    fun formatChain(t: Throwable?): String {
        if (t == null) return ""
        val sb = StringBuilder()
        var cur: Throwable? = t
        var depth = 0
        while (cur != null && depth < 8) {
            if (depth > 0) sb.append(" ← ")
            sb.append(cur.javaClass.simpleName).append(": ").append(cur.message ?: "(no message)")
            cur = cur.cause
            depth++
        }
        return sb.toString()
    }

    private fun logCauseChain(stage: String, t: Throwable) {
        var cur: Throwable? = t.cause
        var depth = 1
        while (cur != null && depth < 8) {
            Log.e(
                TAG,
                "[$stage] cause[$depth] ${cur.javaClass.name}: ${cur.message}",
                cur,
            )
            cur = cur.cause
            depth++
        }
    }
}
