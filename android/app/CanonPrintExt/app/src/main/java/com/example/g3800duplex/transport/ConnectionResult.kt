package com.example.g3800duplex.transport

/**
 * Result of prepare / discover / probe shown in the UI.
 * Failures must carry a concrete [message] (stage + reason).
 */
data class ConnectionResult(
    val ok: Boolean,
    val protocol: PrintProtocol,
    val stage: String,
    val endpoint: String = "",
    val message: String,
    val latencyMs: Long = 0L,
    val cause: Throwable? = null,
) {
    fun summaryLine(): String {
        val status = if (ok) "成功" else "失败"
        val ep = if (endpoint.isNotBlank()) " · $endpoint" else ""
        val ms = if (latencyMs > 0) " · ${latencyMs}ms" else ""
        return "[$status] ${protocol.label} · $stage$ep$ms\n$message"
    }

    companion object {
        fun ok(
            protocol: PrintProtocol,
            stage: String,
            message: String,
            endpoint: String = "",
            latencyMs: Long = 0L,
        ) = ConnectionResult(true, protocol, stage, endpoint, message, latencyMs)

        fun fail(
            protocol: PrintProtocol,
            stage: String,
            message: String,
            endpoint: String = "",
            latencyMs: Long = 0L,
            cause: Throwable? = null,
        ) = ConnectionResult(false, protocol, stage, endpoint, message, latencyMs, cause)
    }
}
