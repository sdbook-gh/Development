package com.example.g3800duplex.transport

import java.net.InetSocketAddress
import java.net.Socket

object TcpProbe {
    data class Result(val ok: Boolean, val message: String, val latencyMs: Long)

    fun connect(host: String, port: Int, timeoutMs: Int = 3_000): Result {
        val start = System.currentTimeMillis()
        return try {
            Socket().use { sock ->
                sock.connect(InetSocketAddress(host, port), timeoutMs)
                val ms = System.currentTimeMillis() - start
                Result(true, "TCP $host:$port 连通", ms)
            }
        } catch (t: Throwable) {
            val ms = System.currentTimeMillis() - start
            Result(
                false,
                "TCP $host:$port 失败: ${t.javaClass.simpleName}: ${t.message}",
                ms,
            )
        }
    }
}
