package com.example.g3800duplex.transport

enum class PrintProtocol {
    CanonClss,
    ;

    val label: String
        get() = "私有协议 (CLSS/BJNP)"

    val discoverHint: String
        get() = "SNMP + BJNP 搜索中…"
}
