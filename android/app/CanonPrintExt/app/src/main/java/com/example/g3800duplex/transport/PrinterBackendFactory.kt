package com.example.g3800duplex.transport

import android.content.Context
import com.example.g3800duplex.canon.CanonSdkBridge

object PrinterBackendFactory {
    fun create(
        context: Context,
        canonBridge: CanonSdkBridge,
        protocol: PrintProtocol = PrintProtocol.CanonClss,
    ): PrinterBackend {
        require(protocol == PrintProtocol.CanonClss) { "only CanonClss is supported" }
        return CanonClssBackend(context, canonBridge)
    }
}
