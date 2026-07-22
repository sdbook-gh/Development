package com.example.g3800duplex.canon

import i7.C1673a
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import jp.co.canon.bsd.ad.sdk.core.printer.b as IjPrinter
import jp.co.canon.bsd.ad.sdk.core.search.SnmpSearch
import m2.AbstractC1862a
import m2.b as SearchPrinter

enum class DiscoverySource {
    Snmp,
    Bjnp,
    Both,
}

/**
 * Official IJ LAN discovery: SNMP + BJNP in parallel (SearchPrinterUseCase IJ mode).
 */
class IjParallelSearch(
    private val broadcastAddress: String,
) {
    fun search(timeoutMs: Long = 10_000L): List<DiscoveredPrinter> {
        val byKey = ConcurrentHashMap<String, DiscoveredPrinter>()
        val remaining = CountDownLatch(2)
        val snmpCode = AtomicInteger(-1)
        val bjnpCode = AtomicInteger(-1)

        val snmp = SnmpSearch(broadcastAddress)
        val bjnp = C1673a(broadcastAddress)

        fun merge(printer: AbstractC1862a, source: DiscoverySource) {
            val ij = printer as? IjPrinter
            val model = printer.modelName ?: "Canon"
            val ip = printer.ipAddress ?: return
            val mac = printer.macAddress.orEmpty()
            val key = mac.ifBlank { ip }.uppercase()
            byKey.compute(key) { _, existing ->
                if (existing == null) {
                    DiscoveredPrinter(
                        name = printer.nickname ?: model,
                        model = model,
                        ipAddress = ip,
                        macAddress = mac,
                        deviceId = ij?.deviceId.orEmpty(),
                        serial = ij?.productSerialnumber.orEmpty(),
                        source = source,
                        raw = printer,
                    )
                } else {
                    val mergedSource =
                        when {
                            existing.source == source -> existing.source
                            existing.source == DiscoverySource.Both -> DiscoverySource.Both
                            else -> DiscoverySource.Both
                        }
                    existing.copy(
                        deviceId = existing.deviceId.ifBlank { ij?.deviceId.orEmpty() },
                        serial = existing.serial.ifBlank { ij?.productSerialnumber.orEmpty() },
                        source = mergedSource,
                        raw = existing.raw ?: printer,
                    )
                }
            }
        }

        val snmpStarted = snmp.startSearch(
            object : SearchPrinter.a {
                override fun a(resultCode: Int) {
                    snmpCode.set(resultCode)
                    remaining.countDown()
                }

                override fun b(printer: AbstractC1862a) {
                    merge(printer, DiscoverySource.Snmp)
                }
            },
        )
        if (snmpStarted != 0) {
            remaining.countDown()
        }

        val bjnpStarted = bjnp.startSearch(
            object : SearchPrinter.a {
                override fun a(resultCode: Int) {
                    bjnpCode.set(resultCode)
                    remaining.countDown()
                }

                override fun b(printer: AbstractC1862a) {
                    merge(printer, DiscoverySource.Bjnp)
                }
            },
        )
        if (bjnpStarted != 0) {
            remaining.countDown()
        }

        val waitMs = timeoutMs.coerceAtLeast(3_000L)
        remaining.await(waitMs, TimeUnit.MILLISECONDS)
        snmp.stopSearch()
        bjnp.stopSearch()
        // Brief drain so late callbacks can land after stop.
        remaining.await(500, TimeUnit.MILLISECONDS)

        return byKey.values
            .sortedBy { it.model }
            .toList()
    }
}
