package com.example.g3800duplex.canon

import android.content.Context
import android.net.ConnectivityManager
import android.net.LinkProperties
import android.net.wifi.WifiManager
import android.os.Build
import java.net.Inet4Address
import java.net.NetworkInterface

/**
 * Resolves IPv4 broadcast address for SNMP discovery (same idea as Canon Util.k).
 */
object BroadcastAddress {
    fun resolve(context: Context): String {
        tryWifiBroadcast(context)?.let { return it }
        tryLinkProperties(context)?.let { return it }
        tryInterfaces()?.let { return it }
        return "255.255.255.255"
    }

    private fun tryWifiBroadcast(context: Context): String? {
        return try {
            @Suppress("DEPRECATION")
            val wm = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            @Suppress("DEPRECATION")
            val dhcp = wm.dhcpInfo ?: return null
            val ip = dhcp.ipAddress
            val mask = dhcp.netmask
            if (ip == 0) return null
            val broadcast = (ip and mask) or mask.inv()
            String.format(
                "%d.%d.%d.%d",
                broadcast and 0xff,
                broadcast shr 8 and 0xff,
                broadcast shr 16 and 0xff,
                broadcast shr 24 and 0xff,
            )
        } catch (_: Throwable) {
            null
        }
    }

    private fun tryLinkProperties(context: Context): String? {
        if (Build.VERSION.SDK_INT < 23) return null
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = cm.activeNetwork ?: return null
            val props: LinkProperties = cm.getLinkProperties(network) ?: return null
            for (link in props.linkAddresses) {
                val addr = link.address
                if (addr is Inet4Address && !addr.isLoopbackAddress) {
                    val iface = NetworkInterface.getByInetAddress(addr) ?: continue
                    for (ia in iface.interfaceAddresses) {
                        val b = ia.broadcast?.hostAddress
                        if (!b.isNullOrBlank()) return b
                    }
                }
            }
            null
        } catch (_: Throwable) {
            null
        }
    }

    private fun tryInterfaces(): String? {
        return try {
            val ifaces = NetworkInterface.getNetworkInterfaces() ?: return null
            while (ifaces.hasMoreElements()) {
                val nif = ifaces.nextElement()
                if (!nif.isUp || nif.isLoopback) continue
                for (ia in nif.interfaceAddresses) {
                    val addr = ia.address
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        val b = ia.broadcast?.hostAddress
                        if (!b.isNullOrBlank()) return b
                    }
                }
            }
            null
        } catch (_: Throwable) {
            null
        }
    }
}
