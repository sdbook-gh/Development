package com.example.g3800duplex.cloud

import android.content.Context
import android.os.Build
import java.io.File
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.TimeUnit
import jp.co.canon.oip.android.opal.mobileatp.ATPCAMSConnectSetting
import jp.co.canon.oip.android.opal.mobileatp.ATPMobileATP
import jp.co.canon.oip.android.opal.mobileatp.deviceregistration.ATPDeviceRegistrationRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

/**
 * Region + ATP register/token for Canon CNPS document convert.
 *
 * ATP registration [applicationId] must be the Canon PRINT id that CNPS allowlists
 * (`jp.co.canon.bsd.ad.pixmaprint`). Android [Context.getPackageName] stays this app's id;
 * only the CAMS client field is aligned with official DeviceRegistAsyncTask.
 */
object CanonAtpAuth {

    private const val SCOPE = "oip.prt.AppPrint"
    private const val REALM = "/oip/prt.AppPrint"
    private const val SELECT_PS = "https://rs.ciggws.net/select_ps/print"
    private const val PREFS = "canon_atp"
    /** Bumped when registration identity changes (forces credential wipe + re-register). */
    private const val KEY_REGISTERED = "device_registered_v2"

    /** Official Canon PRINT applicationId used by W5.c / DeviceRegistAsyncTask. */
    private const val ATP_APPLICATION_ID = "jp.co.canon.bsd.ad.pixmaprint"
    private const val CLIENT_NAME = "PIXUS Print"
    private const val CLIENT_DESCRIPTION =
        "This is a free application for Canon inkjet printers or multi-function inkjet printers to easily print images, PDF files, etc."

    data class RegionEndpoints(
        val psCode: Int,
        val prtBaseUrl: String,
        val registrationServer: String,
        val tokenServer: String,
    )

    data class AuthSession(
        val accessToken: String,
        val encryptionKey: String,
        val region: RegionEndpoints,
        val userAgent: String,
    )

    suspend fun ensureAccessToken(context: Context): AuthSession = withContext(Dispatchers.IO) {
        val app = context.applicationContext
        CloudLog.i(
            "atp",
            "ensureAccessToken package=${app.packageName} atpAppId=$ATP_APPLICATION_ID",
        )
        val region = resolveRegion(userAgent(app))
        val ua = userAgent(app)
        CloudLog.i(
            "atp",
            "region ps=${region.psCode} reg=${region.registrationServer} prt=${region.prtBaseUrl}",
        )
        val cams = ATPCAMSConnectSetting(
            3,
            1000L,
            10_000,
            10_000,
            ua,
            region.registrationServer,
            region.tokenServer,
        )
        val atp = ATPMobileATP(cams)
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        // Drop v1 prefs / credentials registered under this app's package name (CNPS 19001).
        if (prefs.contains("device_registered_v1")) {
            CloudLog.i("atp", "migrating away from device_registered_v1 credentials")
            clearAtpCredentials(app)
            prefs.edit().remove("device_registered_v1").apply()
        }
        if (!prefs.getBoolean(KEY_REGISTERED, false)) {
            clearAtpCredentials(app)
            val req = ATPDeviceRegistrationRequest().apply {
                applicationId = ATP_APPLICATION_ID
                clientName = CLIENT_NAME
                clientDescription = CLIENT_DESCRIPTION
                scopes = arrayOf(SCOPE)
                defaultScopes = arrayOf(SCOPE)
                realm = REALM
            }
            CloudLog.i("atp", "registerDevice applicationId=$ATP_APPLICATION_ID realm=$REALM")
            val reg = atp.registerDevice(req, app, null)
            if (reg.resultCode != 0) {
                throw CloudConvertException(
                    atpFailureMessage(
                        "ATP 设备注册失败",
                        reg.resultCode,
                        reg.httpStatusCode,
                        reg.errorCode,
                    ),
                    stage = "atp-register",
                )
            }
            prefs.edit().putBoolean(KEY_REGISTERED, true).apply()
            CloudLog.i("atp", "registerDevice OK")
        }

        // registerDevice teardown clears the CAMS singleton (empty tokenServer → MalformedURLException).
        atp.setCAMSConnectSetting(cams)
        val tokenResult = atp.getAccessToken(arrayOf(SCOPE, null), app, REALM, null)
        if (tokenResult.resultCode != 0) {
            prefs.edit().putBoolean(KEY_REGISTERED, false).apply()
            clearAtpCredentials(app)
            throw CloudConvertException(
                atpFailureMessage(
                    "ATP 获取 access_token 失败",
                    tokenResult.resultCode,
                    tokenResult.httpStatusCode,
                    tokenResult.errorCode,
                ),
                stage = "atp-token",
            )
        }
        val token = tokenResult.accessToken
            ?: throw CloudConvertException("ATP 返回空 access_token", stage = "atp-token")
        CloudLog.i("atp", "access_token OK len=${token.length}")
        AuthSession(
            accessToken = token,
            encryptionKey = encryptionKeyFromToken(token),
            region = region,
            userAgent = ua,
        )
    }

    fun encryptionKeyFromToken(accessToken: String): String {
        return if (accessToken.length > 16) {
            accessToken.substring(accessToken.length - 16)
        } else {
            accessToken.padStart(16, '0')
        }
    }

    fun resolveRegion(userAgent: String): RegionEndpoints {
        var lastError: Exception? = null
        repeat(4) { attempt ->
            try {
                val conn = (URL(SELECT_PS).openConnection() as HttpURLConnection).apply {
                    requestMethod = "HEAD"
                    connectTimeout = 10_000
                    readTimeout = 10_000
                    setRequestProperty("User-Agent", userAgent)
                    instanceFollowRedirects = true
                }
                try {
                    conn.connect()
                    val http = conn.responseCode
                    CloudLog.i("select_ps", "HEAD attempt=$attempt HTTP $http ps_code=${conn.getHeaderField("ps_code")}")
                    if (http == 200) {
                        val ps = conn.getHeaderField("ps_code")?.trim().orEmpty()
                        if (ps.isEmpty()) {
                            throw CloudConvertException(
                                "select_ps 未返回 ps_code",
                                stage = "select_ps",
                            )
                        }
                        if (ps == "maintenance" || ps == "out_of_svr_service" || ps == "out_of_app_service") {
                            throw CloudConvertException(
                                "佳能云文档转换服务不可用: $ps",
                                stage = "select_ps",
                            )
                        }
                        val code = ps.toInt()
                        return endpointsForPsCode(code)
                    }
                } finally {
                    conn.disconnect()
                }
            } catch (e: CloudConvertException) {
                throw e
            } catch (e: Exception) {
                CloudLog.w("select_ps", "attempt=$attempt failed: ${e.javaClass.simpleName}: ${e.message}", e)
                lastError = e
            }
            if (attempt < 3) {
                TimeUnit.MILLISECONDS.sleep(1000)
            }
        }
        throw CloudConvertException(
            "区域探测失败: ${lastError?.javaClass?.simpleName}: ${lastError?.message ?: "unknown"}",
            lastError,
            stage = "select_ps",
        )
    }

    fun endpointsForPsCode(psCode: Int): RegionEndpoints = when (psCode) {
        3 -> RegionEndpoints(
            psCode = 3,
            prtBaseUrl = "https://prt-uw2.srv.ygles.com",
            registrationServer = "https://ccb-uw2.srv.ygles.com",
            tokenServer = "https://ccb-uw2.srv.ygles.com",
        )
        7 -> RegionEndpoints(
            psCode = 7,
            prtBaseUrl = "https://prt-cn1w.ugw2.canon.com.cn",
            registrationServer = "https://ccb-cn1w.ugw2.canon.com.cn",
            tokenServer = "https://ccb-cn1w.ugw2.canon.com.cn",
        )
        else -> RegionEndpoints(
            psCode = psCode,
            prtBaseUrl = "https://prt-ec1.srv.ygles.com",
            registrationServer = "https://ccb-ec1.srv.ygles.com",
            tokenServer = "https://ccb-ec1.srv.ygles.com",
        )
    }

    fun userAgent(context: Context): String {
        val locale = try {
            context.resources.configuration.locales[0]
        } catch (_: Throwable) {
            Locale.getDefault()
        }
        val tz = TimeZone.getDefault().id
        val ver = try {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "0.1"
        } catch (_: Throwable) {
            "0.1"
        }
        return "PIXUS Print/$ver(Android ${Build.VERSION.RELEASE};$locale)$tz"
    }

    /** Wipe ATP on-disk client/device credentials (filesDir/mobileATP/...). */
    fun clearAtpCredentials(context: Context) {
        val root = File(context.applicationContext.filesDir, "mobileATP")
        if (!root.exists()) return
        root.walkBottomUp().forEach { f ->
            if (!f.delete() && f.exists()) {
                CloudLog.w("atp", "failed to delete ${f.absolutePath}")
            }
        }
        CloudLog.i("atp", "cleared mobileATP credentials under ${root.absolutePath}")
    }

    private fun atpFailureMessage(
        prefix: String,
        resultCode: Int,
        httpStatus: Int,
        errorCode: String?,
    ): String {
        val hint = when {
            resultCode == 805 -> "（连接超时；检查外网与区域端点）"
            resultCode == 2 && httpStatus == 401 ->
                "（CAMS Digest 鉴权失败或服务端拒绝；请确认外网可达佳能云）"
            errorCode?.contains("invalid_client", ignoreCase = true) == true
                || errorCode?.contains("unauthorized", ignoreCase = true) == true ->
                "（CAMS 拒绝客户端；检查 ATP applicationId）"
            else -> ""
        }
        return "$prefix result=$resultCode http=$httpStatus error=${errorCode.orEmpty()}$hint"
    }
}
