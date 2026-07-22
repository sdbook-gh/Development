package com.example.g3800duplex

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.g3800duplex.canon.CanonSnmpSdkBridge
import com.example.g3800duplex.ui.DuplexPrintScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Activity context required for PrintManager.print (system / Canon Print Service).
        val bridge = CanonSnmpSdkBridge(applicationContext)
        setContent {
            MaterialTheme {
                Surface {
                    DuplexPrintScreen(
                        activity = this@MainActivity,
                        bridge = bridge,
                    )
                }
            }
        }
    }
}
