package com.example.jniapp;

import androidx.appcompat.app.AppCompatActivity;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import java.util.Map;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        System.loadLibrary("test_libusb");
    }

    public native String stringFromJNI();
    public static native boolean openusb(int fd, int vid, int pid);
    public static native void closeusb();

    private Context ctx;
    private UsbManager usbManager;
    private static final String ACTION_USB_PERMISSION = "com.example.jniapp.USB_PERMISSION";
    private BroadcastReceiver permissionReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            synchronized (this) {
                Toast.makeText(ctx, "requestPermission result", Toast.LENGTH_SHORT).show();
                UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
                boolean granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false);
                if (granted) {
                    testUSB(device);
                } else {
                    Toast.makeText(ctx, "requestPermission result: denied", Toast.LENGTH_LONG).show();
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ctx = getApplicationContext();
        usbManager = (UsbManager) ctx.getSystemService(Context.USB_SERVICE);
        IntentFilter filter = new IntentFilter(ACTION_USB_PERMISSION);
        ctx.registerReceiver(permissionReceiver, filter);

        setContentView(R.layout.activity_main);
        TextView tv = findViewById(R.id.onlyYou);
        tv.setText("jni app");
        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "test USB", Snackbar.LENGTH_LONG).setAction("Action", null).show();

                int vid = 0x2dbb;
                int pid = 0x0503;
                Map<String, UsbDevice> deviceList = usbManager.getDeviceList();
                for (UsbDevice dev : deviceList.values()) {
                    if (dev.getVendorId() == vid && dev.getProductId() == pid) {
                        if (usbManager.hasPermission(dev)) {
                            Toast.makeText(ctx, "hasPermission", Toast.LENGTH_LONG).show();
                            testUSB(dev);
                        } else {
                            Toast.makeText(ctx, "requestPermission", Toast.LENGTH_SHORT).show();
                            Intent intent = new Intent(ACTION_USB_PERMISSION);
                            PendingIntent pi = PendingIntent.getBroadcast(ctx, 0, intent, PendingIntent.FLAG_IMMUTABLE);
                            usbManager.requestPermission(dev, pi);
                        }
                    }
                }
                Log.e("testusb", String.format("No device with VID=0x%04X PID=0x%04X", vid, pid));
            }
        });
    }

    private void testUSB(UsbDevice usbDevice) {
        Log.i("testusb", "设备名称: " + usbDevice.getDeviceName() + "\n" +
                "厂商ID: " + usbDevice.getVendorId() + "\n" +
                "产品ID: " + usbDevice.getProductId());
        int fd = usbManager.openDevice(usbDevice).getFileDescriptor();
        if (openusb(fd, usbDevice.getVendorId(), usbDevice.getProductId())) {
            Toast.makeText(ctx, "open usb success", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(ctx, "open usb failed", Toast.LENGTH_LONG).show();
        }
        closeusb();
    }
}
