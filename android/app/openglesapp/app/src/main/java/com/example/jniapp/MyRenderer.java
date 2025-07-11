package com.example.jniapp;

import android.content.res.AssetManager;
import android.opengl.GLSurfaceView;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

public class MyRenderer implements GLSurfaceView.Renderer {
    private AssetManager assetManager;

    public void setAssetManager(AssetManager am) {
        assetManager = am;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        nativeOnSurfaceCreated(assetManager);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        nativeOnSurfaceChanged(width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        nativeOnDrawFrame();
    }

    // JNI 接口声明
    private native void nativeOnSurfaceCreated(AssetManager assetManager);
    private native void nativeOnSurfaceChanged(int width, int height);
    private native void nativeOnDrawFrame();
}
