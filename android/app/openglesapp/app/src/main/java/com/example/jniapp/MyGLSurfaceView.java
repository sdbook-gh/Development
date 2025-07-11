package com.example.jniapp;

import android.content.Context;
import android.content.res.AssetManager;
import android.opengl.GLSurfaceView;

public class MyGLSurfaceView extends GLSurfaceView {
    private final MyRenderer renderer;

    public MyGLSurfaceView(Context context) {
        super(context);
        setEGLContextClientVersion(2);
        renderer = new MyRenderer();
        setRenderer(renderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }

    public void setAssetManager(AssetManager assetManager) {
        renderer.setAssetManager(assetManager);
    }
}
