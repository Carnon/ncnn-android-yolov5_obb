package com.izl.yolov5obb;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class Yolov5Obb {

    static {
        System.loadLibrary("yolov5obb");
    }

    public static native boolean Init(AssetManager mgr);

    public static native BoxObb[] Detect(Bitmap bitmap, boolean useGPU);

}
