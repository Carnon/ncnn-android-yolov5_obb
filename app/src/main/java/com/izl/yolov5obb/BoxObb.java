package com.izl.yolov5obb;

import android.graphics.Color;

import java.util.Random;

public class BoxObb {

    public float x0,y0,x1,y1,x2,y2,x3,y3;
    private int label;
    private float score;
    private static String[] labels={"plane", "baseball-diamond", "bridge", "ground-track-field",
            "small-vehicle", "large-vehicle", "ship", "tennis-court",
            "basketball-court", "storage-tank", "soccer-ball-field",
            "roundabout", "harbor", "swimming-pool", "helicopter"};

    public BoxObb(float x0,float y0, float x1, float y1, float x2, float y2, float x3, float y3, int label, float score){
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.x3 = x3;
        this.y3 = y3;
        this.label = label;
        this.score = score;
    }

    public String getLabel(){
        return labels[label];
    }

    public float getScore(){
        return score;
    }

    public int getColor(){
        Random random = new Random(label);
        return Color.argb(255,random.nextInt(256),random.nextInt(256),random.nextInt(256));
    }

}
