//
// Created by carnon on 2023/3/7.
//

#ifndef NCNN_ANDROID_YOLOV5_OBB_YOLOV5OBB_H
#define NCNN_ANDROID_YOLOV5_OBB_YOLOV5OBB_H

#include "net.h"
#include "opencv2/imgproc.hpp"


typedef struct BoxObb {
    float x0;
    float y0;
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float score;
    int label;
} BoxObb;


typedef struct BoxRotated{
    cv::RotatedRect rotatedRect;
    float score;
    int label;
} BoxRotated;


class YoloV5Obb {
public:
    YoloV5Obb(AAssetManager *mgr, const char* param, const char* bin);

    ~YoloV5Obb();

    std::vector<BoxObb> detect(JNIEnv *env, jobject image, jboolean useGpu);
    std::vector<std::string> labels{"plane", "baseball-diamond", "bridge", "ground-track-field",
                                    "small-vehicle", "large-vehicle", "ship", "tennis-court",
                                    "basketball-court", "storage-tank", "soccer-ball-field",
                                    "roundabout", "harbor", "swimming-pool", "helicopter"};

private:
    ncnn::Net *Net;
    int input_w = 864;
    int input_h = 864;
    int num_class = 15;


public:
    static YoloV5Obb *detector;
    static bool hasGPU;
    static bool toUseGPU;
};

#endif //NCNN_ANDROID_YOLOV5_OBB_YOLOV5OBB_H
