//
// Created by carnon on 2023/3/7.
//
#include <jni.h>
#include <android/asset_manager_jni.h>
#include "gpu.h"
#include "yolov5obb.h"

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
    ncnn::create_gpu_instance();
    if(ncnn::get_gpu_count()>0){
        YoloV5Obb::hasGPU = true;
    }
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    ncnn::destroy_gpu_instance();
    delete YoloV5Obb::detector;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_izl_yolov5obb_Yolov5Obb_Init(JNIEnv *env, jclass thiz, jobject assetManager) {
    if(YoloV5Obb::detector != nullptr){
        delete YoloV5Obb::detector;
        YoloV5Obb::detector = nullptr;
    }
    if(YoloV5Obb::detector == nullptr){
        AAssetManager* mgr =   AAssetManager_fromJava(env, assetManager);
        YoloV5Obb::detector = new YoloV5Obb(mgr, "yolov5obb.param", "yolov5obb.bin");
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_izl_yolov5obb_Yolov5Obb_Detect(JNIEnv *env, jclass clazz, jobject bitmap,
                                        jboolean use_gpu) {
    std::vector<BoxObb> result = YoloV5Obb::detector->detect(env, bitmap, use_gpu);
    auto box_cls = env->FindClass("com/izl/yolov5obb/BoxObb");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFFFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i=0;
    for(auto &box: result){
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid,box.x0, box.y0, box.x1, box.y1, box.x2, box.y2, box.x3, box.y3, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;

}

