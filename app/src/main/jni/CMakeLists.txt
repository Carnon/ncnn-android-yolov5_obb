project(yolov5obb)

cmake_minimum_required(VERSION 3.10)


set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)


aux_source_directory(. SRC_LIST)

add_library(yolov5obb SHARED ${SRC_LIST})

target_link_libraries(yolov5obb ncnn ${OpenCV_LIBS})
