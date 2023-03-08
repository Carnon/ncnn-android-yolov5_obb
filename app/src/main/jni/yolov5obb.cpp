#include "yolov5obb.h"


bool YoloV5Obb::hasGPU = true;
bool YoloV5Obb::toUseGPU = true;
YoloV5Obb *YoloV5Obb::detector = nullptr;


YoloV5Obb::YoloV5Obb(AAssetManager *mgr, const char *param, const char *bin) {
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU;

    Net = new ncnn::Net();
    // opt 需要在加载前设置
    Net->opt.use_vulkan_compute = toUseGPU;  // gpu
    Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}


YoloV5Obb::~YoloV5Obb() {
    Net->clear();
    delete Net;
}


void generate_boxes(const float* output, std::vector<BoxRotated> &bbox_collection, int num_classes, int input_w, int input_h, int dw_, int dh_, float scale) {
    int strides[3] = {8, 16, 32};
    int anchors[3][6][2] = {{{10,  13}, {16,  30},  {33,  23}},
                            {{30,  61}, {62,  45},  {59,  119}},
                            {{116, 90}, {156, 198}, {373, 326}}};
    std::vector<std::vector<int>> grids = {
            {3, int(input_w/strides[0]), int(input_h/strides[0])},
            {3, int(input_w/strides[1]), int(input_h/strides[1])},
            {3, int(input_w/strides[2]), int(input_h/strides[2])}
    };

    std::vector<cv::RotatedRect> rotatedRects;

    bbox_collection.clear();

    int position = 0;
    for(int n=0; n < (int)grids.size(); n++){
        for(int c=0; c <grids[n][0]; c++){
            int* anchor = anchors[n][c];
            for(int h=0; h<grids[n][1]; h++){
                for(int w=0; w<grids[n][2]; w++){
                    const float *row = output + (position * (num_classes + 5 + 180));
                    position ++;
                    float prob = row[4];
                    if(prob < 0.5)
                        continue;

                    BoxRotated boxRotated;
                    float cx = float((row[0] * 2 - 0.5 + (float)w)*(float)strides[n] - (float)dw_) / scale;
                    float cy = float((row[1] * 2 - 0.5 + (float)h)*(float)strides[n] - (float)dh_) / scale;
                    float w_ = (float)pow(row[2]*2, 2) * (float)anchor[0] / scale;
                    float h_ = (float)pow(row[3]*2, 2) * (float)anchor[1] / scale;

                    auto max_label_pos = std::max_element(row+5, row+5+num_classes);
                    auto max_angle_pos = std::max_element(row+5+num_classes, row+5+num_classes+180);
                    boxRotated.score = prob * row[max_label_pos - row];
                    boxRotated.label = (int)(max_label_pos - row) -5;
                    int angle_id = (int)(max_angle_pos - row) - 20;
                    boxRotated.rotatedRect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(w_, h_), float(90-angle_id));

                    bbox_collection.push_back(boxRotated);
                }
            }
        }
    }
}


float IOUCalculate(BoxRotated &box_a, BoxRotated &box_b) {
    std::vector<cv::Point2f> intersectingRegion;
    rotatedRectangleIntersection(box_a.rotatedRect, box_b.rotatedRect, intersectingRegion);
    // 对轮廓进行顺时针排序
    cv::Point2f center = {0.f, 0.f};
    for (auto &pt: intersectingRegion) {
        center.x += pt.x;
        center.y += pt.y;
    }
    center.x /= intersectingRegion.size();
    center.y /= intersectingRegion.size();
    std::sort(intersectingRegion.begin(), intersectingRegion.end(), [&](cv::Point2f &a, cv::Point2f &b) {
        auto a_offset = a - center;
        auto b_offset = b - center;
        return cv::fastAtan2(a_offset.y, a_offset.x) < cv::fastAtan2(b_offset.y, b_offset.x);
    });

    if (intersectingRegion.empty())
        return 0.f;
    float inter = (float) cv::contourArea(intersectingRegion);
    if (inter < 0.1)
        return 0.f;
    float iou = inter / (box_a.rotatedRect.size.area() + box_b.rotatedRect.size.area() - inter);
    return iou;
}


void nms_angle(std::vector<BoxRotated> &input_boxes) {
    sort(input_boxes.begin(), input_boxes.end(), [=](BoxRotated a, BoxRotated b) { return a.score > b.score; });
    for (int i = 0; i < int(input_boxes.size()); i++) {
        for (int j = i + 1; j < int(input_boxes.size()); j++) {
            if (input_boxes[i].label == input_boxes[j].label) {
                float iou = IOUCalculate(input_boxes[i], input_boxes[j]);
                if (iou >= 0.35) {
                    input_boxes[j].score = 0.0;
                }
            }
        }
    }

    input_boxes.erase(std::remove_if(input_boxes.begin(), input_boxes.end(), [](const BoxRotated &f) {
        return f.score == 0.0;
    }), input_boxes.end());

}


std::vector<BoxObb> YoloV5Obb::detect(JNIEnv *env, jobject image, jboolean useGpu) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
    // letterbox pad to multiple of 32
    // src_img
    float ratio;
    int dw_;
    int dh_;
    int w = img_size.width;
    int h = img_size.height;
    ncnn::Mat in_pad;
    float hw_scale = float(h)/float(w);
    if(hw_scale > 1.0){
        int new_h = input_h;
        int new_w = int(float(input_w) * hw_scale);
        ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, new_w, new_h);

        int left = int((input_w - new_w) * 0.5);
        ncnn::copy_make_border(in_net, in_pad, 0, 0, left, input_w-new_w-left, ncnn::BORDER_CONSTANT,
                               114.f);
        ratio = float(input_h) / float(h);
        dw_ = left;
        dh_ = 0;
    }else{
        int new_h = int(float(input_h) * hw_scale);
        int new_w = input_w;
        ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, new_w, new_h);

        int top = int((input_h - new_h) * 0.5);
        ncnn::copy_make_border(in_net, in_pad, top, input_h-new_h -top, 0, 0, ncnn::BORDER_CONSTANT,114.f);
        ratio = float(input_w) / float(w);
        dw_ = 0;
        dh_ = top;
    }

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_pad.substract_mean_normalize(mean, norm);

    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.set_vulkan_compute(useGpu);
    ex.input("input", in_pad);
    ncnn::Mat output;
    ex.extract("prob", output);
    const float* out = (float *)output.data;

    std::vector<BoxRotated> boxRotated;
    generate_boxes(out, boxRotated, num_class, input_w, input_h, dw_, dh_, ratio);
    nms_angle(boxRotated);

    std::vector<BoxObb> boxObbs;
    for(const auto& rotated: boxRotated){
        BoxObb boxObb;
        boxObb.score = rotated.score;
        boxObb.label = rotated.label;

        cv::Point2f vertices[4];
        rotated.rotatedRect.points(vertices);
        boxObb.x0 = (std::min)((std::max)(vertices[0].x, 0.f), float(img_size.width));
        boxObb.y0 = (std::min)((std::max)(vertices[0].y, 0.f), float(img_size.height));
        boxObb.x1 = (std::min)((std::max)(vertices[1].x, 0.f), float(img_size.width));
        boxObb.y1 = (std::min)((std::max)(vertices[1].y, 0.f), float(img_size.height));
        boxObb.x2 = (std::min)((std::max)(vertices[2].x, 0.f), float(img_size.width));
        boxObb.y2 = (std::min)((std::max)(vertices[2].y, 0.f), float(img_size.height));
        boxObb.x3 = (std::min)((std::max)(vertices[3].x, 0.f), float(img_size.width));
        boxObb.y3 = (std::min)((std::max)(vertices[3].y, 0.f), float(img_size.height));
        boxObbs.push_back(boxObb);
    }

    return boxObbs;
}

