package com.sample.yolotflite;

import android.graphics.RectF;

public class Recognition {
    final String[] fire_names = new String[]{
            "smoke", "fire"
    };


    final String[] coco_names = new String[]{
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    //객체 숫자
    public Integer class_Idx;

    //화면에 비춰질 객체이름
    public String title;

    //객체 탐지 확률
    public Float confidence;

    //디텍션 위치
    public RectF location;

    //생성자
    public Recognition(int class_Idx, Float confidence, RectF rectf) {
        this.class_Idx = class_Idx;
        this.confidence = confidence;
        this.location = rectf;
    }

    public String className_Fire(int class_Idx) {
        return fire_names[class_Idx];
    }

}
