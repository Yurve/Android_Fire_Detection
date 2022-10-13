package com.sample.yolotflite;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


public class TfLiteRunner {
    Context context;
    Activity activity;
    Interpreter interpreter;
    String fileName = "yolov5s_fire_fp16_320.tflite";
    //아래값 = 6300
    int outputBox = (int) ((Math.pow((MainActivity.INPUT_SIZE / 32.0), 2) + Math.pow((MainActivity.INPUT_SIZE / 16.0), 2) + Math.pow((MainActivity.INPUT_SIZE / 8.0), 2)) * 3);
    String[] classes;
    static int BatchSize = 4; //입력 사진의 형태 int -> 1 , float ->4
    static int PixelSize = 3; //RGB 3개
    float imageMean = 0f;
    float imageSTD = 255.0f;
    int numClass = 80;
    float objectThresh = 0.7f;
    ArrayList<Recognition> detections;

    //생성자
    public TfLiteRunner(Context context, Activity activity) {
        this.context = context;
        this.activity = activity;
    }

    //기기에 모델 다운로드 및 TensorFlow Lite 인터프리터 초기화
    public void setInterpreter() {
        AssetManager assetManager = context.getAssets();
        File outputFile = new File(context.getFilesDir() + "/" + fileName);
        try {
            InputStream inputStream = assetManager.open(fileName);
            OutputStream outputStream = new FileOutputStream(outputFile);

            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            inputStream.close();
            outputStream.flush();
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        interpreter = new Interpreter(outputFile, new Interpreter.Options());
        int[] shape = interpreter.getOutputTensor(0).shape();
        numClass = shape[shape.length - 1] - 5;
    }

    public void readCoCo() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open("label_fire.txt")));
            String line;
            List<String> classList = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                classList.add(line);
            }
            //arrayList -> String []
            classes = new String[classList.size()];
            classList.toArray(classes);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    //모델의 입력과 출력 형식을 확인했으면 입력 데이터를 가져와 모델에 적합한 입력 모양을 만드는 데 필요한 데이터 변환을 수행합니다.
    public void setInput(Bitmap bitmap) {
        //90도 회전 되어있는 상태라 다시 돌려줘야함
        Matrix matrix = new Matrix();
        matrix.setRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        //새 바이트 버퍼를 할당
        ByteBuffer buffer = ByteBuffer.allocateDirect(BatchSize * PixelSize * MainActivity.INPUT_SIZE * MainActivity.INPUT_SIZE)
                .order(ByteOrder.nativeOrder());
        //position 을 0으로 (원래 0이 디폴트긴 한데 오류방지)
        buffer.rewind();
        //총 넓이
        int area = MainActivity.INPUT_SIZE * MainActivity.INPUT_SIZE;
        //넓이 크기 만큼 배열 생성
        int[] bitmapData = new int[area];
        //pixel 값 받아와서 bitmapData 배열에 넣기
        bitmap.getPixels(bitmapData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convert the image to floating point.
        for (int i = 0; i < MainActivity.INPUT_SIZE; i++) {
            for (int j = 0; j < MainActivity.INPUT_SIZE; ++j) {
                int pixelValue = bitmapData[i * MainActivity.INPUT_SIZE + j];

                buffer.putFloat((((pixelValue >> 16) & 0xFF) - imageMean) / imageSTD);
                buffer.putFloat((((pixelValue >> 8) & 0xFF) - imageMean) / imageSTD);
                buffer.putFloat(((pixelValue & 0xFF) - imageMean) / imageSTD);
            }
        }

        //position 을 처음위치(0)으로
        buffer.rewind();

        //  ByteBuffer modelOutput = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.nativeOrder());
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(outputBox * (numClass + 5) * BatchSize).order(ByteOrder.nativeOrder());
        modelOutput.rewind();

        //추론실행
        interpreter.run(buffer, modelOutput);


        //가져오기
        modelOutput.rewind();
        float[][][] out = new float[1][outputBox][numClass + 5];
        for (int i = 0; i < outputBox; ++i) {
            for (int j = 0; j < numClass + 5; ++j) {
                out[0][i][j] = modelOutput.getFloat();
            }
            // Denormalize x,y,w,h
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= MainActivity.INPUT_SIZE;
            }
        }

        // 각 bounding box 에 대해 가장 확률이 높은 Class 예측
        for (int i = 0; i < outputBox; ++i) {
            float confidence = out[0][i][4];
            int detectionClass = -1;
            float maxClass = 0;

            float[] _classes = new float[classes.length];
            for (int c = 0; c < classes.length; ++c) {
                _classes[c] = out[0][i][5 + c]; // classes: 각 class 의 확률 계산
            }

            for (int c = 0; c < classes.length; ++c) {
                if (_classes[c] > maxClass) {
                    detectionClass = c;
                    maxClass = _classes[c]; // 가장 큰 확률의 class 로 선정
                }
            }


            float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > objectThresh) {
                float xPos = out[0][i][0];
                float yPos = out[0][i][1];
                float width = out[0][i][2];
                float height = out[0][i][3];

                RectF rect = new RectF(Math.max(0, xPos - width)
                        , Math.max(0, yPos - height)
                        , Math.min(bitmap.getWidth() - 1, xPos + width / 2)
                        , Math.min(bitmap.getHeight() - 1, yPos + height / 2));
                detections = new ArrayList<>();
                Recognition recognition = new Recognition(detectionClass, confidenceInClass, rect);
                detections.add(recognition);
            }
        }
        JSONObject jsonObject = new JSONObject();
        if (detections != null && detections.size() > 0) {
            ArrayList<Recognition> recognitions = nms(detections);
            if (recognitions.size() > 0) {
                for (Recognition recognition : recognitions) {
                    String name = recognition.className_Fire(recognition.class_Idx);
                    String num = Math.round(recognition.confidence * 100) + "%";
                    try {
                        jsonObject.put(name, num);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                }
                Log.d("확인: ", jsonObject.toString());
                activity.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (jsonObject.length() >= 0) {
                            Toast.makeText(context, jsonObject.toString(), Toast.LENGTH_SHORT).show();
                        }
                    }
                });
            }

        }
    }

    public ArrayList<Recognition> nms(ArrayList<Recognition> recognitions) {
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < classes.length; k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(Recognition o1, Recognition o2) {
                                    return Float.compare(o1.confidence, o2.confidence);
                                }
                            });

            for (int i = 0; i < recognitions.size(); i++) {
                if (recognitions.get(i).class_Idx == k) {
                    pq.add(recognitions.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.location;
                    if (box_iou(max.location, b) < 0.6f) {
                        pq.add(detection);
                    }
                }
            }
        }

        return nmsList;
    }

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }

}

