package com.sample.yolotflite;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

public class PermissionSupport {
    public final static int MULTIPLE_PERMISSIONS = 1;
    private final Context context;
    private final Activity activity;
    public String[] permissions = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.INTERNET};


    //생성자
    public PermissionSupport(Activity activity, Context context) {
        this.activity = activity;
        this.context = context;
    }


    public void onCheckPermission() {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                if (ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)) {
                    Toast.makeText(context, "권한 허용", Toast.LENGTH_SHORT).show();
                } else {
                    ActivityCompat.requestPermissions(activity, permissions, MULTIPLE_PERMISSIONS);
                }
            }
        }
    }

}
