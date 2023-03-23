package com.example.touchlessbiometrics
import android.Manifest

object Constants {

    const val TAG = "camerax"
    const val FILE_NAME_FORMAT = "yy-MM-dd-HH-mm-ss-sss"
    const val REQUEST_CODE_PERMISSIONS = 123
    val SHEET_NAME = "entry"
    val FILE_NAME = "DATA"
    val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_COARSE_LOCATION,Manifest.permission.ACCESS_FINE_LOCATION)
}