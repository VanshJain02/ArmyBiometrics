package com.example.touchlessbiometrics

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import java.io.*
import java.util.HashMap
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.pow
import kotlin.math.sqrt

class BitmapProcessing: AppCompatActivity() {
    fun processing(bitmap: Bitmap) {
        var greymap: HashMap<Any, Bitmap> =
            hashMapOf(7 to bitmap, 11 to bitmap, 15 to bitmap, 19 to bitmap)

        greymap = makeGray(bitmap)

        Log.d("IMAGE", "PYTHON SCRIPT ACCESSING")

        var py = Python.getInstance()
        var pyObj = py.getModule("myscript")
        if (greymap != null) {
            for (key in greymap.keys) {
                try {
                    Log.d("IMAGE", "PYTHON SCRIPT ACCESSed")

                    var imagestr = getStringImage(greymap[key])
                    var obj = pyObj.callAttr("main", imagestr)
                    var imgstr = obj.toString()
                    var data = android.util.Base64.decode(imgstr, android.util.Base64.DEFAULT)
                    var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)

                    Log.d("IMAGE", greymap[key].toString())
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
    }

    fun makeGray(bitmap: Bitmap) : HashMap<Any, Bitmap> {

        var hands = Hands(this, HandsOptions.builder()
            .setStaticImageMode(true)
            .setMaxNumHands(2)
            .setRunOnGpu(true)
            .build()
        )
        hands.send(bitmap)

        var top1: Bitmap = bitmap
        var fingers: HashMap<Any, Bitmap> = hashMapOf(7 to bitmap,11 to bitmap,15 to bitmap,19 to bitmap)
        // Connects MediaPipe Hands solution to the user-defined HandsResultImageView.
        hands.setResultListener { handsResult: HandsResult? ->
            if (handsResult != null) {
//                var coords: HashMap<Any,Pair<Array<Double>,Array<Int>>> = hashMapOf(7 to Pair(arrayOf<Double>(-1.0), arrayOf<Int>(-1)),11 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),15 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),19 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)))
                for(key in fingers.keys) {
                    Log.d("IMAGE", "Start Cropping")
                    var direction = 0
                    var start_point = arrayOf<Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).y) * bitmap.height)).toDouble())
                    var end_point = arrayOf<Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).y) * bitmap.height)).toDouble())
                    var m = (-end_point[1]+start_point[1])/(-end_point[0]+start_point[0]+0.000001)
                    var angle= atan(m) *180/Math.PI
                    var dist_pt = sqrt((end_point[0] - start_point[0]).pow(2.0) + (end_point[1] - start_point[1]).pow(2.0))
                    var mid_point = arrayOf<Double>(((start_point[0] + end_point[0]) / 2), ((start_point[1] + end_point[1]) / 2))
                    var axesy = (dist_pt * 1.6 / 2)
                    var axesLength: Array<Int>
                    var palm_pointx = ((handsResult.multiHandLandmarks().get(0).landmarkList.get((9)).x )* bitmap.width).toInt()
                    if(abs(angle) <45) {
                        if(palm_pointx < mid_point[0]) {
                            direction = 1
                        }
                        else {
                            direction = 2
                        }
                        axesLength = arrayOf<Int>(
                            (abs(axesy/2)).toInt(),
                            (abs(axesy) + abs(axesy / 12)).toInt()
                        )
                    }
                    else{
                        axesLength = arrayOf<Int>(
                            (abs(axesy) + abs(axesy / 12)).toInt(),
                            (abs(axesy / 2)).toInt()
                        )

                    }
//                    coords[key] = Pair<Array<Double>, Array<Int>>(mid_point, axesLength)


                    top1 = Bitmap.createBitmap(bitmap, (mid_point[0] - axesLength[1]).toInt(), (mid_point[1] - axesLength[0]).toInt(), 2 * axesLength[1].toInt(), 2 * axesLength[0].toInt())
                    if(abs(angle) <45) {
                        val matrix = Matrix()
                        if(direction == 1) {matrix.postRotate(270F)}
                        if(direction == 2) {matrix.postRotate(90F)}
                        top1 = Bitmap.createBitmap(
                            top1,
                            0,
                            0,
                            top1.getWidth(),
                            top1.getHeight(),
                            matrix,
                            true
                        )
                    }
                    fingers[key]=top1
                }
            }
            else{
                Log.d(
                    "IMAGE",
                    "Hands error"
                )
            }
        }
        hands.setErrorListener { message: String, e: RuntimeException? ->
            Log.d(
                "IMAGE",
                "MediaPipe Hands error"
            )
        }
        return fingers
    }

    private fun getStringImage(grayBitmap: Bitmap?): String? {
        var baos= ByteArrayOutputStream()
        grayBitmap?.compress(Bitmap.CompressFormat.PNG,100,baos)
        var imgByte = baos.toByteArray()
        var encodedImg = android.util.Base64.encodeToString(imgByte,android.util.Base64.DEFAULT)
        return encodedImg
    }
}