package com.example.touchlessbiometrics

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.media.Image
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity

import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.lifecycle.VIEW_MODEL_STORE_OWNER_KEY
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.touchlessbiometrics.ml.SiameseModel
import com.google.firebase.storage.UploadTask
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.schema.Tensor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.lang.Double
import java.math.BigDecimal
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.HashMap
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.pow
import kotlin.math.sqrt


private val REQUEST_IMAGE_PICK1 = 70
private val REQUEST_IMAGE_PICK2 = 71
val dateFormat = SimpleDateFormat("yyyy-MM-dd HH-mm-ss")


class PAGE_Matching : AppCompatActivity() {
    private lateinit var image1: ImageView
    private lateinit var image2: ImageView
    private var imageSize = 96
    private lateinit var image1bitmap: Bitmap
    private lateinit var image2bitmap: Bitmap
    private var TAG = "MATCHING"

//    @SuppressLint("MissingInflatedId")
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_page_matching)

        image1 = findViewById(R.id.page_matching_imageView1)
        image2 = findViewById(R.id.page_matching_imageView2)

        if(!Python.isStarted()){
            Python.start(AndroidPlatform(this))
        }
        var progressstatus=0
        findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress= progressstatus
        findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()

        findViewById<Button>(R.id.page_matching_image1).setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_IMAGE_PICK1)
        }
        findViewById<Button>(R.id.page_matching_image2).setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_IMAGE_PICK2)
        }



        findViewById<Button>(R.id.page_matching_matchbtn).setOnClickListener {
            findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = false
            CoroutineScope(Dispatchers.Default).launch {
                runOnUiThread {
                    progressstatus = 0
                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                }


                var greymap: HashMap<Any, Bitmap> = hashMapOf(7 to image1bitmap, 11 to image1bitmap, 15 to image1bitmap, 19 to image1bitmap)
                var greymap1: HashMap<Any, Bitmap> = hashMapOf(7 to image2bitmap, 11 to image2bitmap, 15 to image2bitmap, 19 to image2bitmap)
                runOnUiThread {
                    progressstatus += 10
                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                }

                val job = CoroutineScope(Dispatchers.Default).launch {
                greymap=makeGray(image1bitmap)
                greymap1=makeGray(image2bitmap)


                }


            runBlocking {
                job.join()
            }

            var py = Python.getInstance()
            var pyObj = py.getModule("myscript")
            val prediction = ArrayList<Float>()

            Log.d(TAG,image1bitmap.width.toString()+"     "+image1bitmap.height.toString())

            var outputUri = makeImageDirectory(dateFormat.format(Date()))
            var outputUri1 = makeImageDirectory(dateFormat.format(Date()))
            runOnUiThread {
                progressstatus += 10
                findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
            }



            if (greymap != null && greymap1!=null) {
                for (key in greymap.keys) {
                    try {

                        var imagestr = getStringImage(greymap[key])
                        var obj = pyObj.callAttr("main", imagestr,3)
                        var imgstr = obj.toString()
                        var data = android.util.Base64.decode(imgstr, android.util.Base64.DEFAULT)
                        var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)

                        var imagestr1 = getStringImage(greymap1[key])
                        var obj1 = pyObj.callAttr("main", imagestr1,3)
                        var imgstr1 = obj1.toString()
                        var data1 = android.util.Base64.decode(imgstr1, android.util.Base64.DEFAULT)
                        var btmp1 = BitmapFactory.decodeByteArray(data1, 0, data1.size)
                        runOnUiThread {
                            progressstatus += 10
                            findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                            findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                        }

                        Log.d("IMAGE", greymap[key].toString())
                        if (btmp != null && btmp1!=null) {
                            var image1 = Bitmap.createScaledBitmap(btmp, imageSize, imageSize, false)
                            var image2 = Bitmap.createScaledBitmap(btmp1, imageSize, imageSize, false)
                            Log.d(TAG,"scaled done")

                            var fOut1: OutputStream? = null
                            val file1 = File(
                                outputUri,
                                  "imageog_"+key.toString()+".png"
                            )
                            fOut1 = FileOutputStream(file1)
                            greymap[key]?.compress(Bitmap.CompressFormat.PNG, 100, fOut1)
                            var fOut: OutputStream? = null
                            val file = File(
                                outputUri,
                                "image_"+key.toString()+".png"
                            ) // the File to save , append increasing numeric counter to prevent files from getting overwritten.
                            fOut = FileOutputStream(file)
                            Log.d("IMAGE", "NEXT2")

                            btmp?.compress(Bitmap.CompressFormat.PNG, 100, fOut)

                            var fOut2: OutputStream? = null
                            val file2 = File(
                                outputUri1,
                                "image2og_"+key.toString()+".png"
                            )
                            fOut2 = FileOutputStream(file2)
                            greymap1[key]?.compress(Bitmap.CompressFormat.PNG, 100, fOut2)
                            var fOut3: OutputStream? = null
                            val file3 = File(
                                outputUri1,
                                "image2_"+key.toString()+".png"
                            ) // the File to save , append increasing numeric counter to prevent files from getting overwritten.
                            fOut3 = FileOutputStream(file3)
                            Log.d("IMAGE", "NEXT2")

                            btmp1?.compress(Bitmap.CompressFormat.PNG, 100, fOut3)

//                            var image1 = Bitmap.createScaledBitmap(image1bitmap, imageSize, imageSize, false)
//                            var image2 = Bitmap.createScaledBitmap(image2bitmap, imageSize, imageSize, false)


                            var py = Python.getInstance()
                            var pyObj = py.getModule("myscript")


                            var imagestr = getStringImage(btmp)
                            var obj = pyObj.callAttr("getpixel",imagestr)
                            var imgstr = obj.toString()
                            val intValues = imgstr.split(" ")
//
                            var imagestr1 = getStringImage(btmp1)
                            var obj1 = pyObj.callAttr("getpixel",imagestr1)
                            var imgstr1 = obj1.toString()
                            val intValues1 = imgstr1.split(" ")


                            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 96, 96, 3), DataType.FLOAT32)
                            var byteBuffer1: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
                            byteBuffer1.order(ByteOrder.nativeOrder())
//            var intValues = IntArray(imageSize * imageSize)
//            image1.getPixels(intValues, 0, image1.width, 0, 0, image1.width, image1.height)
                            var pixel = 0
                            for (i in 0 until imageSize) {
                                for (j in 0 until imageSize) {
                                    for (k in 0 until 3){
                                        var vals = intValues[pixel++].toInt()// RGB
                                        byteBuffer1.putFloat((vals * (1F / 255f).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat()))
//                    byteBuffer1.putFloat(((vals and 0xff) * (1F / 255f).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat()))
//                    byteBuffer1.putFloat((((vals shr 8) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat())
//                    byteBuffer1.putFloat((((vals shr 16) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat())
//                    Log.d(TAG,(((vals shr 16) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()+"\t"+(((vals shr 8) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()+"\t"+(((vals and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()))
                                    }
                                }
                            }



                            val inputFeature1 = TensorBuffer.createFixedSize(intArrayOf(1, 96, 96, 3), DataType.FLOAT32)
                            var byteBuffer2: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
                            byteBuffer2.order(ByteOrder.nativeOrder())
                            var pixel1 = 0

                            pixel1=0
                            for (i in 0 until imageSize) {
                                for (j in 0 until imageSize) {
                                    for (k in 0 until 3){
                                        var vals = intValues1[pixel1++].toInt()// RGB
                                        byteBuffer2.putFloat((vals * (1F / 255f).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat()))
//                    byteBuffer1.putFloat(((vals and 0xff) * (1F / 255f).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat()))
//                    byteBuffer1.putFloat((((vals shr 8) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat())
//                    byteBuffer1.putFloat((((vals shr 16) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat())
//                    Log.d(TAG,(((vals shr 16) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()+"\t"+(((vals shr 8) and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()+"\t"+(((vals and 0xff) * (1F / 255f)).toDouble().toBigDecimal().setScale(6, BigDecimal.ROUND_HALF_UP).toFloat().toString()))
                                    }
                                }
                            }

                            Log.d(TAG, "buffer done")
                            val model = SiameseModel.newInstance(this@PAGE_Matching)

// Creates inputs for reference.
                            Log.d(TAG, inputFeature0.toString())
                            inputFeature0.loadBuffer(byteBuffer1)
// Runs model inference and gets result.
                            val outputs = model.process(inputFeature0)
                            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                            inputFeature1.loadBuffer(byteBuffer2)
                            val outputs1 = model.process(inputFeature1)
                            val outputFeature1 = outputs1.outputFeature0AsTensorBuffer

                            var confidence = outputFeature0.floatArray
                            var confidence1 = outputFeature1.floatArray



                            val anc_enc = confidence
                            val pos_enc = confidence1


// compute the Euclidean distance
                            val diff = DoubleArray(confidence.size) { (anc_enc[it]).toDouble() - (pos_enc[it]).toDouble() }
                            val distance = (1 - sqrt(diff.map { it.pow(2.0) }.sum()))
                            Log.d(TAG,key.toString()+"\t"+distance.toString())

                            prediction.add(distance.toFloat())
                            runOnUiThread {
                                progressstatus += 10
                                findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                                findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                            }

                            Log.d(TAG,"started matching")

                            Log.d(TAG,distance.toString())
                            Log.d(TAG,confidence.size.toString())
                            model.close()


                        }
                        Log.d("IMAGE", "PYTHON SCRIPT Processed")

                    } catch (e: IOException) {
                        e.printStackTrace()
                    }

                }
                //                    greymap=null
            }//

            var count=0
            var maxl=0.0
            var maxlen=0
            var maxle=0
            var mean=0.0





            for (q in prediction){
                mean+=q
                if(q>0.4){
                count+=1}
                if(q>0.0){
                maxlen+=1}
                if(q>0.3){
                maxl+=q
                maxle+=1}
                Log.d(TAG, q.toString())
            }
            var final_answer=0
            if(count>=2 || (maxl>1 && maxlen>=3)){
                if(maxl/maxle>0.4) {
                    final_answer = 1
            }
            else {
                final_answer=0
            }
            }
            else {
                final_answer=0
            }
            //MATHING START
                runOnUiThread {
                    findViewById<TextView>(R.id.page_matching_matching_score).text =
                        (DecimalFormat("#,###.0").format((mean / 4) * 100)).toString()

                    if (final_answer == 1) {

                        findViewById<TextView>(R.id.page_matching_matchstatus).text = "SAME"
                        findViewById<TextView>(R.id.page_matching_matchstatus).setTextColor(Color.GREEN)
                        findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = true


                    } else {
                        findViewById<TextView>(R.id.page_matching_matchstatus).text = "DIFFERENT"
                        findViewById<TextView>(R.id.page_matching_matchstatus).setTextColor(Color.RED)
                        findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = true
                    }


                }

// Releases model resources if no longer used.
        }

        }



    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_PICK1 && resultCode == Activity.RESULT_OK && data != null) {
            val imageUri = data.data
            val inputStream = contentResolver.openInputStream(imageUri!!)
            image1bitmap = BitmapFactory.decodeStream(inputStream)
            image1.setImageBitmap(image1bitmap)
        }
        if (requestCode == REQUEST_IMAGE_PICK2 && resultCode == Activity.RESULT_OK && data != null) {
            val imageUri = data.data
            val inputStream = contentResolver.openInputStream(imageUri!!)
            image2bitmap = BitmapFactory.decodeStream(inputStream)
            image2.setImageBitmap(image2bitmap)
        }

    }


    private fun getStringImage(grayBitmap: Bitmap?): String? {
        var baos= ByteArrayOutputStream()
        grayBitmap?.compress(Bitmap.CompressFormat.PNG,100,baos)
        var imgByte = baos.toByteArray()
        var encodedImg = android.util.Base64.encodeToString(imgByte,android.util.Base64.DEFAULT)
        return encodedImg
    }
    fun bgrToRgb(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (i in pixels.indices) {
            pixels[i] = Color.rgb(Color.blue(pixels[i]), Color.green(pixels[i]), Color.red(pixels[i]))
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
    fun flipBitmapHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    suspend fun makeGray(bitma: Bitmap) : HashMap<Any, Bitmap> {
        val bitmap = flipBitmapHorizontally(bitma)
        var rotatebool = false
        if(bitma.width>bitma.height){
            rotatebool= true
        }
        var hands = Hands(this, HandsOptions.builder()
            .setStaticImageMode(true)
            .setMaxNumHands(2)
            .setRunOnGpu(true)
            .build()
        )
        val  job = CoroutineScope(Dispatchers.Default).launch {
            hands.send(bitmap)
        }
        job.join()
        var top1: Bitmap = bitmap

        var fingers: HashMap<Any, Bitmap> = hashMapOf(7 to bitmap,11 to bitmap,15 to bitmap,19 to bitmap)
        Log.d(TAG,"SHAPE:"+bitmap.width+"\t"+bitmap.height)

        // Connects MediaPipe Hands solution to the user-defined HandsResultImageView.
        Log.d("IMAGE","Waiting for result")
        hands.setResultListener { handsResult: HandsResult? ->
            Log.d("IMAGE","Waiting for result")
            if (handsResult != null) {
                //                var coords: HashMap<Any,Pair<Array<Double>,Array<Int>>> = hashMapOf(7 to Pair(arrayOf<Double>(-1.0), arrayOf<Int>(-1)),11 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),15 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),19 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)))
                for(key in fingers.keys) {
                    Log.d("IMAGE", "Start Cropping")
                    var direction = 0
                    var start_point = arrayOf<kotlin.Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).y) * bitmap.height)).toDouble())
                    Log.d("MATCHING",key.toString()+ " startpt"+"\t"+start_point[0].toString()+"\t"+start_point[1].toString())


                    var end_point = arrayOf<kotlin.Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).y) * bitmap.height)).toDouble())
                    Log.d("MATCHING",key.toString()+ " endpt"+"\t"+end_point[0].toString()+"\t"+end_point[1].toString())

                    var m = (-end_point[1]+start_point[1])/(-end_point[0]+start_point[0]+0.000001)
                    var angle= atan(m) *180/Math.PI
                    var dist_pt = sqrt((end_point[0] - start_point[0]).pow(2.0) + (end_point[1] - start_point[1]).pow(2.0))
                    Log.d("MATCHING",key.toString()+ " distpt"+"\t"+dist_pt.toString())

                    var mid_point = arrayOf<kotlin.Double>(((start_point[0] + end_point[0]) / 2), ((start_point[1] + end_point[1]) / 2))
                    var axesy = (dist_pt * 1.6 / 2)
                    Log.d("MATCHING",key.toString()+ " axesy"+"\t"+axesy.toString())

                    var axesLength: Array<Int>
                    var half_length_diag = (dist_pt*2.5)/2

                    var palm_pointx = ((handsResult.multiHandLandmarks().get(0).landmarkList.get((9)).x )* bitmap.width).toInt()
                    if(abs(angle) <45) {
                        if(palm_pointx < mid_point[0]) {
                            direction = 1
                        }
                        else {
                            direction = 2
                        }
                        axesLength = arrayOf<Int>(
                            (abs(axesy/2)+abs(axesy/6)).toInt(),
                            (abs(axesy) + abs(axesy / 10)).toInt()
                        )
                    }
                    else{
                        axesLength = arrayOf<Int>(
                            (abs(axesy) + abs(axesy / 10)).toInt(),
                            (abs(axesy / 2)+abs(axesy/6)).toInt()
                        )
                    }
//                    Log.d("MATCHING",key.toString()+ "\t"+axesLength[0].toString()+"\t"+axesLength[1].toString())

                    top1 = Bitmap.createBitmap(bitmap, (mid_point[0] - axesLength[1]).toInt(), (mid_point[1] - axesLength[0]*1.2).toInt(), 2 * axesLength[1].toInt(), (3.2 * axesLength[0]-half_length_diag).toInt())

                    if(rotatebool){
                        top1 = Bitmap.createBitmap(bitmap, (mid_point[0] - axesLength[1]*0.4).toInt(), (mid_point[1] - axesLength[0]*0.9).toInt(), (1.6 * axesLength[1]).toInt(), (1.8* axesLength[0]).toInt())
                    }
                    Log.d("MATCHING",key.toString()+ "\t"+top1.getWidth().toString()+"\t"+ top1.getHeight())
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
                Log.d("IMAGE", "Hands error")
            }
        }
        hands.setErrorListener { message: String, e: RuntimeException? ->
            Log.d(
                "IMAGE",
                "MediaPipe Hands error"
            )
        }
        Log.d("IMAGE","MAKE GRAY PROCESSED")
        delay(2000)
        return fingers
    }
    private fun makeImageDirectory(name: String): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
            Log.d("FILE", mFile.toString())
            File((mFile.toString() + "/"+resources.getString(R.string.app_name)), name).apply {
                mkdirs()
            }
        }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

}
