package com.example.touchlessbiometrics

//import com.example.touchlessbiometrics.ml.SiameseModel
//import com.example.touchlessbiometrics.ml.SiamesemodelEnh

//import com.google.mediapipe.solutions.hands.Hands
//import com.google.mediapipe.solutions.hands.HandsOptions
//import com.google.mediapipe.solutions.hands.HandsResult
//import org.tensorflow.lite.DataType
//import org.tensorflow.lite.schema.Tensor
//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.firebase.FirebaseApp
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageReference
import kotlinx.coroutines.*
import java.io.*
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.*


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

    var x= FirebaseApp.initializeApp(this)

    lateinit var firebase: FirebaseFirestore

    lateinit var storage: FirebaseStorage
    lateinit var storageReference: StorageReference


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

        storage = FirebaseStorage.getInstance()
        firebase= FirebaseFirestore.getInstance()
        storageReference = storage.reference

        var progressstatus=0
        findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress= progressstatus
        findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()

        findViewById<Button>(R.id.page_matching_image1).setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_IMAGE_PICK1)
        }




        findViewById<Button>(R.id.page_matching_matchbtn).setOnClickListener {
            findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = false
            CoroutineScope(Dispatchers.Default).launch {

                runOnUiThread {
                    progressstatus = 0
                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                }
//                var stream: InputStream = getAssets().open("Indian Army Inkprint_Scattering Coeff_Array.npy")
//                var npy = Npy(stream)
//                var npyData = npy.floatElements()




                runOnUiThread {
                    progressstatus += 10
                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                }


                var py = Python.getInstance()
                var pyObj = py.getModule("myscript")
                val prediction = ArrayList<Float>()

                Log.d(TAG,image1bitmap.width.toString()+"     "+image1bitmap.height.toString())

                var outputUri = makeImageDirectory(dateFormat.format(Date()))
//                var outputUri1 = makeImageDirectory(dateFormat.format(Date()))
                runOnUiThread {
                    progressstatus += 10
                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
                }
//                    Log.d(TAG,handtype.toString()+"     "+handtype1.toString())

//                if(handtype==handtype1){
                            try {
                                var imagestr1 = getStringImage(image1bitmap)
                                var obj1 = pyObj.callAttr("saket_proces", imagestr1,8)
                                var imgstr1 = obj1.toString()
                                var data1 = android.util.Base64.decode(imgstr1, android.util.Base64.DEFAULT)
                                var btmp1 = BitmapFactory.decodeByteArray(data1, 0, data1.size)
                                Log.d("DATA",imgstr1.toString())
                                runOnUiThread {
                                    image2.setImageBitmap(btmp1)
                                }


                                var imagestr = getStringImage(image1bitmap)
                                var obj = pyObj.callAttr("main", imagestr,7)
                                var imgstr = obj.toString()
                                var data = android.util.Base64.decode(imgstr, android.util.Base64.DEFAULT)
                                Log.d("DATA",imgstr.toString())
                                var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)


                                runOnUiThread {
                                    progressstatus = 100
                                    findViewById<ProgressBar>(R.id.page_matching_progress_bar).progress = progressstatus
                                    findViewById<TextView>(R.id.page_matching_progress_percent).text = progressstatus.toString()
//                                    findViewById<TextView>(R.id.page_matching_matching_score).text = (DecimalFormat("#,###.###").format(((imgstr.split(",")[1].split(")")[0]).toFloat())).toString())
                                    if (1 == 1) {
                                        findViewById<TextView>(R.id.page_matching_matchstatus).text = imgstr.split(",")[0].split("(")[1]
                                        findViewById<TextView>(R.id.page_matching_matchstatus).setTextColor(Color.GREEN)
                                        findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = true
                                    } else {
                                        findViewById<TextView>(R.id.page_matching_matchstatus).text = "DIFFERENT"
                                        findViewById<TextView>(R.id.page_matching_matchstatus).setTextColor(Color.RED)
                                        findViewById<Button>(R.id.page_matching_matchbtn).isEnabled = true
                                    }
                                }

                                Log.d("IMAGE", "PYTHON SCRIPT Processed")

                            } catch (e: IOException) {
                                e.printStackTrace()
                            }

        }


//
        }



    }

    @SuppressLint("Range")
    fun getFileName(uri: Uri): String? {
        var result: String? = null
        if (uri.scheme == "content") {
            val cursor = contentResolver.query(uri, null, null, null, null)
            try {
                if (cursor != null && cursor.moveToFirst()) {
                    result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME))
                }
            } finally {
                cursor!!.close()
            }
        }
        if (result == null) {
            result = uri.path
            val cut = result!!.lastIndexOf('/')
            if (cut != -1) {
                result = result.substring(cut + 1)
            }
        }
        return result
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_PICK1 && resultCode == Activity.RESULT_OK && data != null) {
            val imageUri = data.data
            imageUri?.let { getFileName(it).toString() }?.let { Log.d("uri", it) }

            findViewById<Button>(R.id.page_matching_image1).text =
                (imageUri?.let { getFileName(it) })?.split(".")?.get(0)

            val inputStream = contentResolver.openInputStream(imageUri!!)
            image1bitmap = BitmapFactory.decodeStream(inputStream)
            if(image1bitmap.width>image1bitmap.height){
                val matrix = Matrix()
                matrix.postRotate(90F)
                val top = Bitmap.createBitmap(
                    image1bitmap,
                    0,
                    0,
                    image1bitmap.getWidth(),
                    image1bitmap.getHeight(),
                    matrix,
                    true
                )
                image1.setImageBitmap(top)
            }
            else {
                image1.setImageBitmap(image1bitmap)
            }
        }
        if (requestCode == REQUEST_IMAGE_PICK2 && resultCode == Activity.RESULT_OK && data != null) {
            val imageUri = data.data
            val inputStream = contentResolver.openInputStream(imageUri!!)
            image2bitmap = BitmapFactory.decodeStream(inputStream)
//            image2.setImageBitmap(image2bitmap)
            if(image2bitmap.width>image2bitmap.height){
                val matrix = Matrix()
                matrix.postRotate(90F)
                val top = Bitmap.createBitmap(
                    image2bitmap,
                    0,
                    0,
                    image2bitmap.getWidth(),
                    image2bitmap.getHeight(),
                    matrix,
                    true
                )
                image2.setImageBitmap(top)
            }
            else {
                image2.setImageBitmap(image2bitmap)
            }
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
