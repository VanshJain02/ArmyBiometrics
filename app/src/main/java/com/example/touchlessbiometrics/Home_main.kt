package com.example.touchlessbiometrics


import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Log
import android.util.SparseArray
import android.view.View
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.Toast
import androidx.annotation.IdRes
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.touchlessbiometrics.screens.HomeFragment
import com.example.touchlessbiometrics.screens.SettingFragment
import com.google.android.gms.location.*
import com.google.android.gms.tasks.OnFailureListener
import com.google.android.gms.tasks.OnSuccessListener
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.firebase.FirebaseApp
import com.google.firebase.firestore.DocumentReference
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageReference
import com.google.firebase.storage.UploadTask
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import kotlinx.coroutines.*
import java.io.*
import java.lang.Runnable
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.pow
import kotlin.math.sqrt


class Home_main : AppCompatActivity() {


    private lateinit var outputDirectory: File
    private val cameraRequestCode = 42

    private lateinit var photoFile: File

    private val rotateOpen: Animation by lazy { AnimationUtils.loadAnimation( this, R.anim.rotate_open_anim)}
    private val rotateClose: Animation by lazy { AnimationUtils.loadAnimation( this, R.anim.rotate_close_anim) }
    private val fromBottom: Animation by lazy { AnimationUtils. loadAnimation( this, R.anim.from_bottom_anim)}
    private val toBottom: Animation by lazy { AnimationUtils.loadAnimation(this, R.anim.to_bottom_anim) }
    private var clicked = false
    private var processingMod: Int = 1
    private var capMod: Int = 1

    private var homeFragment = HomeFragment()
    private lateinit var imagePathArrayList: ArrayList<String>
    private var currentSelectItemId = R.id.miHome
    private var savedStateSparseArray = SparseArray<Fragment.SavedState>()
    var mFusedLocationClient: FusedLocationProviderClient? = null
    var lat=""
    var lon=""
    var x= FirebaseApp.initializeApp(this)

    var firebase: FirebaseFirestore = FirebaseFirestore.getInstance()

    lateinit var storage: FirebaseStorage
    lateinit var storageReference: StorageReference



    companion object {
        const val SAVED_STATE_CONTAINER_KEY = "ContainerKey"
        const val SAVED_STATE_CURRENT_TAB_KEY = "CurrentTabKey"
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home_main)
        val bottom_navigation = findViewById<BottomNavigationView>(R.id.bottomNavigationView)
        val camera_fab = findViewById<FloatingActionButton>(R.id.fab)
        val camera_btn = findViewById<FloatingActionButton>(R.id.btn_camera)
        val gallery_btn = findViewById<FloatingActionButton>(R.id.btn_gallery)

        var count=0

        bottom_navigation.background = null


        if(!Python.isStarted()){
            Python.start(AndroidPlatform(this))
        }

        val settingFragment = SettingFragment()
        if (savedInstanceState != null) {
            Log.d("SAVE","SAVESTATEUSED")
            savedStateSparseArray = savedInstanceState.getSparseParcelableArray(SAVED_STATE_CONTAINER_KEY)
                ?: savedStateSparseArray
            currentSelectItemId = savedInstanceState.getInt(SAVED_STATE_CURRENT_TAB_KEY)
        } else {
            makeCurrentFragment(homeFragment,R.id.miHome)
        }

        bottom_navigation.setOnNavigationItemSelectedListener {
            when(it.itemId){
                R.id.miHome -> {
                    if(currentSelectItemId!=R.id.miHome) {
                        currentSelectItemId=R.id.miHome
                        makeCurrentFragment(homeFragment, R.id.miHome)
                    }
                }
                R.id.miSetting ->{
                    if(currentSelectItemId!=R.id.miSetting){
                        currentSelectItemId=R.id.miSetting
                        makeCurrentFragment(settingFragment,R.id.miSetting)}
                }
            }
            true
        }
        Log.d("IMAGE","HomeMain")


        processingMod = getProcessingMod()
        capMod = getCaptureMod()
        imagePathArrayList = ArrayList()
        camera_fab.setOnClickListener{
//            val allimagespath = ArrayList<String>()
//            val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
//                Log.d("FILE", mFile.toString())
//                File(mFile.toString() + "/" + "palm").walkTopDown().forEach {
//                    Log.d("PROCESSING", it.toString())
//                    allimagespath.add(it.path)
//                }
//            }
//            var name_count =0
//            for (i in 1 until allimagespath.size) {
//                val job = CoroutineScope(Dispatchers.Default).launch {
//
//                    name_count += 1
//                    Log.d("PROCESSING", i.toString() + "\t" + "STARTED PROCESSING")
//                    var bitmap = getBitmapFromFilePath(allimagespath.get(i))
//                    var greymap: HashMap<Any, Bitmap> =
//                        hashMapOf(7 to bitmap, 11 to bitmap, 15 to bitmap, 19 to bitmap)
//
//                    val job = CoroutineScope(Dispatchers.Default).launch {
//                        greymap = makeGray(bitmap)
//                    }
//                    runBlocking {
//                        job.join()
//                    }
//                    Log.d("IMAGE", "PYTHON SCRIPT ACCESSING")
//
//                    var py = Python.getInstance()
//                    var pyObj = py.getModule("myscript")
//                    var outputUri = makeProcessingImageDirectory(dateFormat.format(Date()))
//                    if (greymap != null) {
//                        for (key in greymap.keys) {
//                            try {
//
//                                var imagestr = getStringImage(greymap[key])
//                                var obj = pyObj.callAttr("main", imagestr, processingMod)
//                                var imgstr = obj.toString()
//                                var data =
//                                    android.util.Base64.decode(imgstr, android.util.Base64.DEFAULT)
//                                var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)
//
//                                Log.d("IMAGE", greymap[key].toString())
//                                if (btmp != null) {
//                                    var fOut1: OutputStream? = null
//                                    val file1 = File(
//                                        outputUri,
//                                        name_count.toString() + "_" + key.toString() + "og.png"
//                                    )
//                                    fOut1 = FileOutputStream(file1)
//                                    greymap[key]?.compress(Bitmap.CompressFormat.PNG, 100, fOut1)
//                                    var fOut: OutputStream? = null
//                                    val file = File(
//                                        outputUri,
//                                        name_count.toString() + "_" + key.toString() + ".png"
//                                    ) // the File to save , append increasing numeric counter to prevent files from getting overwritten.
//                                    fOut = FileOutputStream(file)
//                                    Log.d("IMAGE", "NEXT2")
//
//                                    btmp?.compress(Bitmap.CompressFormat.PNG, 100, fOut)
//
////                                    var progressB: ProgressDialog = ProgressDialog(this@Home_main)
////                                    progressB.setTitle("uploading")
////                                    progressB.show()
//
//
//                                    // the File to save , append increasing numeric counter to prevent files from getting overwritten.
//                                    //                                fOut.flush(); // Not really required
//                                    fOut.close();
//                                    fOut1.close()
//
//                                }
//                                Log.d("IMAGE", "PYTHON SCRIPT Processed")
//
//                            } catch (e: IOException) {
//                                e.printStackTrace()
//                            }
//
//                        }
//                        //                    greymap=null
//                    }//
//                    Log.d("PROCESSED", bitmap.toString())
//                }
//                runBlocking {
//                    job.join()
//                }
//            }



//            Log.d("PROCESSING")
            onCameraFabClicked()
        }
        camera_btn.setOnClickListener{

            var intent= Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            photoFile = getPhotoFile(count.toString()+".png")
            count+=1
            val fileProvider = FileProvider.getUriForFile(this,"com.example.touchlessbiometrics.fileprovider",photoFile)
            intent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            startActivityForResult(intent,cameraRequestCode)
        }
        gallery_btn.setOnClickListener{
            val gallery = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI)
            photoFile = getPhotoFile(count.toString()+".png")
            count+=1
            val fileProvider = FileProvider.getUriForFile(this,"com.example.touchlessbiometrics.fileprovider",photoFile)
            intent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            this.startActivityForResult(gallery, 100)
        }
        outputDirectory = getOutputDirectory()
        storage = FirebaseStorage.getInstance()
        storageReference = storage.reference
        listImageDirectory()
        if(allPermissionGranted()){


        }
        else{
            ActivityCompat.requestPermissions(
                this, Constants.REQUIRED_PERMISSIONS,
                Constants.REQUEST_CODE_PERMISSIONS)
        }
    }
    private fun getProcessingMod(): Int{
        val sharedPref = getSharedPreferences("processing", Context.MODE_PRIVATE)
        return sharedPref.getInt("mod", 1)
    }
    private fun getCaptureMod(): Int{
        val sharedPref = getSharedPreferences("capturemode", Context.MODE_PRIVATE)
        return sharedPref.getInt("cap_mod", 2)
    }




    private fun onCameraFabClicked() {
        setVisibility(clicked)
        setAnimation(clicked)
        clicked = !clicked
    }
    private fun setVisibility(clicked: Boolean) {
        val edit_btn = findViewById<FloatingActionButton>(R.id.btn_camera)
        val image_btn = findViewById<FloatingActionButton>(R.id.btn_gallery)

        if (!clicked) {
            edit_btn.visibility = View.VISIBLE
            image_btn.visibility = View.VISIBLE
        } else {
            edit_btn.visibility = View.GONE
            image_btn.visibility = View.GONE
        }
    }
    private fun setAnimation(clicked: Boolean) {
        val edit_btn = findViewById<FloatingActionButton>(R.id.btn_camera)
        val image_btn = findViewById<FloatingActionButton>(R.id.btn_gallery)
        val add_btn = findViewById<FloatingActionButton>(R.id.fab)

        if(!clicked) {
            edit_btn.startAnimation(fromBottom)
            image_btn.startAnimation(fromBottom)
            add_btn.startAnimation(rotateOpen)
        }else {
            edit_btn.startAnimation(toBottom)
            image_btn.startAnimation(toBottom)
            add_btn.startAnimation(rotateClose)
        }
    }
    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putSparseParcelableArray(SAVED_STATE_CONTAINER_KEY, savedStateSparseArray)
        outState.putInt(SAVED_STATE_CURRENT_TAB_KEY, currentSelectItemId)
    }
    private fun savedFragmentState(actionId: Int) {
        val currentFragment = supportFragmentManager.findFragmentById(R.id.fl_wrapper)
        if (currentFragment != null) {
            savedStateSparseArray.put(currentSelectItemId,
                supportFragmentManager.saveFragmentInstanceState(currentFragment)
            )
        }
        currentSelectItemId = actionId
    }
    private fun createFragment(fragment: Fragment,actionId: Int) {
        homeFragment = HomeFragment()
        fragment.setInitialSavedState(savedStateSparseArray[actionId])
        supportFragmentManager.beginTransaction()
            .replace(R.id.fl_wrapper, fragment)
            .commit()
    }

    fun getBitmapFromFilePath(filePath: String): Bitmap {
        return BitmapFactory.decodeFile(filePath)
    }

    //FRAGMENT CHANGE
    private fun makeCurrentFragment(Fragment: Fragment,@IdRes actionId: Int) {
        savedFragmentState(actionId)
//        createFragment(Fragment,actionId)
        Fragment.setInitialSavedState(savedStateSparseArray[actionId])

        supportFragmentManager.beginTransaction().apply {
            replace(R.id.fl_wrapper,Fragment)
            commit()
        }
    }

    private fun getPhotoFile(s: String): File {

        val storageDirectory =  getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(s,".jpg",storageDirectory)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        CoroutineScope(Dispatchers.Default).launch {
            Log.d("IMAGE","THREAD: "+ Looper.myLooper().toString())
            Log.d("IMAGE","THREAD: "+ Looper.getMainLooper().toString())
            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH-mm-ss")


            var progressBar = 0

            mFusedLocationClient = LocationServices.getFusedLocationProviderClient(this@Home_main)

            // method to get the location
            getLastLocation()


            Log.d("latitude",lat.toString())

            if (resultCode == AppCompatActivity.RESULT_OK && allPermissionGranted()) {
                if (requestCode == cameraRequestCode || requestCode==100) {
                    var outputUri = makeImageDirectory(dateFormat.format(Date()))
                    Log.d("IMAGE", outputUri.toString())
                    imagePathArrayList.add(outputUri.path.toString())
                    homeFragment.imagePaths.add(outputUri.path.toString())


                    var bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    if(requestCode==100) {
                        val parcelFileDescriptor: ParcelFileDescriptor? =
                            data?.data?.let { contentResolver.openFileDescriptor(it, "r") }
                        val fileDescriptor = parcelFileDescriptor?.fileDescriptor
                        bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor)

                    }
                    processingMod = getProcessingMod()
                    capMod = getCaptureMod()


                    var greymap: HashMap<Any, Bitmap> = hashMapOf(7 to bitmap, 11 to bitmap, 15 to bitmap, 19 to bitmap)

                    val job = CoroutineScope(Dispatchers.Default).launch {
                       if(capMod==2){
                           greymap=makeGray(bitmap)
                       }
                    }
                    runBlocking {
                        job.join()
                    }
                    runOnUiThread {
                        homeFragment.imageRVAdapter.updateList(imagePathArrayList)
                    }
                    progressBar= updateRecyclerView(progressBar)

                    Log.d("IMAGE", "PYTHON SCRIPT ACCESSING")

                    var py = Python.getInstance()
                    var pyObj = py.getModule("myscript")
                    progressBar= updateRecyclerView(progressBar)

                    if (greymap != null) {
                        for (key in greymap.keys) {
                            try {
                                progressBar= updateRecyclerView(progressBar)

                                var imagestr = getStringImage(greymap[key])
                                var obj = pyObj.callAttr("main", imagestr,processingMod)
                                var imgstr = obj.toString()
                                var data =
                                    android.util.Base64.decode(imgstr, android.util.Base64.DEFAULT)
                                var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)

                                Log.d("IMAGE", greymap[key].toString())
                                if (btmp != null) {
                                    var fOut1: OutputStream? = null
                                    val file1 = File(
                                        outputUri,
                                        greymap[key].toString() + "og.png"
                                    )
                                    fOut1 = FileOutputStream(file1)
                                    greymap[key]?.compress(Bitmap.CompressFormat.PNG, 100, fOut1)
                                    var fOut: OutputStream? = null
                                    val file = File(
                                        outputUri,
                                        greymap[key].toString() + ".png"
                                    ) // the File to save , append increasing numeric counter to prevent files from getting overwritten.
                                    fOut = FileOutputStream(file)
                                    Log.d("IMAGE", "NEXT2")

                                    btmp?.compress(Bitmap.CompressFormat.PNG, 100, fOut)
                                    progressBar= updateRecyclerView(progressBar)

//                                    var progressB: ProgressDialog = ProgressDialog(this@Home_main)
//                                    progressB.setTitle("uploading")
//                                    progressB.show()

                                    var ref = storageReference.child("images/"+greymap[key].toString()+".png")

                                    ref.putFile(Uri.fromFile(file)).addOnSuccessListener(OnSuccessListener<UploadTask.TaskSnapshot>{
//                                        progressB.dismiss()
//                                        Toast.makeText(this@Home_main,"UPLOADED",Toast.LENGTH_SHORT).show()
                                    }).addOnFailureListener(OnFailureListener {
//                                        progressB.dismiss()
                                        Toast.makeText(this@Home_main,"UPLOAD FAILED!",Toast.LENGTH_SHORT).show()

                                    })

                                    // the File to save , append increasing numeric counter to prevent files from getting overwritten.
    //                                fOut.flush(); // Not really required
                                    fOut.close();
                                    fOut1.close()

                                }
                                Log.d("IMAGE", "PYTHON SCRIPT Processed")
                                if(capMod==1){
                                    break
                                }

                            } catch (e: IOException) {
                                e.printStackTrace()
                            }

                        }
    //                    greymap=null
                    }//
                    progressBar= updateRecyclerView(90)
                    Log.d("IMAGE","ARRAYLIST REMOVED")

                    imagePathArrayList.remove(outputUri.path.toString())
                    homeFragment.imagePaths.remove(outputUri.path.toString())
                    runOnUiThread {
                        homeFragment.imageRVAdapter.updateList(imagePathArrayList)
                        updateCompletedRecyclerView(outputUri.path.toString())
                    }

                    var map = hashMapOf<String,String>("name" to "A","time" to dateFormat.format(Date()),"latitude" to lat ,"longitude" to lon)
                    firebase.collection("data").add(map).addOnSuccessListener(OnSuccessListener<DocumentReference>{
                        Toast.makeText(this@Home_main,"ATTENDANCE MARKED",Toast.LENGTH_SHORT).show()
                    }).addOnFailureListener(OnFailureListener {
                        Toast.makeText(this@Home_main,"FAILED! TRY AGAIN",Toast.LENGTH_SHORT).show()

                    })
                    runOnUiThread(Runnable {
                        Toast.makeText(
                            this@Home_main,
                            "IMAGE PROCESSED",
                            Toast.LENGTH_SHORT
                        ).show()
                    })

                    Log.d("IMAGE", "PROCESSED")

                }
            }
            else{
                Log.d("IMAGE","Permission Not Granted")
            }

        }
    }




    private fun updateRecyclerView (bar: Int) :Int{

        runOnUiThread {
            homeFragment.imageRVAdapter.update(bar)
            imagePathArrayList
        }
        return bar+10
    }
    private fun updateCompletedRecyclerView (imgfilepath: String) {
        runOnUiThread {
            homeFragment.imageCompletedPaths.add(imgfilepath)
            homeFragment.imageCompletedRVAdapter.updateList(homeFragment.imageCompletedPaths)
            imagePathArrayList
        }
    }
    private fun listImageDirectory(): Array<out File>? {

        var uri = externalMediaDirs.firstOrNull()?.let { mFile->
            File((mFile.toString() + "/"+resources.getString(R.string.app_name)))
        }
        Log.d("IMAGE","DIR+" + uri.toString())
        Log.d("IMAGE","DIR+" + uri?.listFiles().toString())
        var files = uri?.listFiles(FileFilter { it.isDirectory })
        return files
    }
    private fun makeImageDirectory(name: String): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
            Log.d("IMAGE", mFile.toString())
            File((mFile.toString() + "/"+resources.getString(R.string.app_name)), name).apply {
                mkdirs()
            }
        }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }
    private fun makeProcessingImageDirectory(name: String): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
            Log.d("IMAGE", mFile.toString())
            File((mFile.toString() + "/"+"palm"), name).apply {
                mkdirs()
            }
        }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }
    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile->
            File(mFile,resources.getString(R.string.app_name)).apply {
                mkdirs()
            }
        }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir

    }
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == Constants.REQUEST_CODE_PERMISSIONS) {
            Log.d("IMAGE","PERMISSIONS")
            if (allPermissionGranted()) {
                Log.d("IMAGE","GRANTED")

                //our code
                val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
                cameraProviderFuture.addListener({
                    try {
                        val cameraProvider = cameraProviderFuture.get()
    //                        startCameraX(cameraProvider)
                    }catch (e: Exception)
                    {
                        Log.d(Constants.TAG, "Start Camera Failed", e)
                    }
                }, ContextCompat.getMainExecutor(this))
            } else {
                Toast.makeText(this,"Permission not granted", Toast.LENGTH_SHORT).show()
                finish()

            }
        }

//        if (requestCode == GPS_PERMISSION_ID) {
//            if (grantResults.size > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                getLastLocation()
//            }
//        }
    }
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val planeProxy = image.planes[0]
        val buffer: ByteBuffer = planeProxy.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    fun flipBitmapHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    suspend fun makeGray(bitma: Bitmap) : HashMap<Any, Bitmap> {

//        var bi = flipBitmapHorizontally(bitma)
        var rotatebool = false
        if(bitma.width>bitma.height){
            val matrix = Matrix()
            rotatebool= true
//            matrix.postRotate(90f)
//            Log.d("PROCESSING","ROTATE IMAGE")
//            val rotatedBitmap = Bitmap.createBitmap(
//                bitma,
//                0,
//                0,
//                bitma.width,
//                bitma.height,
//                matrix,
//                true
//            )
//            bi = rotatedBitmap
//
        }
        var bitmap = flipBitmapHorizontally(bitma)
//        var bitmap  = bi
        var hands = Hands(this, HandsOptions.builder()
            .setStaticImageMode(true)
            .setMaxNumHands(2)
            .setRunOnGpu(true)
            .build()
        )
        val job = CoroutineScope(Dispatchers.Default).launch {
            hands.send(bitmap)
        }
        job.join()
        var top1: Bitmap = bitmap

        var fingers: HashMap<Any, Bitmap> = hashMapOf(7 to bitmap,11 to bitmap,15 to bitmap,19 to bitmap)

        // Connects MediaPipe Hands solution to the user-defined HandsResultImageView.
        Log.d("IMAGE","Waiting for result")
        hands.setResultListener { handsResult: HandsResult? ->
            Log.d("IMAGE","Waiting for result")
            if (handsResult != null) {
    //                var coords: HashMap<Any,Pair<Array<Double>,Array<Int>>> = hashMapOf(7 to Pair(arrayOf<Double>(-1.0), arrayOf<Int>(-1)),11 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),15 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)),19 to Pair(arrayOf<Double>(-1.0),arrayOf<Int>(-1)))
                for(key in fingers.keys) {
                    Log.d("IMAGE", "Start Cropping")
                    var direction = 0
                    var start_point = arrayOf<Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int) + 1).y) * bitmap.height)).toDouble())
//                    Log.d("MATCHING",key.toString()+ "startpt"+"\t"+start_point.toString())

                    var end_point = arrayOf<Double>(((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).x) * bitmap.width).toDouble(), (((handsResult.multiHandLandmarks().get(0).landmarkList.get((key as Int)).y) * bitmap.height)).toDouble())
//                    Log.d("MATCHING",key.toString()+ "endpt"+"\t"+end_point.toString())

                    var m = (-end_point[1]+start_point[1])/(-end_point[0]+start_point[0]+0.000001)
                    var angle= atan(m) *180/Math.PI
                    var dist_pt = sqrt((end_point[0] - start_point[0]).pow(2.0) + (end_point[1] - start_point[1]).pow(2.0))

                    var mid_point = arrayOf<Double>(((start_point[0] + end_point[0]) / 2), ((start_point[1] + end_point[1]) / 2))
                    var axesy = (dist_pt * 1.6 / 2)
                    Log.d("MATCHING",key.toString()+ " midpt"+"\t"+mid_point[0].toString()+"\t"+mid_point[1].toString())
//                    Log.d("MATCHING",key.toString()+ "axesy"+"\t"+axesy.toString())

                    var half_length_diag = (dist_pt*2.5)/2

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
                    Log.d("MATCHING",key.toString()+ "\t"+axesLength[0].toString()+"\t"+axesLength[1].toString())

                    top1 = Bitmap.createBitmap(bitmap, (mid_point[0] - axesLength[1]).toInt(), (mid_point[1] - axesLength[0]*1.2).toInt(), 2 * axesLength[1].toInt(), (3.2 * axesLength[0]-half_length_diag).toInt())
                    Log.d("MATCHING",key.toString()+ "\t"+top1.getWidth().toString()+"\t"+ top1.getHeight())
                    if(rotatebool){
                        top1 = Bitmap.createBitmap(bitmap, (mid_point[0] - axesLength[1]*0.4).toInt(), (mid_point[1] - axesLength[0]*0.9).toInt(), (1.6 * axesLength[1]).toInt(), (1.8* axesLength[0]).toInt())

                    }
                    if(abs(angle) <45) {
//                        top1 = Bitmap...createBitmap(bitmap, (mid_point[0] - axesLength[1]).toInt(), (mid_point[1] - axesLength[0]*1.2).toInt(), 2 * axesLength[1].toInt(), (3.2 * axesLength[0]-half_length_diag).toInt())

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
    private fun getStringImage(grayBitmap: Bitmap?): String? {
        var baos= ByteArrayOutputStream()
        grayBitmap?.compress(Bitmap.CompressFormat.PNG,100,baos)
        var imgByte = baos.toByteArray()
        var encodedImg = android.util.Base64.encodeToString(imgByte,android.util.Base64.DEFAULT)
        return encodedImg
    }
    @SuppressLint("MissingPermission")
    private fun getLastLocation() {
        var location: Location
        // check if permissions are given
        if (allPermissionGranted()) {

            // check if location is enabled
            if (isLocationEnabled()) {

                // getting last
                // location from
                // FusedLocationClient
                // object
//                mFusedLocationClient!!.lastLocation.addOnCompleteListener(OnCompleteListener { task->
//                    location = task.result
//                    if (location == null) {
//                        requestNewLocationData()
//                    }
//                    else{
//                        lat = location.latitude.toString()
//                        lon = location.longitude.toString()
//                        Log.d("LAtitude",location.latitude.toString())
//                        Log.d("Longitude",location.longitude.toString())
//                        }
//
//                })
            } else {
//                Toast.makeText(this, "Please turn on" + " your location...", Toast.LENGTH_LONG)
//                    .show()
//                val intent: Intent = Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS)
//                startActivity(intent)
            }
        } else {
            // if permissions aren't available,
            // request for permissions
            requestPermissions()
        }
    }
    @SuppressLint("MissingPermission")
    private fun requestNewLocationData() {

        // Initializing LocationRequest
        // object with appropriate methods
        val mLocationRequest = LocationRequest()
        mLocationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY)
        mLocationRequest.setInterval(5)
        mLocationRequest.setFastestInterval(0)
        mLocationRequest.setNumUpdates(1)

        // setting LocationRequest
        // on FusedLocationClient
        mFusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        mFusedLocationClient!!.requestLocationUpdates(
            mLocationRequest,
            mLocationCallback,
            Looper.myLooper()
        )
    }
    private val mLocationCallback: LocationCallback = object : LocationCallback() {
        override fun onLocationResult(locationResult: LocationResult) {
            val mLastLocation = locationResult.lastLocation
            lat = mLastLocation?.latitude.toString()
            lon = mLastLocation?.longitude.toString()
            Log.d("LAtitude",mLastLocation?.latitude.toString())
            Log.d("Longitude",mLastLocation?.longitude.toString())
        }
    }

    // method to request for permissions
    private fun requestPermissions() {


    }

    // method to check
    // if location is enabled
    private fun isLocationEnabled(): Boolean {
        val locationManager: LocationManager = getSystemService(LOCATION_SERVICE) as LocationManager
        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) || locationManager.isProviderEnabled(
            LocationManager.NETWORK_PROVIDER
        )
    }


    private fun allPermissionGranted() =
        Constants.REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(
                baseContext, it
            ) == PackageManager.PERMISSION_GRANTED
        }
    override fun onDestroy() {
        super.onDestroy()
    }
}