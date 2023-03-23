package com.example.touchlessbiometrics


import android.annotation.SuppressLint
import android.media.ImageReader
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File
import java.io.FileFilter


class GalleryActivity : AppCompatActivity() {
    private lateinit var imagePaths: ArrayList<String>
    private lateinit var imagesRV: RecyclerView
    private lateinit var outputDirectory: File

    private lateinit var imageRVAdapter: RecyclerViewAdapter
    var imgFileLocation: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gallery)

        imgFileLocation = intent.getStringExtra("imgLoc")
        Log.d("IMAGE","FILE"+ imgFileLocation.toString())

        imagePaths = ArrayList()
        imagesRV = findViewById(R.id.idIVImage)
//        getImagePath()
        listImageDirectory(imgFileLocation)
        prepareRecyclerView()
//        Log.d("IMAGE",imagePaths.toString())



    }
    private fun prepareRecyclerView() {


        // in this method we are preparing our recycler view.

        // on below line we are initializing our adapter class.
        Log.d("IMAGE",imagePaths.toString())
        imageRVAdapter = RecyclerViewAdapter(this@GalleryActivity, imagePaths)


        // on below line we are creating a new grid layout manager.
        val manager = GridLayoutManager(this@GalleryActivity, 4)


        // on below line we are setting layout

        // manager and adapter to our recycler view.
        imagesRV.layoutManager = manager
        imagesRV.adapter = imageRVAdapter
    }
    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile ->
            File(mFile, resources.getString(R.string.app_name)).apply {
                mkdirs()
            }
        }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir

    }

    private fun listImageDirectory(filename: String?): Array<out File>? {
        var uri =externalMediaDirs?.firstOrNull()?.let { mFile->
            File(imgFileLocation)
//            File((mFile.toString() + "/"+resources.getString(R.string.app_name)+"/"+filename))
//                    +"/"+filename))
        }
        Log.d("IMAGE","DIR+" + uri.toString())
//FileFilter { it.isFile }
        var files = uri?.listFiles()
        Log.d("IMAGE","DIR+" + files.toString())
        if (files != null) {
            for(i in files){
                imagePaths.add(i.path.toString())
            }
        }
        return files
//        if (files != null) {
//            for(i in files){
//                Log.d("IMAGE","FILES+" + i.name.toString())
//
//            }
//        }

    }


    private fun getImagePath() {

        // in this method we are adding all our image paths
        // in our arraylist which we have created.
        // on below line we are checking if the device is having an sd card or not.
        val isSDPresent = Environment.getExternalStorageState() == Environment.MEDIA_MOUNTED
        if (isSDPresent) {
            // if the sd card is present we are creating a new list in
            // which we are getting our images data with their ids.
//            outputDirectory = imgFileLocation
            val columns = arrayOf(MediaStore.Images.Media.DATA, MediaStore.Images.Media._ID)
            // on below line we are creating a new
            // string to order our images by string.
            val orderBy = MediaStore.Images.Media._ID
            // this method will stores all the images
            // from the gallery in Cursor
            val cursor = contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                columns,
                null,
                null,
                orderBy
            )
            // below line is to get total number of images
            val count = cursor!!.count
            // on below line we are running a loop to add
            // the image file path in our array list.
            for (i in 0 until count) {
                // on below line we are moving our cursor position
                cursor.moveToPosition(i)
                // on below line we are getting image file path
                val dataColumnIndex = cursor.getColumnIndex(MediaStore.Images.Media.DATA)

                // after that we are getting the image file path
                // and adding that path in our array list.
//                Log.d("IMAGE",cursor.getString(dataColumnIndex).toString())
                imagePaths.add(cursor.getString(dataColumnIndex))
            }
//            imageRVAdapter.notifyDataSetChanged()
            // after adding the data to our
            // array list we are closing our cursor.
            cursor.close()
        }
    }
}