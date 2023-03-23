package com.example.touchlessbiometrics.screens

import android.content.Intent
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.touchlessbiometrics.*
import com.example.touchlessbiometrics.databinding.FragmentHomeBinding
import kotlinx.coroutines.*
import java.io.File
import java.io.FileFilter


class HomeFragment : Fragment() {

    private lateinit var binding: FragmentHomeBinding



    public lateinit var imageCompletedRVAdapter: Home_CompletedRecyclerViewAdapter
    public var imagePaths: ArrayList<String> = ArrayList()
    public var imageCompletedPaths: ArrayList<String> = ArrayList()
    public var imageRVAdapter: Home_RecyclerViewAdapter = Home_RecyclerViewAdapter(context,imagePaths,0)

    private var bar = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        // Inflate the layout for this fragment
        binding = FragmentHomeBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if(savedInstanceState!= null){
            Log.d("IMAGE","INSTANCE USED")

//            imagePaths = savedInstanceState?.getStringArrayList("key") as ArrayList<String>
        }
        else{
            listImageDirectory()
        }


        imageRVAdapter = Home_RecyclerViewAdapter(context,imagePaths,bar)
        val manager = LinearLayoutManager(context)
        imageRVAdapter.notifyDataSetChanged()
        binding.homeRecyclerView.layoutManager = manager
        binding.homeRecyclerView.adapter = imageRVAdapter

        Log.d("IMAGE","HomeFragment")

        imageCompletedPaths.sortByDescending{it}
        imageCompletedRVAdapter = Home_CompletedRecyclerViewAdapter(context,imageCompletedPaths,bar)
        val manage = LinearLayoutManager(context)
        Log.d("IMAGE",imagePaths.toString())
        Log.d("IMAGE",imageCompletedPaths.toString())

        binding.homeBtnMatching.setOnClickListener{
            val matching = Intent(context, PAGE_Matching::class.java)
            context?.startActivity(matching)
        }

        binding.homeCompletedRecyclerView.layoutManager = manage
        binding.homeCompletedRecyclerView.adapter = imageCompletedRVAdapter

    }

    private fun listImageDirectory(){
        var uri = context?.externalMediaDirs?.firstOrNull()?.let { mFile->
            File((mFile.toString() + "/"+resources.getString(R.string.app_name)))
        }
        Log.d("IMAGE","DIR+" + uri.toString())
        Log.d("IMAGE","DIR+" + uri?.listFiles().toString())
        var files = uri?.listFiles(FileFilter { it.isDirectory })
        if (files != null) {
            for(i in files){
                imageCompletedPaths.add(i.toString())
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        Log.d("IMAGE","INSTANCE SAVED")
        outState.putStringArrayList("key",imagePaths)
        outState.putStringArrayList("key1",imageCompletedPaths)
    }


}