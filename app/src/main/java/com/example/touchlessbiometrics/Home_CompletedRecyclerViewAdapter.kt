package com.example.touchlessbiometrics

import android.content.Context
import android.content.Intent
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.squareup.picasso.Picasso
import java.io.File


class Home_CompletedRecyclerViewAdapter(private var context: Context?, private var imagefilePathArrayList: ArrayList<String>, private var bar: Int) : RecyclerView.Adapter<Home_CompletedRecyclerViewAdapter.HomeCompleted_RecyclerViewHolder>() {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HomeCompleted_RecyclerViewHolder {

        // Inflate Layout in this method which we have created.
        val view = LayoutInflater.from(parent.context).inflate(R.layout.homecompletedcarditems, parent, false)

        return HomeCompleted_RecyclerViewHolder(view)
    }

    fun update(bar: Int) {
        this.bar=bar
        notifyDataSetChanged();
    }
    fun updateList(imagefilePathArrayList: ArrayList<String>) {
        imagefilePathArrayList.sortByDescending{it}
        this.imagefilePathArrayList= imagefilePathArrayList
        notifyDataSetChanged();
    }

    fun deleteDirectory(file: File) {
        if (file.isDirectory) {
            val contents = file.listFiles()
            for (f in contents) {
                deleteDirectory(f)
            }
        }
        file.delete()
    }

    override fun onBindViewHolder(holder: HomeCompleted_RecyclerViewHolder, position: Int) {

        // on below line we are getting the file from the

        // path which we have stored in our list.
        val imgFile = File(imagefilePathArrayList[position])

        // on below line we are checking if the file exists or not.
        if (imgFile.exists()) {
            // if the file exists then we are displaying that file in our image view using picasso library.
//            Picasso.get().load(imgFile).placeholder(R.drawable.ic_launcher_background).into(holder.imageIV)
            // on below line we are adding click listener to our item of recycler view.

            holder.filename.text = imgFile.name
//
            holder.itemView.setOnClickListener { // inside on click listener we are creating a new intent
                val i = Intent(context, GalleryActivity::class.java)
                // on below line we are passing the image path to our new activity.
                i.putExtra("imgLoc", imagefilePathArrayList[position])
                // at last we are starting our activity.
                context?.startActivity(i)
            }
            holder.itemView.setOnLongClickListener{
                Toast.makeText(context,"Long Pressed",Toast.LENGTH_SHORT).show()
                val directory = File(imagefilePathArrayList[position])
                imagefilePathArrayList.remove(imagefilePathArrayList[position])
                deleteDirectory(directory)
                directory.delete()
                notifyDataSetChanged()
                true
            }
        }
    }



    override fun getItemCount(): Int {

        // this method returns

        // the size of recyclerview
        return imagefilePathArrayList.size
    }

    // View Holder Class to handle Recycler View.
    class HomeCompleted_RecyclerViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {

        var filename: TextView = itemView.findViewById(R.id.home_completed_file_name)

    }
}