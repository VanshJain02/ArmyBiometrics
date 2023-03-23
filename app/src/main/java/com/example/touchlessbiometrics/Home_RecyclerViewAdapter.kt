package com.example.touchlessbiometrics

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File


class Home_RecyclerViewAdapter(private var context: Context?,private var imagePathArrayList: ArrayList<String>,private var bar: Int) : RecyclerView.Adapter<Home_RecyclerViewAdapter.Home_RecyclerViewHolder>() {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Home_RecyclerViewHolder {

        // Inflate Layout in this method which we have created.
        val view = LayoutInflater.from(parent.context).inflate(R.layout.homecarditems, parent, false)

        return Home_RecyclerViewHolder(view)
    }

    fun update(bar: Int) {
        this.bar=bar
        notifyDataSetChanged();
    }
    fun updateList(imagePathArrayList: ArrayList<String>) {
        this.imagePathArrayList= imagePathArrayList
        notifyDataSetChanged();
    }

    override fun onBindViewHolder(holder: Home_RecyclerViewHolder, position: Int) {

        // on below line we are getting the file from the

        // path which we have stored in our list.
        val imgFile = File(imagePathArrayList[position])

        // on below line we are checking if the file exists or not.
        if (imgFile.exists()) {
            holder.filename.text = imgFile.name
            holder.progresspercent.text = (bar).toString()
            holder.progressbar.progress = bar

        }
    }

    override fun getItemCount(): Int {

        // this method returns

        // the size of recyclerview
        return imagePathArrayList.size
    }

    // View Holder Class to handle Recycler View.
    class Home_RecyclerViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {

        var filename: TextView = itemView.findViewById(R.id.home_file_name)
        var progresspercent: TextView = itemView.findViewById(R.id.home_progress_percent)
        var progressbar: ProgressBar = itemView.findViewById(R.id.home_progress_bar)

    }
}