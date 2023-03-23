package com.example.touchlessbiometrics

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.text.Html
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.viewpager.widget.ViewPager
import androidx.viewpager.widget.ViewPager.OnPageChangeListener

class onboarding : AppCompatActivity() {
    var mSLideViewPager: ViewPager? = null
    var mDotLayout: LinearLayout? = null
    var viewPagerAdapter: onboarding_adapter? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_onboarding)
        supportActionBar?.hide()

        var backbtn = findViewById<Button>(R.id.backbtn)
        var nextbtn = findViewById<Button>(R.id.nextbtn)
        var skipbtn = findViewById<Button>(R.id.skipButton)

        backbtn.setOnClickListener(View.OnClickListener {
            if (getitem(0) > 0) {
                mSLideViewPager!!.setCurrentItem(getitem(-1), true)
            }
        })

        nextbtn.setOnClickListener(View.OnClickListener {
            if (getitem(0) < 3){
                mSLideViewPager!!.setCurrentItem(getitem(1), true)
            } else {
                val sharedPref = getSharedPreferences("onBoarding", Context.MODE_PRIVATE)
                val editor = sharedPref.edit()
                editor.putBoolean("Finished", true)
                editor.apply()
                val i: Intent = Intent(this@onboarding, Home_main::class.java)
                startActivity(i)
                finish()
            }
        })

        skipbtn.setOnClickListener(View.OnClickListener {
            val sharedPref = getSharedPreferences("onBoarding", Context.MODE_PRIVATE)
            val editor = sharedPref.edit()
            editor.putBoolean("Finished", true)
            editor.apply()
            val i: Intent = Intent(this@onboarding, Home_main::class.java)
            startActivity(i)
            finish()
        })

        mSLideViewPager = findViewById<View>(R.id.slideViewPager) as ViewPager
        mDotLayout = findViewById<View>(R.id.indicator_layout) as LinearLayout

        viewPagerAdapter = onboarding_adapter(this)

        mSLideViewPager!!.adapter = viewPagerAdapter

        setUpindicator(0)
        mSLideViewPager!!.addOnPageChangeListener(viewListener)
    }

    fun setUpindicator(position: Int) {
        var dots = arrayOfNulls<TextView>(4)
        mDotLayout!!.removeAllViews()
        for (i in dots.indices) {
            dots[i] = TextView(this)
            dots[i]?.text = Html.fromHtml("&#8226")
            dots[i]?.textSize = 35f
            dots[i]?.setTextColor(resources.getColor(R.color.inactive, applicationContext.theme))
            mDotLayout!!.addView(dots[i])
        }
        dots[position]?.setTextColor(resources.getColor(R.color.active, applicationContext.theme))
    }

    var viewListener: OnPageChangeListener = object : OnPageChangeListener {
        override fun onPageScrolled(
            position: Int,
            positionOffset: Float,
            positionOffsetPixels: Int
        ) {
        }

        override fun onPageSelected(position: Int) {
            setUpindicator(position)
//            if (position > 0) {
//                backbtn!!.visibility = View.VISIBLE
//            } else {
//                backbtn!!.visibility = View.INVISIBLE
//            }
        }

        override fun onPageScrollStateChanged(state: Int) {}
    }

    private fun getitem(i: Int): Int {
        return mSLideViewPager!!.currentItem + i
    }
}