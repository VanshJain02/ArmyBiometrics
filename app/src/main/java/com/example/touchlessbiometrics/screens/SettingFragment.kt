package com.example.touchlessbiometrics.screens


import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatDelegate
import androidx.fragment.app.Fragment
import com.example.touchlessbiometrics.R
import com.example.touchlessbiometrics.databinding.FragmentSettingBinding


class SettingFragment : Fragment() {
    private lateinit var binding: FragmentSettingBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

//        var theme_light = view.findViewById<>()

    }
        override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
//        return inflater.inflate(R.layout.fragment_setting, container, false)
           binding = FragmentSettingBinding.inflate(layoutInflater)
            return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.settingsBtnMulti.isChecked = true
        when(getCaptureMod()){
            1-> binding.settingsBtnSingle.isChecked = true
            2-> binding.settingsBtnMulti.isChecked = true
        }
        when(getProcessingMod()){
            1-> binding.settingsBtnMod1.isChecked = true
            2-> binding.settingsBtnMod2.isChecked = true
            3-> binding.settingsBtnMod3.isChecked = true
            4-> binding.settingsBtnMod4.isChecked = true
//            5-> binding.settingsBtnMod5.isChecked = true
            6-> binding.settingsBtnMod6.isChecked = true


            else ->{ binding.settingsBtnMod1.isChecked = true }
        }

        binding.settingsThemeGrp.setOnCheckedChangeListener{group,checkedId ->
            if (R.id.settings_btn_single == checkedId) {
//                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
                val sharedPref = context?.getSharedPreferences("capturemode", Context.MODE_PRIVATE)
                val editor = sharedPref?.edit()
                editor?.putInt("cap_mod",1)
                editor?.apply()
            }

            else{
                val sharedPref = context?.getSharedPreferences("capturemode", Context.MODE_PRIVATE)
                val editor = sharedPref?.edit()
                editor?.putInt("cap_mod",2)
                editor?.apply()

//                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
            }
        }
        binding.settingsBtnMod1.setOnClickListener {
            binding.settingsBtnMod2.isChecked = false
            binding.settingsBtnMod3.isChecked = false
            binding.settingsBtnMod4.isChecked = false
//            binding.settingsBtnMod5.isChecked = false
            binding.settingsBtnMod6.isChecked = false

            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
            val editor = sharedPref?.edit()
            editor?.putInt("mod",1)
            editor?.apply()
        }
        binding.settingsBtnMod2.setOnClickListener {
            binding.settingsBtnMod1.isChecked = false
            binding.settingsBtnMod3.isChecked = false
            binding.settingsBtnMod4.isChecked = false
//            binding.settingsBtnMod5.isChecked = false
            binding.settingsBtnMod6.isChecked = false

            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
            val editor = sharedPref?.edit()
            editor?.putInt("mod",2)
            editor?.apply()
        }
        binding.settingsBtnMod3.setOnClickListener {
            binding.settingsBtnMod2.isChecked = false
            binding.settingsBtnMod1.isChecked = false
            binding.settingsBtnMod4.isChecked = false
//            binding.settingsBtnMod5.isChecked = false
            binding.settingsBtnMod6.isChecked = false


            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
            val editor = sharedPref?.edit()
            editor?.putInt("mod",3)
            editor?.apply()
        }
        binding.settingsBtnMod4.setOnClickListener {
            binding.settingsBtnMod2.isChecked = false
            binding.settingsBtnMod3.isChecked = false
            binding.settingsBtnMod1.isChecked = false
//            binding.settingsBtnMod5.isChecked = false
            binding.settingsBtnMod6.isChecked = false

            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
            val editor = sharedPref?.edit()
            editor?.putInt("mod",4)
            editor?.apply()
        }
//        binding.settingsBtnMod5.setOnClickListener {
//            binding.settingsBtnMod2.isChecked = false
//            binding.settingsBtnMod3.isChecked = false
//            binding.settingsBtnMod1.isChecked = false
//            binding.settingsBtnMod4.isChecked = false
//            binding.settingsBtnMod6.isChecked = false
//
//            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
//            val editor = sharedPref?.edit()
//            editor?.putInt("mod",5)
//            editor?.apply()
//        }
        binding.settingsBtnMod6.setOnClickListener {
            binding.settingsBtnMod2.isChecked = false
            binding.settingsBtnMod3.isChecked = false
            binding.settingsBtnMod1.isChecked = false
            binding.settingsBtnMod4.isChecked = false
//            binding.settingsBtnMod5.isChecked = false
            val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
            val editor = sharedPref?.edit()
            editor?.putInt("mod",6)
            editor?.apply()
        }
//        binding.settingsProcessingGrp.setOnCheckedChangeListener(group,)


    }
    private fun getProcessingMod(): Int?{
        val sharedPref = context?.getSharedPreferences("processing", Context.MODE_PRIVATE)
        return sharedPref?.getInt("mod", 1)
    }
    private fun getCaptureMod(): Int?{
        val sharedPref = context?.getSharedPreferences("capturemode", Context.MODE_PRIVATE)
        return sharedPref?.getInt("cap_mod", 1)
    }
}