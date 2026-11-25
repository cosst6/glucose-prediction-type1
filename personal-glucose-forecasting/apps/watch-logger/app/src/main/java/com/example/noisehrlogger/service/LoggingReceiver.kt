package com.example.noisehrlogger.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

class LoggingReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        val actionReceived = intent?.action
        Log.d("LoggingReceiver", "Received broadcast: $actionReceived")
        
        when (actionReceived) {
            ACTION_LOG_TICK -> {
                val serviceIntent = Intent(context, LoggingService::class.java).apply {
                    action = LoggingService.ACTION_MEASURE_TICK
                }
                context?.startService(serviceIntent)
            }
            Intent.ACTION_BOOT_COMPLETED, "android.intent.action.QUICKBOOT_POWERON" -> {
                val prefs = context?.getSharedPreferences("LoggingPrefs", Context.MODE_PRIVATE)
                val isRecording = prefs?.getBoolean("isRecording", false) ?: false
                if (isRecording) {
                    Log.i("LoggingReceiver", "Detected boot, restarting logging service...")
                    val serviceIntent = Intent(context, LoggingService::class.java)
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                        context.startForegroundService(serviceIntent)
                    } else {
                        context.startService(serviceIntent)
                    }
                }
            }
        }
    }

    companion object {
        const val ACTION_LOG_TICK = "com.example.noisehrlogger.ACTION_LOG_TICK"
    }
}
