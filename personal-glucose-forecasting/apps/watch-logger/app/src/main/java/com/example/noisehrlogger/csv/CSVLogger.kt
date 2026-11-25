package com.example.noisehrlogger.csv

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CSVLogger(
    private val context: Context
) {

    private val TAG = "CSVLogger"
    private var currentFile: File? = null
    private var sessionStartTime: Long = 0
    private var deviceName: String = ""
    private var readingsCount = 0

    fun startNewSession(deviceName: String): String {
        this.deviceName = deviceName
        this.readingsCount = 0
        this.sessionStartTime = System.currentTimeMillis()

        // New naming: NoiseHR_Session_Aug22_17-45.csv
        val dateStr = SimpleDateFormat("MMMdd", Locale.getDefault()).format(Date(sessionStartTime))
        val timeStr = SimpleDateFormat("HH-mm", Locale.getDefault()).format(Date(sessionStartTime))
        val fileName = "NoiseHR_Session_${dateStr}_$timeStr.csv"

        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        val file = File(dir, fileName)

        val headerTimestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(sessionStartTime))

        file.writeText("Device Name,$deviceName\n")
        file.writeText("Session Start,$headerTimestamp\n")
        file.appendText("\n")
        file.appendText("Timestamp,heart_rate,steps,distance_m,calories\n")

        currentFile = file
        Log.d(TAG, "Started new session: ${file.absolutePath}")
        return file.absolutePath
    }

    fun resumeSession(filePath: String, deviceName: String) {
        val file = File(filePath)
        if (file.exists()) {
            this.currentFile = file
            this.deviceName = deviceName
            this.sessionStartTime = file.lastModified()
            this.readingsCount = 1 
            Log.i(TAG, "Resumed session from: $filePath")
        } else {
            Log.w(TAG, "Failed to resume session: File does not exist: $filePath")
            startNewSession(deviceName)
        }
    }

    @Synchronized
    fun logHeartRate(bpm: Int): String {
        // Keeping for backward compatibility if needed, but we should use logData
        return logData(bpm, 0, 0, 0)
    }

    @Synchronized
    fun logData(bpm: Int, steps: Int, distance: Int, calories: Int): String {
        val file = currentFile ?: return ""
        readingsCount++

        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val line = "$timestamp,$bpm,$steps,$distance,$calories\n"
        file.appendText(line)

        Log.d(TAG, "Logged: HR=$bpm, Steps=$steps, Dist=$distance, Cal=$calories at $timestamp")
        return "$bpm BPM, $steps steps at ${SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())}"
    }

    fun finalizeSession(): String? {
        val file = currentFile ?: return null
        if (readingsCount == 0) {
            file.delete()
            currentFile = null
            return null
        }
        return file.absolutePath
    }

    fun getFile(): File? {
        return currentFile
    }
    
    fun getDisplayPath(): String {
        return currentFile?.name ?: "No active file"
    }

    fun getReadingsCount(): Int = readingsCount

    fun getAllLogs(): List<File> {
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: return emptyList()
        return dir.listFiles { file -> 
            file.isFile && file.name.startsWith("NoiseHR_Session") && file.extension == "csv"
        }?.toList() ?: emptyList()
    }
}
