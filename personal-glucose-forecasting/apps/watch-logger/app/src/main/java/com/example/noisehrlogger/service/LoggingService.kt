package com.example.noisehrlogger.service

import android.annotation.SuppressLint
import android.app.AlarmManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.example.noisehrlogger.MainActivity
import com.example.noisehrlogger.ble.ActivityManager
import com.example.noisehrlogger.ble.BLEManager
import com.example.noisehrlogger.ble.HeartRateManager
import com.example.noisehrlogger.csv.CSVLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class LoggingService : Service() {

    private val TAG = "LoggingService"
    private val CHANNEL_ID = "HeartRateLoggingChannel"
    private val ALARM_CHANNEL_ID = "HeartRateAlarmChannel"
    private val NOTIFICATION_ID = 1
    private val ALARM_NOTIFICATION_ID = 999
    private val PREFS_NAME = "LoggingPrefs"

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    private lateinit var manager: BLEManager
    private lateinit var heartRateManager: HeartRateManager
    private lateinit var activityManager: ActivityManager
    private lateinit var csvLogger: CSVLogger
    
    private var connectionJob: Job? = null
    private var isServiceStarted = false
    private var wakeLock: PowerManager.WakeLock? = null
    
    private var lastSuccessfulReadingTime: Long = 0
    private var lastAlarmTime: Long = 0
    private var lastDisconnectTime: Long = 0

    private var windowTimerJob: Job? = null

    // Aggregation Buffers for 30s window
    private var hrSum: Long = 0
    private var hrCount: Int = 0
    private var latestHeartRate: Int = 0
    private var latestSteps: Int = 0
    private var latestDistance: Int = 0
    private var latestCalories: Int = 0

    // Prev values for Delta calculation
    private var prevLoggedSteps: Int = 0
    private var prevLoggedDistance: Int = 0
    private var prevLoggedCalories: Int = 0

    private val bluetoothReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == BluetoothAdapter.ACTION_STATE_CHANGED) {
                val state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
                when (state) {
                    BluetoothAdapter.STATE_ON -> {
                        Log.i(TAG, "Bluetooth turned ON, checking for recovery")
                        if (LoggingState.isRecording.value && !LoggingState.isConnected.value) {
                            attemptRecovery()
                        }
                    }
                    BluetoothAdapter.STATE_OFF -> {
                        Log.w(TAG, "Bluetooth turned OFF")
                        LoggingState.setError("Bluetooth is disabled")
                        LoggingState.setConnected(false)
                        if (LoggingState.isRecording.value) {
                            alertUser("CRITICAL: Bluetooth was turned OFF!")
                        }
                    }
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service onCreate")
        
        manager = BLEManager(applicationContext)
        heartRateManager = HeartRateManager(manager)
        activityManager = ActivityManager(manager)
        csvLogger = CSVLogger(applicationContext)
        
        setupCallbacks()
        createNotificationChannels()
        
        registerReceiver(bluetoothReceiver, IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED))
    }

    private fun setupCallbacks() {
        manager.onConnected = {
            Log.i(TAG, "Connected to device")
            LoggingState.setConnected(true)
            LoggingState.setError(null)
            LoggingState.setFilePath(csvLogger.getDisplayPath())
            
            // Sequential enablement to avoid GATT_BUSY errors
            serviceScope.launch {
                Log.d(TAG, "Enabling Heart Rate notifications...")
                heartRateManager.enableHeartRateNotifications()
                delay(1200) // Delay between descriptor writes
                Log.d(TAG, "Enabling Activity notifications...")
                activityManager.enableActivityNotifications()
            }
        }
        
        manager.onServicesReady = {
            // Trigger onConnected after services are ready
            manager.onConnected?.invoke()
        }

        manager.onDisconnected = { status ->
            Log.w(TAG, "Disconnected from device. Status: $status")
            LoggingState.setConnected(false)
            lastDisconnectTime = System.currentTimeMillis()
            
            val errorMsg = when (status) {
                133 -> "GATT Error 133 (Watch busy)"
                8 -> "Connection timeout (Out of range)"
                else -> "Disconnected ($status)"
            }
            
            if (LoggingState.isRecording.value) {
                LoggingState.setError("Reconnecting... ($errorMsg)")
                // Cooldown: wait at least 5 seconds before retrying to prevent oscillation
                serviceScope.launch {
                    delay(5000)
                    attemptRecovery()
                }
                if (System.currentTimeMillis() - lastSuccessfulReadingTime > 2 * 60 * 1000L) {
                    alertUser("WARNING: Watch disconnected for > 2 mins!")
                }
            }
        }

        heartRateManager.onHeartRateChanged = { bpm ->
            lastSuccessfulReadingTime = System.currentTimeMillis()
            latestHeartRate = bpm
            
            // Add to 30s average buffer
            hrSum += bpm
            hrCount++
            
            LoggingState.setHeartRate(bpm)
            updateNotification("HR: $bpm BPM | Steps: $latestSteps")
        }

        activityManager.onActivityDataChanged = { data ->
            lastSuccessfulReadingTime = System.currentTimeMillis()
            latestSteps = data.steps
            latestDistance = data.distanceMeters
            latestCalories = data.calories
            
            LoggingState.setActivityData(data.steps, data.distanceMeters, data.calories)
            
            // Live Delta for UI
            LoggingState.setLiveDeltas(
                data.steps - prevLoggedSteps,
                data.distanceMeters - prevLoggedDistance,
                data.calories - prevLoggedCalories
            )
            
            updateNotification("HR: $latestHeartRate BPM | Steps: $latestSteps")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        
        if (intent == null) {
            resumeFromPersistence()
            return START_STICKY
        }

        val deviceAddress = intent.getStringExtra("EXTRA_DEVICE_ADDRESS")
        val deviceName = intent.getStringExtra("EXTRA_DEVICE_NAME") ?: "Unknown Device"

        when (action) {
            ACTION_CONNECT -> {
                saveConnectionInfo(deviceAddress, deviceName)
                if (deviceAddress != null) {
                    connectToDevice(deviceAddress, deviceName)
                }
            }
            ACTION_START -> {
                if (!isServiceStarted) {
                    acquireWakeLock()
                    startForeground(NOTIFICATION_ID, createNotification("Recording activity data..."))
                    isServiceStarted = true
                    saveRecordingState(true)
                    startRecording(deviceName)
                }
            }
            ACTION_MEASURE_TICK -> {
                if (LoggingState.isRecording.value) {
                    flushAggregatedData()
                    watchdogCheck()
                    scheduleNextWatchdog()
                }
            }
            ACTION_STOP -> {
                saveRecordingState(false)
                stopRecordingAndService()
            }
        }

        return START_STICKY
    }

    private fun flushAggregatedData() {
        if (!LoggingState.isRecording.value) return

        // Calculate average HR for this 30s window
        val avgHr = if (hrCount > 0) (hrSum / hrCount).toInt() else latestHeartRate
        
        // Calculate Deltas for this window (respecting Ignore flags)
        val deltaSteps = if (LoggingState.isIgnoringSteps.value) 0 else latestSteps - prevLoggedSteps
        val deltaDist = if (LoggingState.isIgnoringDistance.value) 0 else latestDistance - prevLoggedDistance
        val deltaCal = if (LoggingState.isIgnoringCalories.value) 0 else latestCalories - prevLoggedCalories
        
        Log.d(TAG, "Flushing 30s delta: AvgHR=$avgHr, dSteps=$deltaSteps, dDist=$deltaDist, dCal=$deltaCal")
        
        val logEntry = csvLogger.logData(
            avgHr,
            Math.max(0, deltaSteps),
            Math.max(0, deltaDist),
            Math.max(0, deltaCal)
        )
        LoggingState.setLastLogEntry(logEntry)
        
        // Update prev values for next 30s
        prevLoggedSteps = latestSteps
        prevLoggedDistance = latestDistance
        prevLoggedCalories = latestCalories
        
        csvLogger.getFile()?.absolutePath?.let { saveCurrentSessionFile(it) }

        // Reset buffers for next 30s
        hrSum = 0
        hrCount = 0
    }

    private fun watchdogCheck() {
        if (!LoggingState.isConnected.value && LoggingState.isRecording.value) {
            val now = System.currentTimeMillis()
            if (now - lastDisconnectTime > 10000) { // Only recover if disconnected for > 10s
                attemptRecovery()
            }
        }
        
        val now = System.currentTimeMillis()
        if (LoggingState.isRecording.value && 
            now - lastSuccessfulReadingTime > 2 * 60 * 1000L) {
            LoggingState.setError("NO DATA RECEIVED FOR 2 MINS!")
            alertUser("CRITICAL: No activity data for 2 minutes!")
        }
    }

    private fun alertUser(message: String) {
        val now = System.currentTimeMillis()
        if (now - lastAlarmTime < 60 * 1000L) return
        lastAlarmTime = now

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE)
        val alarmSound = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
        
        val builder = NotificationCompat.Builder(this, ALARM_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("Logging Alert!")
            .setContentText(message)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setSound(alarmSound)
            .setVibrate(longArrayOf(0, 500, 200, 500))
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)

        notificationManager.notify(ALARM_NOTIFICATION_ID, builder.build())
    }

    private fun scheduleNextWatchdog() {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(this, LoggingReceiver::class.java).apply {
            action = LoggingReceiver.ACTION_LOG_TICK
        }
        val pendingIntent = PendingIntent.getBroadcast(this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val triggerAtMillis = System.currentTimeMillis() + 30 * 1000L
        
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent)
            } else {
                alarmManager.setExact(AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent)
            }
        } catch (e: SecurityException) {
            alarmManager.set(AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent)
        }
    }

    private fun cancelWatchdog() {
        val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(this, LoggingReceiver::class.java).apply {
            action = LoggingReceiver.ACTION_LOG_TICK
        }
        val pendingIntent = PendingIntent.getBroadcast(this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        alarmManager.cancel(pendingIntent)
    }

    private fun resumeFromPersistence() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val isRecording = prefs.getBoolean("isRecording", false)
        val address = prefs.getString("deviceAddress", null)
        val name = prefs.getString("deviceName", "Unknown Device") ?: "Unknown Device"
        val sessionFile = prefs.getString("currentSessionFile", null)

        if (isRecording && address != null) {
            acquireWakeLock()
            startForeground(NOTIFICATION_ID, createNotification("Resuming recording..."))
            isServiceStarted = true
            if (sessionFile != null) csvLogger.resumeSession(sessionFile, name)
            LoggingState.setRecording(true)
            LoggingState.setFilePath(csvLogger.getDisplayPath())
            LoggingState.setConnected(false, name)
            connectToDevice(address, name)
            lastSuccessfulReadingTime = System.currentTimeMillis()
            scheduleNextWatchdog()
        }
    }

    private fun saveConnectionInfo(address: String?, name: String) {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().apply {
            putString("deviceAddress", address)
            putString("deviceName", name)
            apply()
        }
    }

    private fun saveRecordingState(isRecording: Boolean) {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().putBoolean("isRecording", isRecording).apply()
    }

    private fun saveCurrentSessionFile(path: String) {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().putString("currentSessionFile", path).apply()
    }

    private fun attemptRecovery() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val address = prefs.getString("deviceAddress", null)
        val name = prefs.getString("deviceName", "Unknown Device") ?: "Unknown Device"
        if (address != null && connectionJob?.isActive != true) connectToDevice(address, name)
    }

    private fun connectToDevice(address: String, name: String) {
        connectionJob?.cancel()
        connectionJob = serviceScope.launch {
            try {
                val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
                val device = bluetoothManager.adapter.getRemoteDevice(address)
                
                LoggingState.setConnected(false, name)
                
                // --- STAGE 1: Standard Connection ---
                Log.i(TAG, "Stage 1: Attempting standard connection to $name")
                LoggingState.setError("Connecting (Stage 1/4)...")
                manager.connect(device, useLeTransport = false)
                if (waitForConnection(10000)) return@launch

                // --- STAGE 2: LE Transport Connection ---
                Log.i(TAG, "Stage 2: Attempting LE transport connection to $name")
                LoggingState.setError("Connecting (Stage 2/4)...")
                manager.close() // Full reset
                delay(2000)
                manager.connect(device, useLeTransport = true)
                if (waitForConnection(12000)) return@launch

                // --- STAGE 3: Cache Refresh Recovery ---
                Log.i(TAG, "Stage 3: Attempting connection with cache refresh for $name")
                LoggingState.setError("Refreshing Cache (Stage 3/4)...")
                manager.close()
                delay(3000)
                
                // Connect briefly to get a GATT object for refresh
                manager.connect(device, useLeTransport = true)
                delay(1000)
                val refreshed = manager.refreshDeviceCache()
                Log.d(TAG, "Cache refresh requested: $refreshed")
                
                manager.close()
                delay(2000)
                
                manager.connect(device, useLeTransport = true)
                if (waitForConnection(15000)) return@launch

                // --- STAGE 4: Deep Reset & Rescan ---
                Log.i(TAG, "Stage 4: Deep reset & rescan recovery for $name")
                LoggingState.setError("Deep Recovery (Stage 4/4)...")
                manager.close()
                delay(10000) // Longer rest for BT stack
                
                // We don't implement a full scan here to keep it simple, 
                // but a fresh TRANSPORT_LE attempt after a long delay often works.
                manager.connect(device, useLeTransport = true, autoConnect = true)
                if (waitForConnection(20000)) {
                    Log.i(TAG, "Recovery successful after deep reset")
                    return@launch
                }

                Log.e(TAG, "All reconnection stages failed for $name")
                LoggingState.setError("Reconnection failed. Try toggling Bluetooth.")
            } catch (e: Exception) {
                Log.e(TAG, "Connection error: ${e.message}", e)
                LoggingState.setError("Connection error: ${e.message}")
            }
        }
    }

    private suspend fun waitForConnection(timeoutMs: Long): Boolean {
        val start = System.currentTimeMillis()
        while (System.currentTimeMillis() - start < timeoutMs) {
            if (LoggingState.isConnected.value) return true
            delay(1000)
        }
        return false
    }

    private fun startRecording(deviceName: String) {
        LoggingState.setRecording(true)
        
        // Reset buffers and baselines for the new session
        hrSum = 0
        hrCount = 0
        latestHeartRate = 0
        latestSteps = 0
        latestDistance = 0
        latestCalories = 0
        
        // Reset Delta tracking
        prevLoggedSteps = 0
        prevLoggedDistance = 0
        prevLoggedCalories = 0
        
        activityManager.resetBaselines()
        
        startWindowTimer()
        
        if (LoggingState.isConnected.value) {
            val path = csvLogger.startNewSession(deviceName)
            saveCurrentSessionFile(path)
            LoggingState.setFilePath(csvLogger.getDisplayPath())
            LoggingState.setLastLogEntry("Starting session...")
            heartRateManager.enableHeartRateNotifications()
            activityManager.enableActivityNotifications()
        } else {
            attemptRecovery()
        }
        lastSuccessfulReadingTime = System.currentTimeMillis()
        scheduleNextWatchdog()
    }

    private fun startWindowTimer() {
        windowTimerJob?.cancel()
        windowTimerJob = serviceScope.launch {
            var timeLeft = 30
            while (LoggingState.isRecording.value) {
                LoggingState.setWindowTime(timeLeft)
                delay(1000)
                timeLeft--
                if (timeLeft < 0) {
                    timeLeft = 30
                    // The actual flush is triggered by ACTION_MEASURE_TICK alarm, 
                    // but we clear the live lab UI here for visual feedback.
                    LoggingState.clearLiveLab()
                }
            }
        }
    }

    private fun stopRecordingAndService() {
        LoggingState.setRecording(false)
        connectionJob?.cancel()
        windowTimerJob?.cancel()
        cancelWatchdog()
        heartRateManager.disableHeartRateNotifications()
        csvLogger.finalizeSession()
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().remove("currentSessionFile").apply()
        manager.close()
        LoggingState.setConnected(false)
        LoggingState.setLastLogEntry("Recording stopped.")
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        isServiceStarted = false
    }

    private fun acquireWakeLock() {
        if (wakeLock == null) {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "NoiseHRLogger::LoggingWakeLock")
        }
        if (wakeLock?.isHeld == false) wakeLock?.acquire(24 * 60 * 60 * 1000L)
    }

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) wakeLock?.release()
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Heart Rate Logging", NotificationManager.IMPORTANCE_LOW))
            val alarmChannel = NotificationChannel(ALARM_CHANNEL_ID, "Heart Rate Alarms", NotificationManager.IMPORTANCE_HIGH).apply {
                setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM), AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build())
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(alarmChannel)
        }
    }

    private fun createNotification(content: String): Notification {
        val pendingIntent = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stopPendingIntent = PendingIntent.getService(this, 0, Intent(this, LoggingService::class.java).apply { action = ACTION_STOP }, PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL_ID).setContentTitle("Noise HR Logger").setContentText(content).setSmallIcon(android.R.drawable.stat_notify_sync).setContentIntent(pendingIntent).addAction(android.R.drawable.ic_menu_close_clear_cancel, "Stop", stopPendingIntent).setOngoing(true).build()
    }

    private fun updateNotification(content: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, createNotification(content))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(bluetoothReceiver)
        connectionJob?.cancel()
    }

    companion object {
        const val ACTION_CONNECT = "ACTION_CONNECT_DEVICE"
        const val ACTION_START = "ACTION_START_LOGGING"
        const val ACTION_MEASURE_TICK = "ACTION_MEASURE_TICK"
        const val ACTION_STOP = "ACTION_STOP_LOGGING"
    }
}
