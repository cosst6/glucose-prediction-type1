package com.example.noisehrlogger.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.noisehrlogger.ble.BLEScanner
import com.example.noisehrlogger.csv.CSVLogger
import com.example.noisehrlogger.model.BleDevice
import com.example.noisehrlogger.service.LoggingService
import com.example.noisehrlogger.service.LoggingState
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import java.io.File

class BluetoothViewModel(
    private val application: Application
) : AndroidViewModel(application) {

    private val scanner =
        BLEScanner(application)

    private val csvLogger =
        CSVLogger(application)

    val devices =
        mutableStateListOf<BleDevice>()

    val isConnected =
        mutableStateOf(false)

    val isConnectionPending =
        mutableStateOf(false)

    val connectedDeviceName =
        mutableStateOf<String?>(null)

    val currentHeartRate =
        mutableStateOf("--")

    val steps =
        mutableStateOf(0)

    val distance =
        mutableStateOf(0)

    val calories =
        mutableStateOf(0)

    val isRecording =
        mutableStateOf(false)

    val lastLogEntry =
        mutableStateOf("No readings yet")

    val recentLogs =
        mutableStateOf<List<String>>(emptyList())

    val errorStatus =
        mutableStateOf<String?>(null)

    val currentFilePath =
        mutableStateOf("No active file")

    val historyFiles =
        mutableStateOf<List<File>>(emptyList())

    val isBatteryOptimized =
        mutableStateOf(true)

    val isIgnoringSteps =
        mutableStateOf(false)

    val isIgnoringDistance =
        mutableStateOf(false)

    val isIgnoringCalories =
        mutableStateOf(false)

    val auditStatus =
        mutableStateOf("Initializing...")

    val packetCount =
        mutableStateOf(0)

    val liveHrSamples =
        mutableStateOf<List<Int>>(emptyList())

    val liveDeltaSteps =
        mutableStateOf(0)

    val liveDeltaDist =
        mutableStateOf(0)

    val liveDeltaCal =
        mutableStateOf(0)

    val windowTimeLeft =
        mutableStateOf(30)

    val connectionError =
        mutableStateOf<String?>(null)

    val connectionStatusMessage =
        mutableStateOf("Connecting...")

    val showTroubleshooting =
        mutableStateOf(false)

    private var connectedDevice: BleDevice? = null

    init {

        scanner.onDeviceFound = { foundDevices ->
            devices.clear()
            devices.addAll(foundDevices)
        }

        // Observe LoggingState
        LoggingState.isConnected.onEach {
            isConnected.value = it
            if (it) {
                isConnectionPending.value = false
                connectionError.value = null
                showTroubleshooting.value = false
            }
        }.launchIn(viewModelScope)

        LoggingState.isRecording.onEach {
            isRecording.value = it
            refreshHistory()
        }.launchIn(viewModelScope)

        LoggingState.currentHeartRate.onEach {
            currentHeartRate.value = it
        }.launchIn(viewModelScope)

        LoggingState.steps.onEach {
            steps.value = it
        }.launchIn(viewModelScope)

        LoggingState.distance.onEach {
            distance.value = it
        }.launchIn(viewModelScope)

        LoggingState.calories.onEach {
            calories.value = it
        }.launchIn(viewModelScope)

        LoggingState.lastLogEntry.onEach {
            lastLogEntry.value = it
        }.launchIn(viewModelScope)

        LoggingState.recentLogs.onEach {
            recentLogs.value = it
        }.launchIn(viewModelScope)

        LoggingState.errorStatus.onEach {
            errorStatus.value = it
        }.launchIn(viewModelScope)

        LoggingState.connectedDeviceName.onEach {
            connectedDeviceName.value = it
        }.launchIn(viewModelScope)

        LoggingState.currentFilePath.onEach {
            currentFilePath.value = it
        }.launchIn(viewModelScope)

        LoggingState.historyFiles.onEach {
            historyFiles.value = it
        }.launchIn(viewModelScope)

        LoggingState.isBatteryOptimized.onEach {
            isBatteryOptimized.value = it
        }.launchIn(viewModelScope)

        LoggingState.isIgnoringSteps.onEach {
            isIgnoringSteps.value = it
        }.launchIn(viewModelScope)

        LoggingState.isIgnoringDistance.onEach {
            isIgnoringDistance.value = it
        }.launchIn(viewModelScope)

        LoggingState.isIgnoringCalories.onEach {
            isIgnoringCalories.value = it
        }.launchIn(viewModelScope)

        LoggingState.auditStatus.onEach {
            auditStatus.value = it
        }.launchIn(viewModelScope)

        LoggingState.packetCount.onEach {
            packetCount.value = it
        }.launchIn(viewModelScope)

        LoggingState.liveHrSamples.onEach {
            liveHrSamples.value = it
        }.launchIn(viewModelScope)

        LoggingState.liveDeltaSteps.onEach {
            liveDeltaSteps.value = it
        }.launchIn(viewModelScope)

        LoggingState.liveDeltaDist.onEach {
            liveDeltaDist.value = it
        }.launchIn(viewModelScope)

        LoggingState.liveDeltaCal.onEach {
            liveDeltaCal.value = it
        }.launchIn(viewModelScope)

        LoggingState.windowTimeLeft.onEach {
            windowTimeLeft.value = it
        }.launchIn(viewModelScope)

        checkBatteryOptimization()
        refreshHistory()
    }

    fun refreshHistory() {
        val files = csvLogger.getAllLogs()
        LoggingState.setHistoryFiles(files)
    }

    fun checkBatteryOptimization() {
        val powerManager = application.getSystemService(Context.POWER_SERVICE) as PowerManager
        val isIgnored = powerManager.isIgnoringBatteryOptimizations(application.packageName)
        LoggingState.setBatteryOptimized(!isIgnored)
    }

    fun requestIgnoreBatteryOptimizations() {
        val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        application.startActivity(intent)
    }

    fun toggleIgnoreSteps() {
        LoggingState.setIgnoreSteps(!isIgnoringSteps.value)
    }

    fun toggleIgnoreDistance() {
        LoggingState.setIgnoreDistance(!isIgnoringDistance.value)
    }

    fun toggleIgnoreCalories() {
        LoggingState.setIgnoreCalories(!isIgnoringCalories.value)
    }

    fun startScan() {
        if (!isConnectionPending.value) {
            scanner.startScan()
        }
    }

    fun stopScan() {
        scanner.stopScan()
    }

    fun connect(device: BleDevice) {
        stopScan()
        connectedDevice = device
        isConnectionPending.value = true
        connectionError.value = null
        showTroubleshooting.value = false
        connectionStatusMessage.value = "Connecting to ${device.name}..."
        
        val intent = Intent(application, LoggingService::class.java).apply {
            action = LoggingService.ACTION_CONNECT
            putExtra("EXTRA_DEVICE_ADDRESS", device.address)
            putExtra("EXTRA_DEVICE_NAME", device.name)
        }
        application.startService(intent)
    }

    fun disconnect() {
        stopRecording()
        val intent = Intent(application, LoggingService::class.java).apply {
            action = LoggingService.ACTION_STOP
        }
        application.startService(intent)
    }

    fun startRecording() {
        val deviceName = LoggingState.connectedDeviceName.value ?: connectedDevice?.name ?: "Unknown Device"
        
        val intent = Intent(application, LoggingService::class.java).apply {
            action = LoggingService.ACTION_START
            putExtra("EXTRA_DEVICE_NAME", deviceName)
        }
        
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            application.startForegroundService(intent)
        } else {
            application.startService(intent)
        }
    }

    fun stopRecording() {
        val intent = Intent(application, LoggingService::class.java).apply {
            action = LoggingService.ACTION_STOP
        }
        application.startService(intent)
    }

    fun openFile(file: File) {
        try {
            val uri: Uri = FileProvider.getUriForFile(
                application,
                "com.example.noisehrlogger.provider",
                file
            )
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "text/comma-separated-values")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            application.startActivity(intent)
        } catch (e: Exception) {
            Log.e("BluetoothViewModel", "Error opening file", e)
        }
    }

    override fun onCleared() {
        scanner.stopScan()
        super.onCleared()
    }
}
