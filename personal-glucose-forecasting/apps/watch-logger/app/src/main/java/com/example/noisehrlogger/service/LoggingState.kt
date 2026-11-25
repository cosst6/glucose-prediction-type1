package com.example.noisehrlogger.service

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

object LoggingState {

    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _currentHeartRate = MutableStateFlow("--")
    val currentHeartRate: StateFlow<String> = _currentHeartRate.asStateFlow()

    // Activity stats
    private val _steps = MutableStateFlow(0)
    val steps: StateFlow<Int> = _steps.asStateFlow()

    private val _distance = MutableStateFlow(0)
    val distance: StateFlow<Int> = _distance.asStateFlow()

    private val _calories = MutableStateFlow(0)
    val calories: StateFlow<Int> = _calories.asStateFlow()

    private val _lastLogEntry = MutableStateFlow("No readings yet")
    val lastLogEntry: StateFlow<String> = _lastLogEntry.asStateFlow()

    private val _recentLogs = MutableStateFlow<List<String>>(emptyList())
    val recentLogs: StateFlow<List<String>> = _recentLogs.asStateFlow()

    private val _errorStatus = MutableStateFlow<String?>(null)
    val errorStatus: StateFlow<String?> = _errorStatus.asStateFlow()

    private val _connectedDeviceName = MutableStateFlow<String?>(null)
    val connectedDeviceName: StateFlow<String?> = _connectedDeviceName.asStateFlow()

    private val _currentFilePath = MutableStateFlow("No active file")
    val currentFilePath: StateFlow<String> = _currentFilePath.asStateFlow()

    private val _historyFiles = MutableStateFlow<List<File>>(emptyList())
    val historyFiles: StateFlow<List<File>> = _historyFiles.asStateFlow()

    private val _isBatteryOptimized = MutableStateFlow(true)
    val isBatteryOptimized: StateFlow<Boolean> = _isBatteryOptimized.asStateFlow()

    // Activity Control Flags
    private val _isIgnoringSteps = MutableStateFlow(false)
    val isIgnoringSteps: StateFlow<Boolean> = _isIgnoringSteps.asStateFlow()

    private val _isIgnoringDistance = MutableStateFlow(false)
    val isIgnoringDistance: StateFlow<Boolean> = _isIgnoringDistance.asStateFlow()

    private val _isIgnoringCalories = MutableStateFlow(false)
    val isIgnoringCalories: StateFlow<Boolean> = _isIgnoringCalories.asStateFlow()

    // Audit / Health Monitor
    private val _auditStatus = MutableStateFlow("System Initializing...")
    val auditStatus: StateFlow<String> = _auditStatus.asStateFlow()

    private val _packetCount = MutableStateFlow(0)
    val packetCount: StateFlow<Int> = _packetCount.asStateFlow()

    // Live Processing Lab
    private val _liveHrSamples = MutableStateFlow<List<Int>>(emptyList())
    val liveHrSamples: StateFlow<List<Int>> = _liveHrSamples.asStateFlow()

    private val _liveDeltaSteps = MutableStateFlow(0)
    val liveDeltaSteps: StateFlow<Int> = _liveDeltaSteps.asStateFlow()

    private val _liveDeltaDist = MutableStateFlow(0)
    val liveDeltaDist: StateFlow<Int> = _liveDeltaDist.asStateFlow()

    private val _liveDeltaCal = MutableStateFlow(0)
    val liveDeltaCal: StateFlow<Int> = _liveDeltaCal.asStateFlow()

    private val _windowTimeLeft = MutableStateFlow(30)
    val windowTimeLeft: StateFlow<Int> = _windowTimeLeft.asStateFlow()

    fun setConnected(connected: Boolean, deviceName: String? = null) {
        _isConnected.value = connected
        if (!connected) {
            _connectedDeviceName.value = null
            _currentHeartRate.value = "--"
            _steps.value = 0
            _distance.value = 0
            _calories.value = 0
            _auditStatus.value = "Disconnected"
        } else if (deviceName != null) {
            _connectedDeviceName.value = deviceName
            _errorStatus.value = null
            _auditStatus.value = "Connected & Active"
        }
    }

    fun setRecording(recording: Boolean) {
        _isRecording.value = recording
        if (!recording) {
            _recentLogs.value = emptyList()
            _errorStatus.value = null
            _currentFilePath.value = "No active file"
            _packetCount.value = 0
            _liveHrSamples.value = emptyList()
            _liveDeltaSteps.value = 0
            _liveDeltaDist.value = 0
            _liveDeltaCal.value = 0
            _windowTimeLeft.value = 30
        }
    }

    fun setHeartRate(bpm: Int) {
        _currentHeartRate.value = bpm.toString()
        _auditStatus.value = "Pulse: Receiving ($bpm)"

        val currentSamples = _liveHrSamples.value.toMutableList()
        currentSamples.add(bpm)
        if (currentSamples.size > 8) currentSamples.removeAt(0)
        _liveHrSamples.value = currentSamples
    }

    fun setActivityData(steps: Int, distance: Int, calories: Int) {
        _steps.value = steps
        _distance.value = distance
        _calories.value = calories
        _packetCount.value++
        _auditStatus.value = "Activity: Receiving (Pkt: ${_packetCount.value})"
    }

    fun setLiveDeltas(steps: Int, distance: Int, calories: Int) {
        _liveDeltaSteps.value = steps
        _liveDeltaDist.value = distance
        _liveDeltaCal.value = calories
    }

    fun setWindowTime(seconds: Int) {
        _windowTimeLeft.value = seconds
    }

    fun clearLiveLab() {
        _liveHrSamples.value = emptyList()
        _liveDeltaSteps.value = 0
        _liveDeltaDist.value = 0
        _liveDeltaCal.value = 0
    }

    fun setAuditMessage(msg: String) {
        _auditStatus.value = msg
    }

    fun setLastLogEntry(entry: String) {
        _lastLogEntry.value = entry
        val currentList = _recentLogs.value.toMutableList()
        currentList.add(0, entry)
        if (currentList.size > 20) {
            currentList.removeAt(currentList.size - 1)
        }
        _recentLogs.value = currentList
    }

    fun setError(error: String?) {
        _errorStatus.value = error
    }

    fun setFilePath(path: String) {
        _currentFilePath.value = path
    }

    fun setHistoryFiles(files: List<File>) {
        _historyFiles.value = files.sortedByDescending { it.lastModified() }
    }

    fun setBatteryOptimized(optimized: Boolean) {
        _isBatteryOptimized.value = optimized
    }

    fun setIgnoreSteps(ignore: Boolean) {
        _isIgnoringSteps.value = ignore
    }

    fun setIgnoreDistance(ignore: Boolean) {
        _isIgnoringDistance.value = ignore
    }

    fun setIgnoreCalories(ignore: Boolean) {
        _isIgnoringCalories.value = ignore
    }
}
