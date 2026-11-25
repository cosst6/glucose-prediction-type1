package com.example.noisehrlogger

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.example.noisehrlogger.ui.theme.NoiseHRLoggerTheme
import com.example.noisehrlogger.userinterface.MainScreen
import com.example.noisehrlogger.viewmodel.BluetoothViewModel

class MainActivity : ComponentActivity() {

    private val TAG = "MainActivity"

    private val bluetoothViewModel:
            BluetoothViewModel by viewModels()

    private val permissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->

            val allowed =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {

                    val bluetoothScan = permissions[Manifest.permission.BLUETOOTH_SCAN] == true
                    val bluetoothConnect = permissions[Manifest.permission.BLUETOOTH_CONNECT] == true
                    val postNotifications = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        permissions[Manifest.permission.POST_NOTIFICATIONS] == true
                    } else true
                    
                    bluetoothScan && bluetoothConnect && postNotifications

                } else {

                    permissions[
                        Manifest.permission.ACCESS_FINE_LOCATION
                    ] == true
                }

            if (allowed) {

                bluetoothViewModel.startScan()
            }
        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContent {

            NoiseHRLoggerTheme {

                MainScreen(

                    devices =
                        bluetoothViewModel.devices,

                    isConnected =
                        bluetoothViewModel
                            .isConnected
                            .value,

                    isConnectionPending =
                        bluetoothViewModel
                            .isConnectionPending
                            .value,

                    connectedDeviceName =
                        bluetoothViewModel
                            .connectedDeviceName
                            .value,

                    heartRate =
                        bluetoothViewModel
                            .currentHeartRate
                            .value,

                    steps = 
                        bluetoothViewModel
                            .steps
                            .value,

                    distance = 
                        bluetoothViewModel
                            .distance
                            .value,

                    calories = 
                        bluetoothViewModel
                            .calories
                            .value,

                    isRecording =
                        bluetoothViewModel
                            .isRecording
                            .value,

                    recentLogs = 
                        bluetoothViewModel
                            .recentLogs
                            .value,

                    errorStatus = 
                        bluetoothViewModel
                            .errorStatus
                            .value,

                    currentFilePath = 
                        bluetoothViewModel
                            .currentFilePath
                            .value,

                    historyFiles = 
                        bluetoothViewModel
                            .historyFiles
                            .value,

                    isBatteryOptimized = 
                        bluetoothViewModel
                            .isBatteryOptimized
                            .value,

                    auditStatus = 
                        bluetoothViewModel
                            .auditStatus
                            .value,

                    packetCount = 
                        bluetoothViewModel
                            .packetCount
                            .value,

                    liveHrSamples = 
                        bluetoothViewModel
                            .liveHrSamples
                            .value,

                    liveDeltaSteps = 
                        bluetoothViewModel
                            .liveDeltaSteps
                            .value,

                    liveDeltaDist = 
                        bluetoothViewModel
                            .liveDeltaDist
                            .value,

                    liveDeltaCal = 
                        bluetoothViewModel
                            .liveDeltaCal
                            .value,

                    windowTimeLeft = 
                        bluetoothViewModel
                            .windowTimeLeft
                            .value,

                    isIgnoringSteps = 
                        bluetoothViewModel
                            .isIgnoringSteps
                            .value,

                    isIgnoringDistance = 
                        bluetoothViewModel
                            .isIgnoringDistance
                            .value,

                    isIgnoringCalories = 
                        bluetoothViewModel
                            .isIgnoringCalories
                            .value,

                    onScanClick = {
                        startBluetoothScan()
                    },

                    onDeviceClick = { device ->
                        bluetoothViewModel
                            .connect(device)
                    },

                    onStartRecording = {
                        bluetoothViewModel
                            .startRecording()
                    },

                    onStopRecording = {
                        bluetoothViewModel
                            .stopRecording()
                    },

                    onFileClick = { file ->
                        bluetoothViewModel
                            .openFile(file)
                    },

                    onIgnoreBatteryOptimizations = {
                        bluetoothViewModel.requestIgnoreBatteryOptimizations()
                    },

                    onToggleIgnoreSteps = {
                        bluetoothViewModel.toggleIgnoreSteps()
                    },

                    onToggleIgnoreDistance = {
                        bluetoothViewModel.toggleIgnoreDistance()
                    },

                    onToggleIgnoreCalories = {
                        bluetoothViewModel.toggleIgnoreCalories()
                    }
                )
            }
        }
    }

    private fun startBluetoothScan() {

        Log.d(TAG, "startBluetoothScan() called")

        if (!isBluetoothEnabled()) {

            Log.w(TAG, "Bluetooth is disabled")
            openBluetoothSettings()
            return
        }

        val requiredPermissions =
            getRequiredPermissions()

        val missingPermissions =
            requiredPermissions.filter { permission ->

                val granted = ContextCompat.checkSelfPermission(
                    this,
                    permission
                ) == PackageManager.PERMISSION_GRANTED
                Log.v(TAG, "Permission $permission granted: $granted")
                !granted
            }

        if (missingPermissions.isNotEmpty()) {

            Log.i(TAG, "Requesting permissions: $missingPermissions")
            permissionLauncher.launch(
                missingPermissions.toTypedArray()
            )

        } else {

            Log.i(TAG, "All permissions granted, starting scan via ViewModel")
            bluetoothViewModel.startScan()
        }
    }

    private fun getRequiredPermissions():
            Array<String> {

        val permissions = mutableListOf<String>()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        
        return permissions.toTypedArray()
    }

    private fun isBluetoothEnabled(): Boolean {

        val bluetoothManager =
            getSystemService(
                BLUETOOTH_SERVICE
            ) as BluetoothManager

        val adapter:
                BluetoothAdapter? =
            bluetoothManager.adapter

        return adapter?.isEnabled == true
    }

    private fun openBluetoothSettings() {

        startActivity(
            Intent(
                Settings.ACTION_BLUETOOTH_SETTINGS
            )
        )
    }
}
