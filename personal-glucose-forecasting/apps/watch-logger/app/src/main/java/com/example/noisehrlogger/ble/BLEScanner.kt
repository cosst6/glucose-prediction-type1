package com.example.noisehrlogger.ble

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.ParcelUuid
import android.util.Log
import androidx.core.content.ContextCompat
import com.example.noisehrlogger.model.BleDevice

class BLEScanner(
    private val context: Context
) {

    private val TAG = "BLEScanner"

    private val HEART_RATE_SERVICE_UUID =
        ParcelUuid.fromString("0000180d-0000-1000-8000-00805f9b34fb")

    private val bluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager

    private val bluetoothAdapter: BluetoothAdapter?
        get() = bluetoothManager.adapter

    private val scanner: BluetoothLeScanner?
        get() = bluetoothAdapter?.bluetoothLeScanner

    private val devices = mutableMapOf<String, BleDevice>()

    var onDeviceFound: ((List<BleDevice>) -> Unit)? = null

    @SuppressLint("MissingPermission")
    fun getBondedDevices(): List<BleDevice> {
        if (!hasConnectPermission()) return emptyList()

        return bluetoothAdapter?.bondedDevices?.map { device ->
            val name = try {
                device.name ?: "Unknown Paired Device"
            } catch (_: SecurityException) {
                "Unknown Paired Device"
            }

            BleDevice(
                device = device,
                name = name,
                address = device.address,
                rssi = -100, // Default for bonded devices not in scan range
                isHeartRateDevice = name.contains("QUAD CALL", ignoreCase = true),
                isBonded = true
            )
        } ?: emptyList()
    }

    private val scanCallback = object : ScanCallback() {

        @SuppressLint("MissingPermission")
        override fun onScanResult(
            callbackType: Int,
            result: ScanResult
        ) {

            if (!hasScanPermission()) return
            if (!hasConnectPermission()) return

            val device = result.device
            val scanRecord = result.scanRecord

            val name = try {
                scanRecord?.deviceName ?: device.name ?: "Unknown Device"
            } catch (_: SecurityException) {
                "Unknown Device"
            }

            val serviceUuids = scanRecord?.serviceUuids
            val isHeartRate = serviceUuids?.contains(HEART_RATE_SERVICE_UUID) == true

            val bleDevice = BleDevice(
                device = device,
                name = name,
                address = device.address,
                rssi = result.rssi,
                isHeartRateDevice = isHeartRate,
                isBonded = device.bondState == android.bluetooth.BluetoothDevice.BOND_BONDED
            )

            devices[device.address] = bleDevice

            onDeviceFound?.invoke(getSortedDevices())
        }

        @SuppressLint("MissingPermission")
        override fun onBatchScanResults(
            results: MutableList<ScanResult>
        ) {

            for (result in results) {

                if (!hasScanPermission()) return
                if (!hasConnectPermission()) return

                val device = result.device
                val scanRecord = result.scanRecord

                val name = try {
                    scanRecord?.deviceName ?: device.name ?: "Unknown Device"
                } catch (_: SecurityException) {
                    "Unknown Device"
                }

                val serviceUuids = scanRecord?.serviceUuids
                val isHeartRate = serviceUuids?.contains(HEART_RATE_SERVICE_UUID) == true

                devices[device.address] = BleDevice(
                    device = device,
                    name = name,
                    address = device.address,
                    rssi = result.rssi,
                    isHeartRateDevice = isHeartRate,
                    isBonded = device.bondState == android.bluetooth.BluetoothDevice.BOND_BONDED
                )
            }

            onDeviceFound?.invoke(getSortedDevices())
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "BLE Scan Failed: $errorCode")
        }
    }

    @SuppressLint("MissingPermission")
    fun startScan() {

        Log.d(TAG, "Attempting to start scan...")

        if (!hasScanPermission()) {
            Log.w(TAG, "Missing Scan Permission")
            return
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (!hasConnectPermission()) {
                Log.w(TAG, "Missing Connect Permission")
                return
            }
        }

        val bleScanner = scanner
        if (bleScanner == null) {
            Log.e(TAG, "BluetoothLeScanner is null. Is Bluetooth ON?")
            return
        }

        devices.clear()

        // Include bonded devices immediately
        getBondedDevices().forEach { bonded ->
            devices[bonded.address] = bonded
        }

        onDeviceFound?.invoke(getSortedDevices())

        Log.i(TAG, "Starting BLE Scan")
        bleScanner.startScan(scanCallback)
    }

    private fun getSortedDevices(): List<BleDevice> {
        return devices.values.toList()
            .sortedWith(compareByDescending<BleDevice> { it.isHeartRateDevice }
                .thenByDescending { it.rssi })
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {

        Log.d(TAG, "Stopping scan")
        if (!hasScanPermission()) return

        scanner?.stopScan(scanCallback)
    }

    fun clearDevices() {
        devices.clear()
    }

    private fun hasScanPermission(): Boolean {

        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Manifest.permission.BLUETOOTH_SCAN
        } else {
            Manifest.permission.ACCESS_FINE_LOCATION
        }

        val granted = ContextCompat.checkSelfPermission(
            context,
            permission
        ) == PackageManager.PERMISSION_GRANTED

        Log.v(TAG, "Has $permission: $granted")
        return granted
    }

    private fun hasConnectPermission(): Boolean {

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return true

        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BLUETOOTH_CONNECT
        ) == PackageManager.PERMISSION_GRANTED
    }
}