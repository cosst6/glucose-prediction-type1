package com.example.noisehrlogger.model

import android.bluetooth.BluetoothDevice

data class BleDevice(
    val device: BluetoothDevice,
    val name: String,
    val address: String,
    val rssi: Int,
    val isHeartRateDevice: Boolean = false,
    val isBonded: Boolean = false
)