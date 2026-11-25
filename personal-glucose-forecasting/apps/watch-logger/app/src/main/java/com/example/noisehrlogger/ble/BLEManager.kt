package com.example.noisehrlogger.ble

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.UUID

class BLEManager(
    private val context: Context
) {

    private val TAG = "BLEManager"
    private var bluetoothGatt: BluetoothGatt? = null
    private val serviceDiscoveryScope = CoroutineScope(Dispatchers.IO + Job())

    var onConnected: (() -> Unit)? = null
    var onDisconnected: ((Int) -> Unit)? = null
    var onServicesReady: (() -> Unit)? = null
    var onDescriptorWritten: ((Boolean) -> Unit)? = null

    private val characteristicListeners = mutableListOf<(BluetoothGattCharacteristic) -> Unit>()

    fun addCharacteristicListener(listener: (BluetoothGattCharacteristic) -> Unit) {
        characteristicListeners.add(listener)
    }

    fun removeCharacteristicListener(listener: (BluetoothGattCharacteristic) -> Unit) {
        characteristicListeners.remove(listener)
    }

    private val gattCallback = object : BluetoothGattCallback() {

        override fun onConnectionStateChange(
            gatt: BluetoothGatt,
            status: Int,
            newState: Int
        ) {
            Log.d(TAG, "onConnectionStateChange: status=$status, newState=$newState")

            if (status != BluetoothGatt.GATT_SUCCESS) {
                Log.e(TAG, "GATT Error status: $status. Force closing GATT handle.")
                onDisconnected?.invoke(status)
                close() // Ensure resource cleanup to prevent Stage 1/Status 1 errors
                return
            }

            if (newState == BluetoothProfile.STATE_CONNECTED) {
                Log.i(TAG, "Connected to GATT server. Starting service discovery...")
                bluetoothGatt = gatt
                
                serviceDiscoveryScope.launch {
                    var discovered = false
                    var attempts = 0
                    while (!discovered && attempts < 3) {
                        attempts++
                        Log.d(TAG, "Service discovery attempt $attempts...")
                        if (gatt.discoverServices()) {
                            discovered = true
                        } else {
                            Log.w(TAG, "discoverServices() returned false, retrying in 1.5s...")
                            delay(1500)
                        }
                    }
                    if (!discovered) {
                        Log.e(TAG, "All service discovery attempts failed. Force closing.")
                        onDisconnected?.invoke(-2)
                        close()
                    }
                }

            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                Log.w(TAG, "Disconnected from GATT server. Status: $status")
                onDisconnected?.invoke(status)
                close() // Full close on legitimate disconnect too

                if (bluetoothGatt == gatt) {
                    bluetoothGatt = null
                }
            }
        }

        override fun onServicesDiscovered(
            gatt: BluetoothGatt,
            status: Int
        ) {
            Log.d(TAG, "onServicesDiscovered: status=$status")

            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.i(TAG, "Services discovered successfully")
                onServicesReady?.invoke()
            } else {
                Log.e(TAG, "Service discovery failed with status: $status")
                onDisconnected?.invoke(status)
                close()
            }
        }

        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: BluetoothGattDescriptor,
            status: Int
        ) {
            Log.d(TAG, "onDescriptorWrite: ${descriptor.uuid}, status=$status")
            onDescriptorWritten?.invoke(status == BluetoothGatt.GATT_SUCCESS)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            characteristicListeners.forEach { it.invoke(characteristic) }
        }

        @Deprecated("Deprecated on newer Android versions")
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            characteristicListeners.forEach { it.invoke(characteristic) }
        }
    }

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice, useLeTransport: Boolean = true, autoConnect: Boolean = false) {
        val transportStr = if (useLeTransport) "LE" else "AUTO"
        Log.d(TAG, "Connecting to ${device.address} (Transport: $transportStr, autoConnect: $autoConnect)")
        
        // Clean up previous connection if it exists
        close()

        val transport = if (useLeTransport && Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            BluetoothDevice.TRANSPORT_LE
        } else {
            BluetoothDevice.TRANSPORT_AUTO
        }

        bluetoothGatt = device.connectGatt(
            context,
            autoConnect,
            gattCallback,
            transport
        )
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        Log.d(TAG, "Disconnecting GATT")
        bluetoothGatt?.disconnect()
    }

    @SuppressLint("MissingPermission")
    fun close() {
        Log.d(TAG, "Closing GATT resource")
        bluetoothGatt?.disconnect()
        bluetoothGatt?.close()
        bluetoothGatt = null
    }

    /**
     * Clears the internal GATT cache. Useful for fixing GATT 133 errors.
     */
    fun refreshDeviceCache(): Boolean {
        val gatt = bluetoothGatt ?: return false
        Log.i(TAG, "Refreshing device cache (GATT 133 fix attempt)")
        return try {
            val refreshMethod = gatt.javaClass.getMethod("refresh")
            val success = refreshMethod.invoke(gatt) as Boolean
            Log.d(TAG, "Cache refresh called, result: $success")
            success
        } catch (e: Exception) {
            Log.e(TAG, "CRITICAL: Failed to refresh device cache via reflection", e)
            false
        }
    }

    fun getService(
        serviceUuid: UUID
    ): android.bluetooth.BluetoothGattService? {

        return bluetoothGatt?.getService(serviceUuid)
    }

    @SuppressLint("MissingPermission")
    fun enableNotification(
        characteristic: BluetoothGattCharacteristic
    ): Boolean {

        val gatt = bluetoothGatt ?: return false

        val notificationSet =
            gatt.setCharacteristicNotification(
                characteristic,
                true
            )

        val descriptor =
            characteristic.getDescriptor(
                CLIENT_CHARACTERISTIC_CONFIG
            ) ?: return notificationSet

        descriptor.value =
            BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE

        gatt.writeDescriptor(descriptor)

        return notificationSet
    }

    @SuppressLint("MissingPermission")
    fun disableNotification(
        characteristic: BluetoothGattCharacteristic
    ): Boolean {

        val gatt = bluetoothGatt ?: return false

        val notificationSet =
            gatt.setCharacteristicNotification(
                characteristic,
                false
            )

        val descriptor =
            characteristic.getDescriptor(
                CLIENT_CHARACTERISTIC_CONFIG
            ) ?: return notificationSet

        descriptor.value =
            BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE

        gatt.writeDescriptor(descriptor)

        return notificationSet
    }

    companion object {

        val CLIENT_CHARACTERISTIC_CONFIG: UUID =
            UUID.fromString(
                "00002902-0000-1000-8000-00805f9b34fb"
            )
    }
}