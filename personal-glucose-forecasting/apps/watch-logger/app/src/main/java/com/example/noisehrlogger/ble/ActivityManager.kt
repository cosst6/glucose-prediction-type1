package com.example.noisehrlogger.ble

import android.bluetooth.BluetoothGattCharacteristic
import android.util.Log
import java.util.UUID

class ActivityManager(
    private val manager: BLEManager
) {
    private val TAG = "ActivityManager"

    data class ActivityData(
        val steps: Int,
        val distanceMeters: Int,
        val calories: Int
    )

    var onActivityDataChanged: ((ActivityData) -> Unit)? = null

    private var stepBaseline: Int = -1
    private var distanceBaseline: Int = -1
    private var calorieBaseline: Int = -1

    fun resetBaselines() {
        Log.i(TAG, "Activity baselines reset")
        stepBaseline = -1
        distanceBaseline = -1
        calorieBaseline = -1
    }

    init {
        manager.addCharacteristicListener { characteristic ->
            if (characteristic.uuid == ACTIVITY_CHARACTERISTIC_UUID) {
                val data = decodeActivityPacket(characteristic.value)
                if (data != null) {
                    onActivityDataChanged?.invoke(data)
                }
            }
        }
    }

    fun enableActivityNotifications() {
        val service = manager.getService(ACTIVITY_SERVICE_UUID) ?: return
        val characteristic = service.getCharacteristic(ACTIVITY_CHARACTERISTIC_UUID) ?: return
        manager.enableNotification(characteristic)
    }

    private fun decodeActivityPacket(data: ByteArray?): ActivityData? {
        if (data == null || data.size < 6) {
            Log.w(TAG, "Invalid activity packet received: ${data?.size ?: 0} bytes")
            return null
        }

        // Log hex for debugging
        val hex = data.joinToString("") { String.format("%02X ", it) }
        Log.d(TAG, "Raw Activity Packet: $hex")

        // Most Noise/ColorFit watches use 3-byte little-endian for steps/dist/cal
        // Bytes might start with a header or directly with data.
        // We'll try to detect the format based on size.
        
        val rawSteps: Int
        val rawDistance: Int
        val rawCalories: Int

        if (data.size >= 9) {
            // Assume 3 bytes per value (common in modern Noise watches)
            rawSteps = (data[0].toInt() and 0xFF) or (data[1].toInt() and 0xFF shl 8) or (data[2].toInt() and 0xFF shl 16)
            rawDistance = (data[3].toInt() and 0xFF) or (data[4].toInt() and 0xFF shl 8) or (data[5].toInt() and 0xFF shl 16)
            rawCalories = (data[6].toInt() and 0xFF) or (data[7].toInt() and 0xFF shl 8) or (data[8].toInt() and 0xFF shl 16)
        } else {
            // Fallback to 2 bytes per value
            rawSteps = (data[0].toInt() and 0xFF) or (data[1].toInt() and 0xFF shl 8)
            rawDistance = (data[2].toInt() and 0xFF) or (data[3].toInt() and 0xFF shl 8)
            rawCalories = (data[4].toInt() and 0xFF) or (data[5].toInt() and 0xFF shl 8)
        }

        // Initialize baselines on first valid packet after reset
        if (stepBaseline == -1) {
            stepBaseline = rawSteps
            distanceBaseline = rawDistance
            calorieBaseline = rawCalories
            Log.i(TAG, "Baselines initialized: Steps=$stepBaseline, Dist=$distanceBaseline, Cal=$calorieBaseline")
            return ActivityData(0, 0, 0)
        }

        val steps = rawSteps - stepBaseline
        val distance = rawDistance - distanceBaseline
        val calories = rawCalories - calorieBaseline

        Log.d(TAG, "Decoded: Steps=$steps, Dist=$distance m, Cal=$calories (Raw: $rawSteps, $rawDistance, $rawCalories)")

        return ActivityData(
            steps = Math.max(0, steps),
            distanceMeters = Math.max(0, distance),
            calories = Math.max(0, calories)
        )
    }

    companion object {
        val ACTIVITY_SERVICE_UUID: UUID = UUID.fromString("0000feea-0000-1000-8000-00805f9b34fb")
        val ACTIVITY_CHARACTERISTIC_UUID: UUID = UUID.fromString("0000fee1-0000-1000-8000-00805f9b34fb")
    }
}
