package com.example.noisehrlogger.ble

import android.bluetooth.BluetoothGattCharacteristic
import java.util.UUID

class HeartRateManager(
    private val manager: BLEManager
) {

    var onHeartRateChanged:
            ((Int) -> Unit)? = null

    init {
        manager.addCharacteristicListener { characteristic ->
            if (characteristic.uuid == HEART_RATE_MEASUREMENT_UUID) {
                val bpm = parseHeartRate(characteristic)
                if (bpm != null) {
                    onHeartRateChanged?.invoke(bpm)
                }
            }
        }
    }

    fun enableHeartRateNotifications() {

        val service =
            manager.getService(
                HEART_RATE_SERVICE_UUID
            ) ?: return

        val characteristic =
            service.getCharacteristic(
                HEART_RATE_MEASUREMENT_UUID
            ) ?: return

        manager.enableNotification(
            characteristic
        )
    }

    fun disableHeartRateNotifications() {

        val service =
            manager.getService(
                HEART_RATE_SERVICE_UUID
            ) ?: return

        val characteristic =
            service.getCharacteristic(
                HEART_RATE_MEASUREMENT_UUID
            ) ?: return

        manager.disableNotification(
            characteristic
        )
    }

    private fun parseHeartRate(
        characteristic: BluetoothGattCharacteristic
    ): Int? {

        val data =
            characteristic.value ?: return null

        if (data.isEmpty()) return null

        val flags = data[0].toInt() and 0xFF

        val is16Bit =
            (flags and 0x01) != 0

        return if (!is16Bit) {

            if (data.size < 2) {
                null
            } else {
                data[1].toInt() and 0xFF
            }

        } else {

            if (data.size < 3) {
                null
            } else {

                (data[1].toInt() and 0xFF) or
                        ((data[2].toInt() and 0xFF) shl 8)
            }
        }
    }

    companion object {

        val HEART_RATE_SERVICE_UUID: UUID =
            UUID.fromString(
                "0000180d-0000-1000-8000-00805f9b34fb"
            )

        val HEART_RATE_MEASUREMENT_UUID: UUID =
            UUID.fromString(
                "00002a37-0000-1000-8000-00805f9b34fb"
            )
    }
}
