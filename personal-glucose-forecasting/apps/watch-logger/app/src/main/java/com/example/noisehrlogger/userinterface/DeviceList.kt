package com.example.noisehrlogger.userinterface

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.example.noisehrlogger.model.BleDevice

@Composable
fun DeviceList(
    devices: List<BleDevice>,
    onDeviceClick: (BleDevice) -> Unit
) {

    LazyColumn(
        modifier = Modifier.fillMaxSize()
    ) {

        items(devices) { device ->

            DeviceRow(
                device = device,
                onClick = onDeviceClick
            )

        }

    }
}