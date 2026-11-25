package com.example.noisehrlogger.userinterface

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.noisehrlogger.model.BleDevice
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun MainScreen(
    devices: List<BleDevice>,
    isConnected: Boolean,
    isConnectionPending: Boolean,
    connectedDeviceName: String?,
    heartRate: String,
    steps: Int,
    distance: Int,
    calories: Int,
    isRecording: Boolean,
    recentLogs: List<String>,
    errorStatus: String?,
    currentFilePath: String,
    historyFiles: List<File>,
    isBatteryOptimized: Boolean,
    auditStatus: String,
    packetCount: Int,
    liveHrSamples: List<Int> = emptyList(),
    liveDeltaSteps: Int = 0,
    liveDeltaDist: Int = 0,
    liveDeltaCal: Int = 0,
    windowTimeLeft: Int = 30,
    isIgnoringSteps: Boolean = false,
    isIgnoringDistance: Boolean = false,
    isIgnoringCalories: Boolean = false,
    connectionError: String? = null,
    connectionStatusMessage: String = "",
    showTroubleshooting: Boolean = false,
    onScanClick: () -> Unit,
    onDeviceClick: (BleDevice) -> Unit,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    onFileClick: (File) -> Unit,
    onIgnoreBatteryOptimizations: () -> Unit,
    onToggleIgnoreSteps: () -> Unit = {},
    onToggleIgnoreDistance: () -> Unit = {},
    onToggleIgnoreCalories: () -> Unit = {}
) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {

        Text(
            text = "Noise HR Logger",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Reliability Checklist / Setup Dashboard
        Card(
            modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(text = "🛡️ Reliability Checklist", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(8.dp))
                
                SetupItem(
                    label = "Battery Unrestricted",
                    isDone = !isBatteryOptimized,
                    onClick = onIgnoreBatteryOptimizations
                )
                
                SetupItem(
                    label = "App Locked in Recents (Manual)",
                    isDone = false,
                    onClick = null,
                    showManualTip = true
                )
            }
        }

        // Active Error Status
        if (errorStatus != null) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF9C4)),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = errorStatus,
                        color = Color.Black,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }

        // Internal Audit / Health Monitor
        Card(
            modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.3f))
        ) {
            Row(
                modifier = Modifier.padding(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = "🛡️ Audit:", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = auditStatus,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "Pkts: $packetCount",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }

        // Live Processing Lab
        if (isRecording) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                colors = CardDefaults.cardColors(containerColor = Color.Black),
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "🧪 Live Processing Lab",
                            color = Color.Green,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.weight(1f))
                        Text(
                            text = "Next Log: ${windowTimeLeft}s",
                            color = Color.Yellow,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = "Incoming HR Samples:",
                        color = Color.LightGray,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                        if (liveHrSamples.isEmpty()) {
                            Text(text = "Waiting for pulse...", color = Color.DarkGray, fontSize = 10.sp)
                        } else {
                            liveHrSamples.forEach { sample ->
                                Box(
                                    modifier = Modifier
                                        .padding(end = 4.dp)
                                        .clip(RoundedCornerShape(4.dp))
                                        .background(Color.DarkGray)
                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                ) {
                                    Text(text = sample.toString(), color = Color.White, fontSize = 10.sp)
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = "Calculating Session Delta:",
                        color = Color.LightGray,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = "Steps: +$liveDeltaSteps", color = Color.Cyan, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(text = "Dist: +$liveDeltaDist m", color = Color.Cyan, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(text = "Cal: +$liveDeltaCal", color = Color.Cyan, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                    
                    Text(
                        text = "(Value moves to CSV at 0s)",
                        color = Color.DarkGray,
                        fontSize = 9.sp,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        }

        // Activity Data Controls
        if (isRecording) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f))
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = "⚙️ Activity Controls (Workout Mode)",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(modifier = Modifier.fillMaxWidth()) {
                        IgnoreButton(
                            label = "Steps",
                            isIgnored = isIgnoringSteps,
                            onClick = onToggleIgnoreSteps,
                            modifier = Modifier.weight(1f)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        IgnoreButton(
                            label = "Dist",
                            isIgnored = isIgnoringDistance,
                            onClick = onToggleIgnoreDistance,
                            modifier = Modifier.weight(1f)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        IgnoreButton(
                            label = "Cals",
                            isIgnored = isIgnoringCalories,
                            onClick = onToggleIgnoreCalories,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Text(
                        text = "Enable these to ignore false movement signals (e.g. while cycling).",
                        style = MaterialTheme.typography.labelSmall,
                        fontSize = 9.sp,
                        color = Color.DarkGray,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            }
        }

        LazyColumn(
            modifier = Modifier.weight(1f)
        ) {
            // Status Section
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isConnected) Color(0xFFE8F5E9) else MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(modifier = Modifier.size(12.dp).clip(CircleShape).background(if (isConnected) Color.Green else Color.Gray))
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(text = if (isConnected) "Connected" else "Not Connected", fontWeight = FontWeight.Bold)
                            if (connectedDeviceName != null) Text(text = connectedDeviceName, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))

                // Activity Dashboard
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(text = "Heart Rate", style = MaterialTheme.typography.labelLarge)
                                Text(
                                    text = "$heartRate BPM",
                                    style = MaterialTheme.typography.displaySmall,
                                    fontWeight = FontWeight.Bold,
                                    color = if (isRecording) Color.Red else MaterialTheme.colorScheme.onSurface
                                )
                            }
                            if (isRecording) {
                                Text(text = "REC", color = Color.Red, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                        }
                        
                        HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                        
                        Row(modifier = Modifier.fillMaxWidth()) {
                            ActivityStat(label = "Steps", value = steps.toString(), modifier = Modifier.weight(1f))
                            ActivityStat(label = "Dist (m)", value = distance.toString(), modifier = Modifier.weight(1f))
                            ActivityStat(label = "Cals", value = calories.toString(), modifier = Modifier.weight(1f))
                        }

                        if (isRecording) {
                            HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                            Text(text = "File: $currentFilePath", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Real-time Log Preview
            if (isRecording || recentLogs.isNotEmpty()) {
                item {
                    Text(text = "Live Log Window", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Card(
                        modifier = Modifier.fillMaxWidth().height(200.dp),
                        colors = CardDefaults.cardColors(containerColor = Color.White),
                        border = androidx.compose.foundation.BorderStroke(1.dp, Color.LightGray)
                    ) {
                        Column {
                            Row(modifier = Modifier.fillMaxWidth().background(Color(0xFFF5F5F5)).padding(8.dp)) {
                                Text(text = "Time", modifier = Modifier.weight(1f), fontWeight = FontWeight.Bold, fontSize = 11.sp)
                                Text(text = "Data", modifier = Modifier.weight(3f), fontWeight = FontWeight.Bold, fontSize = 11.sp)
                            }
                            LazyColumn(modifier = Modifier.fillMaxSize()) {
                                items(recentLogs) { entry ->
                                    val parts = entry.split(" at ")
                                    Row(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                                        Text(text = if (parts.size > 1) parts[1] else "--:--", modifier = Modifier.weight(1f), fontSize = 11.sp)
                                        Text(text = if (parts.size > 1) parts[0] else entry, modifier = Modifier.weight(3f), fontSize = 11.sp)
                                    }
                                    HorizontalDivider(thickness = 0.5.dp, color = Color(0xFFEEEEEE))
                                }
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(20.dp))
                }
            }

            // History Section
            if (historyFiles.isNotEmpty()) {
                item {
                    Text(text = "Historical Log Files", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                items(historyFiles) { file ->
                    HistoryFileItem(file = file, onClick = { onFileClick(file) })
                }
                item { Spacer(modifier = Modifier.height(16.dp)) }
            }

            // Scanner Section
            if (!isConnected && !isConnectionPending && !isRecording) {
                item {
                    Text(text = "Nearby Devices:", style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                items(devices) { device ->
                    DeviceCard(device = device, onClick = { onDeviceClick(device) })
                }
                item {
                    Button(onClick = onScanClick, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                        Text("Scan Devices")
                    }
                }
            }
            
            // Build Info Footer
            item {
                Spacer(modifier = Modifier.height(24.dp))
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Version: 1.2.5 (Stable Build)",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray
                    )
                    Text(
                        text = "Build Date: Sep 05, 2026 - 20:55",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.LightGray
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
            }
        }

        // Actions Bottom Bar
        if (isConnected || isRecording) {
            Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Button(
                    onClick = onStartRecording,
                    enabled = !isRecording && isConnected,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text("START")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = onStopRecording,
                    enabled = isRecording,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(8.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                ) {
                    Text("STOP")
                }
            }
        }
    }
}

@Composable
fun IgnoreButton(label: String, isIgnored: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Button(
        onClick = onClick,
        modifier = modifier.height(36.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 4.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isIgnored) Color.Red else MaterialTheme.colorScheme.primary,
            contentColor = Color.White
        ),
        shape = RoundedCornerShape(4.dp)
    ) {
        Text(text = if (isIgnored) "SKIP $label" else "USE $label", fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ActivityStat(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        Text(text = value, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun HistoryFileItem(file: File, onClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp).clickable { onClick() },
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = androidx.compose.foundation.BorderStroke(0.5.dp, Color.LightGray)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = "📄", fontSize = 20.sp)
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(text = file.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                val date = Date(file.lastModified())
                val dateStr = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault()).format(date)
                val sizeStr = "${file.length() / 1024} KB"
                Text(text = "$dateStr • $sizeStr", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        }
    }
}

@Composable
fun DeviceCard(device: BleDevice, onClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        onClick = onClick
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(text = device.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(text = device.address, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
fun SetupItem(
    label: String,
    isDone: Boolean,
    onClick: (() -> Unit)?,
    showManualTip: Boolean = false
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .then(if (onClick != null) Modifier.clickable { onClick() } else Modifier),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = if (isDone) "✅" else "❌", fontSize = 16.sp)
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                fontWeight = if (isDone) FontWeight.Normal else FontWeight.Bold,
                color = if (isDone) Color.Gray else Color.Black
            )
            if (showManualTip) {
                Text(
                    text = "Tip: Open Recents and tap App Icon > 'Lock this app'.",
                    style = MaterialTheme.typography.labelSmall,
                    fontSize = 10.sp,
                    color = Color.DarkGray
                )
            }
        }
    }
}
