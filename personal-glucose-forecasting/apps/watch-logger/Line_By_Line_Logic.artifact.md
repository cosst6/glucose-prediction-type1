# Line-by-Line Code Logic Guide (v1.2.5)

This guide breaks down the core functional blocks of the Noise HR Logger to explain exactly what happens at the code level.

---

## 1. Connection Robustness (`LoggingService.kt` & `BLEManager.kt`)

### Aggressive Cleanup (Preventing Status 1 Loops)
```kotlin
// In BLEManager.kt > onConnectionStateChange
if (status != BluetoothGatt.GATT_SUCCESS) {
    Log.e(TAG, "GATT Error status: $status. Force closing GATT handle.")
    onDisconnected?.invoke(status)
    close() // Wipes the resource handle immediately
    return
}
```
*   **Simple:** If *anything* goes wrong during connection, kill the connection immediately.
*   **Detailed:** "Status 1" often happens because a previous failed connection is still partially "alive" in the phone's memory. By calling `close()` on every non-zero status, we ensure that the Stage 1 recovery always starts with a 100% clean memory slate.

### Reconnection Cooldown (Preventing Oscillation)
```kotlin
// In LoggingService.kt > onDisconnected
serviceScope.launch {
    delay(5000) // 5s Cooldown
    attemptRecovery()
}
```
*   **Simple:** Wait 5 seconds before trying to reconnect after a drop.
*   **Detailed:** If you move away from the phone, the watch will constantly connect and disconnect as the signal flickers. The 5s delay gives the signal time to either stabilize or stay dead, preventing a high-frequency loop that drains battery and crashes the watch firmware.

### Sequential Notification Enablement
```kotlin
// In LoggingService.kt > setupCallbacks
serviceScope.launch {
    heartRateManager.enableHeartRateNotifications()
    delay(1200) // Staggered delay
    activityManager.enableActivityNotifications()
}
```
*   **Simple:** Turn on pulse monitoring first, wait a second, then turn on step monitoring.
*   **Detailed:** BLE commands share a single "queue." Sending two `writeDescriptor` calls at the exact same millisecond causes `GATT_BUSY` errors. The 1.2s delay ensures the watch has finished processing the Pulse request before we send the Step request.

---

## 2. Workout Mode Filtering (`LoggingService.kt`)

### The "Ignore" Logic
```kotlin
val deltaSteps = if (LoggingState.isIgnoringSteps.value) 0 else latestSteps - prevLoggedSteps
```
*   **Simple:** If "Ignore Steps" is on, record 0 regardless of what the watch says.
*   **Detailed:** This checks the `isIgnoringSteps` StateFlow. If true, it skips the math and sends a literal `0` to the CSV Logger. This is the "fix" for false step signals during cycling.

---

## 3. Data Watchdog (`LoggingService.kt`)

### 2-Minute Alert
```kotlin
now - lastSuccessfulReadingTime > 2 * 60 * 1000L
```
*   **Simple:** Ring the alarm if no data comes for 2 minutes.
*   **Detailed:** Changed from 5 minutes (300,000ms) to 2 minutes (120,000ms). This provides much faster feedback if the watch is out of range or the sensor has stopped working.

---

## 4. UI Versioning (`MainScreen.kt`)

### Deployment Verification
```kotlin
Text(text = "Version: 1.2.5 (Stable Build)")
Text(text = "Build Date: Sep 05, 2026 - 20:55")
```
*   **Simple:** Shows you exactly which version is running.
*   **Detailed:** This is your tool to solve the "reverting version" problem. If you restart the phone and see an older date/version here, you know Android has rolled back the app, and you need to perform a "Full Run" from Android Studio again.
