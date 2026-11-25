# Noise HR Logger - Technical Codebase Documentation (v1.2.5)

This document provides a comprehensive breakdown of every file and logic block in the Noise HR Logger project. Updated to include Workout Mode, Reliability Overhaul, and Version Tracking.

---

## 🏗️ Architecture Overview

The app follows a **Clean Architecture** pattern with **MVVM (Model-View-ViewModel)** and **Service-based background processing**:

1.  **UI Layer (`userinterface/`)**: Declarative UI built with Jetpack Compose. Includes a **Version Indicator** to ensure deployment success.
2.  **ViewModel Layer (`viewmodel/`)**: Orchestrates data flow between the Service and the UI.
3.  **Service Layer (`service/`)**: The "brain" that lives in the background, handling Bluetooth lifecycle, data aggregation, and timers.
4.  **BLE Core (`ble/`)**: Low-level drivers for Bluetooth scanning, connecting, and decoding specific watch protocols.
5.  **Storage Layer (`csv/`)**: Handles file writing and session persistence.

---

## 🗺️ Feature Map

| Feature | Primary Files |
| :--- | :--- |
| **Bluetooth Scanning** | [BLEScanner.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/BLEScanner.kt) |
| **Reliable Connection** | [BLEManager.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/BLEManager.kt), [LoggingService.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingService.kt) |
| **Heart Rate Parsing** | [HeartRateManager.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/HeartRateManager.kt) |
| **Step/Activity Parsing** | [ActivityManager.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/ActivityManager.kt) |
| **Data Aggregation (30s)** | [LoggingService.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingService.kt) |
| **CSV Logging** | [CSVLogger.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/csv/CSVLogger.kt) |
| **Live Processing Lab** | [MainScreen.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/userinterface/MainScreen.kt), [LoggingState.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingState.kt) |
| **Workout Mode (Ignore)** | [MainScreen.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/userinterface/MainScreen.kt), [LoggingService.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingService.kt) |

---

## 📄 Detailed File Breakdown

### 1. BLE Core Components

#### [BLEManager.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/BLEManager.kt)
**Summary:** The central engine for Bluetooth Low Energy (BLE) communication.

*   **`gattCallback`**:
    *   **`onConnectionStateChange`**: Now aggressively calls `close()` on **any non-zero status**. This prevents "Status 1" resource leaks that cause infinite reconnection loops.
    *   **`onDescriptorWrite`**: New callback to track the status of notification enablement.
*   **`close()`**: Fully wipes the GATT handle. Critical for hardware-level recovery.
*   **`connect()`**: Updated to support persistent transport types for wearables.

#### [ActivityManager.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/ble/ActivityManager.kt)
**Summary:** Decodes raw binary bytes from Noise/ColorFit watches.

---

### 2. Background Processing Layer

#### [LoggingService.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingService.kt)
**Summary:** A "Foreground Service" managing intervals and reliability.

*   **`onDisconnected`**: Includes a **5-second cooldown** delay before retrying Stage 1. This stops the "oscillation" flickering when the watch is on the edge of signal range.
*   **`setupCallbacks`**: Implements **Sequential Enablement**. Instead of enabling HR and Steps at once, it waits 1.2s between them to avoid `GATT_BUSY` hardware locks.
*   **`flushAggregatedData()`**: Implements **Workout Mode filtering**. If the user taps "Ignore Steps," this function forces the delta to **0** before writing to the CSV.
*   **`watchdogCheck()`**: Reduced to **2 minutes**. Rings the alarm faster if data stops flowing.

#### [LoggingState.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/service/LoggingState.kt)
**Summary:** Global state container.
*   **`isIgnoringSteps/Dist/Cal`**: Boolean flags used to filter out false movement signals (e.g., cycling).

---

### 3. User Interface & ViewModel

#### [BluetoothViewModel.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/viewmodel/BluetoothViewModel.kt)
**Summary:** Bridge between Service and UI.
*   **`toggleIgnoreX()`**: Methods to switch the Workout Mode flags on/off.

#### [MainScreen.kt](file:///C:/Users/admin/Downloads/NoiseHRLogger/app/src/main/java/com/example/noisehrlogger/userinterface/MainScreen.kt)
**Summary:** The dashboard.
*   **`🧪 Activity Controls`**: Three buttons (Steps, Dist, Cals) that turn **RED** when ignoring data.
*   **Build Info Footer**: Displays the version (v1.2.5) and build time at the bottom so you can verify the latest code is actually running.

---

### 💡 Feature Highlight: Reconnection Robustness

**Problem:** Infinite loop or flickering connection when moving away from the phone.
**Solution:**
1.  **Error-Status 1 Protection**: We now force-close the GATT handle on *every* failure, not just 133.
2.  **Sequential Descriptor Writes**: We wait 1.2s between enabling HR and Step notifications, which significantly reduces watch-side firmware crashes.
3.  **Backoff Timers**: The 5s and 10s delays allow the Android OS internal Bluetooth stack to "rest" between attempts.
