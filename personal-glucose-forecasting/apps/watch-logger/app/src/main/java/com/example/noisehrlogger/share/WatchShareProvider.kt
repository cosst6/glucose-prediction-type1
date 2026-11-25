package com.example.noisehrlogger.share

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.Environment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Read-only, on-phone share of the watch rows this app already logs (Documents/NoiseHR_Session_*.csv),
 * so the separate "Glucose Alarms" app can use HR / steps / calories without any cloud.
 * Only apps signed with the same key (your Android Studio) may read it: signature permission READ_WATCH.
 * It never writes, never touches BLE / logging / alarms.
 *
 * content://com.example.noisehrlogger.watchshare/rows?since=<epoch ms>  ->  t_ms, hr, steps, cal  (max 26 h back)
 */
class WatchShareProvider : ContentProvider() {
    private val rowTime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

    override fun onCreate() = true

    override fun query(uri: Uri, projection: Array<out String>?, selection: String?, selectionArgs: Array<out String>?, sortOrder: String?): Cursor {
        val now = System.currentTimeMillis()
        val since = maxOf(uri.getQueryParameter("since")?.toLongOrNull() ?: (now - 3600_000L), now - 26 * 3600_000L)
        val out = MatrixCursor(arrayOf("t_ms", "hr", "steps", "cal"))
        val dir = context?.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: return out
        val files = dir.listFiles { f: File -> f.name.startsWith("NoiseHR_Session_") && f.name.endsWith(".csv") && f.lastModified() >= since } ?: return out
        val seen = HashSet<Long>()
        val rows = ArrayList<Array<Any>>()
        files.forEach { f ->
            try {
                f.forEachLine { line ->
                    val p = line.split(',')
                    if (p.size >= 3 && p[0].length == 19 && p[0][4] == '-') {
                        val ms = try { rowTime.parse(p[0])?.time } catch (e: Exception) { null } ?: return@forEachLine
                        if (ms < since || !seen.add(ms / 1000)) return@forEachLine
                        rows.add(arrayOf(ms, p[1].trim().toIntOrNull() ?: 0, p[2].trim().toIntOrNull() ?: 0, p.getOrNull(4)?.trim()?.toIntOrNull() ?: 0))
                    }
                }
            } catch (_: Exception) {}
        }
        rows.sortedBy { it[0] as Long }.forEach { out.addRow(it) }
        return out
    }

    override fun getType(uri: Uri): String = "vnd.android.cursor.dir/vnd.noisehrlogger.watchrow"
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0
}
