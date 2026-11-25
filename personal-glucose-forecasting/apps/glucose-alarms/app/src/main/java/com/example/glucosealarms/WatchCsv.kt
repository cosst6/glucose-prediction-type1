package com.example.glucosealarms

import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Pure Kotlin (no Android): turns Noise HR Logger CSV rows into the watch inputs the models use, and fills SHORT gaps.
 * CSV row (every 30 s): "yyyy-MM-dd HH:mm:ss,heart_rate,steps,distance_m,calories" where steps/calories are 30-s amounts.
 * When "Ignore steps" is on in the Noise app it writes steps as 0, so ignored steps are never seen or used here.
 * Gap filling: if rows stop for up to MAX_GAP_MIN minutes, the missing 30-s slots get a straight line between the last row
 * before and the first row after (heart rate, steps and calories). Longer gaps are left empty. Heart rate under 35 counts as missing.
 */
object WatchCsv {
    data class Row(val t: Long, val hr: Double?, val steps: Double, val cal: Double, val filled: Boolean = false)
    data class Summary(val watch: Watch, val lastRowMs: Long?, val filledSlots: Int, val rowsUsed: Int)

    const val SLOT_MS = 30_000L
    const val MAX_GAP_MIN = 10
    private val ROW = Regex("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2},")

    fun parse(lines: Sequence<String>): List<Row> {
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        return lines.mapNotNull { raw ->
            val l = raw.trim()
            if (!ROW.containsMatchIn(l)) return@mapNotNull null
            val p = l.split(',')
            if (p.size < 5) return@mapNotNull null
            val t = try { fmt.parse(p[0])?.time } catch (e: Exception) { null } ?: return@mapNotNull null
            val hr = p[1].toDoubleOrNull()?.takeIf { it in 35.0..220.0 }
            Row(t, hr, (p[2].toDoubleOrNull() ?: 0.0).coerceAtLeast(0.0), (p[4].toDoubleOrNull() ?: 0.0).coerceAtLeast(0.0))
        }.toList()
    }

    /** Sorted, one row per timestamp, with short gaps filled by straight lines. */
    fun fill(rowsIn: List<Row>, maxGapMin: Int = MAX_GAP_MIN): List<Row> {
        val rows = rowsIn.sortedBy { it.t }.distinctBy { it.t }
        if (rows.isEmpty()) return rows
        // 1) missing heart rate inside the data (e.g. 0 bpm rows): straight line between good neighbours if close enough
        val hr = rows.map { it.hr }.toMutableList()
        var i = 0
        while (i < rows.size) {
            if (hr[i] == null) {
                var j = i; while (j < rows.size && hr[j] == null) j++
                if (i > 0 && j < rows.size && rows[j].t - rows[i - 1].t <= maxGapMin * 60_000L) {
                    for (k in i until j) { val f = (rows[k].t - rows[i - 1].t).toDouble() / (rows[j].t - rows[i - 1].t); hr[k] = hr[i - 1]!! + f * (hr[j]!! - hr[i - 1]!!) }
                }
                i = j
            } else i++
        }
        val base = rows.mapIndexed { k, r -> r.copy(hr = hr[k]) }
        // 2) missing rows (time gaps): add 30-s slots on a straight line, only for short gaps
        val out = ArrayList<Row>(base.size)
        for (k in base.indices) {
            val r = base[k]
            if (k > 0) {
                val a = base[k - 1]; val gap = r.t - a.t
                if (gap > SLOT_MS * 3 / 2 && gap <= maxGapMin * 60_000L) {
                    var t = a.t + SLOT_MS
                    while (t < r.t - SLOT_MS / 2) {
                        val f = (t - a.t).toDouble() / gap
                        val h = if (a.hr != null && r.hr != null) a.hr + f * (r.hr - a.hr) else null
                        out.add(Row(t, h, a.steps + f * (r.steps - a.steps), a.cal + f * (r.cal - a.cal), filled = true))
                        t += SLOT_MS
                    }
                }
            }
            out.add(r)
        }
        return out
    }

    /** HR mean over the last 15 min, raw step and calorie totals over the last 60 min (and steps over 3 h for NIGHTLOW-1), at [now]. */
    fun summarize(rowsIn: List<Row>, now: Long): Summary {
        val rows = fill(rowsIn.filter { it.t > now - 3 * 3600_000L && it.t <= now + 60_000L })
        val h = rows.filter { it.t > now - 15 * 60_000L && it.t <= now + 60_000L }.mapNotNull { it.hr }
        val w = rows.filter { it.t > now - 60 * 60_000L && it.t <= now + 60_000L }
        return Summary(Watch(if (h.isEmpty()) null else h.average(), w.sumOf { it.steps }, w.sumOf { it.cal }, rows.sumOf { it.steps }),
            rows.filter { !it.filled }.maxOfOrNull { it.t }, w.count { it.filled }, w.size)
    }
}
