package com.example.glucosealarms

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal val RED = Color(0xFFB00020)
internal val AMBER = Color(0xFFFFF4CC)

internal val hm = SimpleDateFormat("h:mm a", Locale.getDefault())
internal fun hhmm(ms: Long) = hm.format(Date(ms))
internal fun pct(p: Double) = "${(p * 100).toInt()}%"

/** SPIKE-2 or RISE40 answer for the latest meal. */
@Composable
internal fun TimedCard(s: AlarmStore.State, kind: String) {
    val info = ModelCatalog.ALL.first { it.key == kind }
    Text("${info.title} · ${info.short}", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val a = s.alarms.filter { it.kind == kind }.maxByOrNull { it.targetMs }
    if (a == null || System.currentTimeMillis() - a.targetMs > 6 * 3600_000L) {
        Text("Log a meal (and NovoRapid) to get an answer. No reading after the meal is needed.", style = MaterialTheme.typography.bodySmall); return
    }
    Text("Meal ${hhmm(a.targetMs)}: ${a.inputs.substringAfter("meal=").substringBefore(';').take(60)}", style = MaterialTheme.typography.bodySmall)
    Text(a.text, color = if (a.alarm) RED else Color(0xFF2E7D32), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
    Text(if (a.step == 1) "Used your pre-meal reading." else "No reading used (a pre-meal reading makes the size better).", style = MaterialTheme.typography.bodySmall)
    a.outcome?.let { Text("Result: $it" + (a.actual?.let { r -> " (real rise ${r.toInt()})" } ?: ""), style = MaterialTheme.typography.bodySmall) }
    Text(if (kind == "rise40") "Tested: warns before about 8 of 10 rises of 40+ (Libre days), 1 in 5 warnings false. On fingerstick days weaker (5 of 11). " +
            "Ranges held the real time 7 of 10 times. Start range is short (~30 min); the highest point is hard to time (range ~3 h)."
         else "Tested: size right about half the time. Ranges held the real time 7 of 10 times; peak range is wide (~3 h) because peak time is hard to predict.",
        style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}

@Composable
internal fun LowCard(s: AlarmStore.State) {
    Text("Low (<70) alarm · LOW-1", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val l = s.lastLow
    if (l == null) { Text("Not run yet.", style = MaterialTheme.typography.bodySmall); return }
    val p120 = l.optDouble("p120")
    val gateOn = LocalContext.current.getSharedPreferences(AlarmEngine.PREFS, Context.MODE_PRIVATE).getBoolean("lowGate", true)
    val iob = l.optDouble("iob", 0.0)
    val gated = gateOn && iob < 0.5
    val alarm = p120 > ModelParams.LOW_THRESHOLD && !gated
    Text("Chance of a low in 2 h: ${pct(p120)}" + if (alarm) "  ⚠ ALARM" else "", color = if (alarm) RED else Color.Unspecified,
        fontWeight = if (alarm) FontWeight.Bold else FontWeight.Normal, style = MaterialTheme.typography.bodyLarge)
    Text("1 h: ${pct(l.optDouble("p60"))}   3 h: ${pct(l.optDouble("p180"))}   (no fingerstick needed)", style = MaterialTheme.typography.bodySmall)
    if (!l.isNull("p120_stick")) Text("Using your last reading ${l.optDouble("stick").toInt()} mg/dL (${l.optDouble("stick_age").toInt()} min ago): ${pct(l.optDouble("p120_stick"))}",
        style = MaterialTheme.typography.bodySmall)
    if (p120 > ModelParams.LOW_THRESHOLD && gated) Text("No alarm: less than 0.5 u NovoRapid on board (see Models tab).", style = MaterialTheme.typography.bodySmall)
    Text("Alarm when the 2 h chance is over 50% and NovoRapid is on board. Tested: caught 8 of 10 lows, about 1 in 4 checks a false alarm. Based on only 10 lows.",
        style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}

@Composable
internal fun SpikeCard(s: AlarmStore.State) {
    Text("Meal spike alarm · SPIKE-1", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val a = s.alarms.filter { it.kind == "spike" }.maxByOrNull { it.targetMs }
    if (a == null || System.currentTimeMillis() - a.targetMs > 6 * 3600_000L) {
        Text("Log a meal (and NovoRapid) to get a spike forecast.", style = MaterialTheme.typography.bodySmall); return
    }
    Text("Meal ${hhmm(a.targetMs)}: ${a.text.take(60)}", style = MaterialTheme.typography.bodySmall)
    Text(ModelParams.SPIKE_BANDS[a.band], color = if (a.band >= 1) RED else Color(0xFF2E7D32), fontWeight = FontWeight.Bold,
        style = MaterialTheme.typography.bodyLarge)
    Text("Rise ≥30: ${pct(a.p)}   ≥60: ${pct(a.p2 ?: 0.0)}   ·   " + when (a.step) { 0 -> "no reading used"; 1 -> "used pre-meal reading"; else -> "used pre-meal + 45-min readings" },
        style = MaterialTheme.typography.bodySmall)
    a.outcome?.let { Text("Result: $it" + (a.actual?.let { r -> " (real rise ${r.toInt()})" } ?: ""), style = MaterialTheme.typography.bodySmall) }
    Text("Uses carbs not covered by insulin (carbs − 8 g × units). Weakest on big spikes (≥60).", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}

@Composable
internal fun History(s: AlarmStore.State) {
    Text("Alarm log (for scoring)", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val lows = s.alarms.filter { it.kind == "low" }
    val seen = s.alarms.filter { it.kind == "low_seen" }
    val spikes = s.alarms.filter { it.kind == "spike" && it.outcome != null && it.actual != null }
    val r40 = s.alarms.filter { it.kind == "rise40" && it.actual != null }
    Text("Low alarms ${lows.size}: right ${lows.count { it.outcome?.startsWith("low") == true }}, false ${lows.count { it.outcome?.startsWith("no low") == true }}, " +
        "no reading ${lows.count { it.outcome?.contains("no reading") == true }}.  Real lows ${seen.size}: warned ${seen.count { it.outcome == "warned before" }}.",
        style = MaterialTheme.typography.bodySmall)
    Text("Meals scored ${spikes.size}: band right ${spikes.count { it.outcome == "band right" }}.", style = MaterialTheme.typography.bodySmall)
    Text("RISE40 scored ${r40.size}: right ${r40.count { it.outcome == "right" }}, false ${r40.count { it.outcome == "false alarm" }}, missed ${r40.count { it.outcome == "missed rise" }}.",
        style = MaterialTheme.typography.bodySmall)
    val nights = s.alarms.filter { it.kind == "night" && it.outcome != null && it.outcome != "no reading" }
    Text("Nights scored ${nights.size}: right ${nights.count { it.outcome == "low (alarm right)" || it.outcome == "no low (right)" }}, " +
        "false alarm ${nights.count { it.outcome!!.startsWith("no low (false") }}, missed ${nights.count { it.outcome == "low (missed)" }}.", style = MaterialTheme.typography.bodySmall)
    s.alarms.sortedByDescending { it.madeMs }.take(8).forEach { a ->
        val line = when (a.kind) {
            "low" -> "LOW ${pct(a.p)} for ${hhmm(a.targetMs)}"
            "spike" -> "MEAL ${ModelParams.SPIKE_BANDS[a.band]} (${a.text.take(25)})"
            "spike2" -> "SPIKE-2 ${ModelParams.SPIKE_BANDS[a.band]} for meal ${hhmm(a.targetMs)}"
            "night" -> "NIGHT ${if (a.alarm) "low likely" else "no low"}, wake ~${a.p2?.toInt()} (bed ${hhmm(a.madeMs)})"
            "rise40" -> "RISE40 ${if (a.alarm) "rise 40+" else "no 40+"} for meal ${hhmm(a.targetMs)}"
            else -> a.text
        }
        Text("${hhmm(a.madeMs)}  $line → ${a.outcome ?: "waiting"}", style = MaterialTheme.typography.labelSmall)
    }
}

internal fun share(ctx: Context) {
    val d = AlarmStore.dir(ctx) ?: return
    val files = listOf("alarms.csv", "low_checks.csv", "events.csv").map { File(d, it) }.filter { it.exists() }
    if (files.isEmpty()) return
    val uris = ArrayList(files.map { FileProvider.getUriForFile(ctx, "com.example.glucosealarms.provider", it) })
    val send = Intent(Intent.ACTION_SEND_MULTIPLE).setType("text/csv").putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    ctx.startActivity(Intent.createChooser(send, "Share alarm log").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}

/** NIGHTLOW-1: "Going to sleep" button (bedtime reading by stepper, no typing) + the last night's answer. */
@Composable
internal fun NightCard(s: AlarmStore.State, onGo: (Int) -> Unit) {
    Text("Night low + wake-up · NIGHTLOW-1", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val now = System.currentTimeMillis()
    val a = s.alarms.filter { it.kind == "night" }.maxByOrNull { it.madeMs }?.takeIf { now - it.madeMs < 10 * 3600_000L }
    if (a != null) {
        val lines = a.text.split("\n")
        Text("Bedtime ${hhmm(a.madeMs)}", style = MaterialTheme.typography.bodySmall)
        Text(lines[0], color = if (a.alarm) RED else Color(0xFF2E7D32), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyLarge)
        if (lines.size > 1) Text(lines[1], fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyLarge)
        a.outcome?.let { Text("Result: $it", style = MaterialTheme.typography.bodySmall) }
    }
    var open by remember { mutableStateOf(false) }
    if (!open) Button(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) { Text(if (a == null) "🌙 Going to sleep" else "🌙 Going to sleep (again)") }
    else {
        val last = s.events.filter { it.type == "stick" && it.units != null && it.t <= now }.maxByOrNull { it.t }?.units?.toInt() ?: 120
        var v by remember { mutableIntStateOf(last) }
        Text("Bedtime fingerstick", style = MaterialTheme.typography.labelLarge)
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(onClick = { v = (v - 10).coerceIn(20, 600) }) { Text("−10") }
            OutlinedButton(onClick = { v = (v - 1).coerceIn(20, 600) }) { Text("−") }
            Text("$v", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 10.dp))
            OutlinedButton(onClick = { v = (v + 1).coerceIn(20, 600) }) { Text("+") }
            OutlinedButton(onClick = { v = (v + 10).coerceIn(20, 600) }) { Text("+10") }
        }
        Row {
            Button(onClick = { open = false; onGo(v) }, modifier = Modifier.weight(1f)) { Text("Save $v and check the night") }
            TextButton(onClick = { open = false }) { Text("Cancel") }
        }
        Text("Log your bedtime milk / snack / NovoRapid first (planned up to 90 min after bed counts).", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    }
    Text("Tested on only 18 nights (2 real lows): caught 1 of 2, about 2 false alarms per 10 nights. Wake-up value often off by ~50. Check with your glucometer.",
        style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}
