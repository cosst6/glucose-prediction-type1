package com.example.glucosealarms

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * v1.2 simple UI: 4 tabs (Now / Log / History / Models). No typing anywhere: food is picked with +/- steppers,
 * insulin and readings with steppers. All existing features kept (LOW-1 gate, prompts, scoring, CSV share, model picker).
 * Research only. Never use for dosing.
 */
@Composable
fun AppScreen() {
    val ctx = LocalContext.current.applicationContext
    var tab by remember { mutableIntStateOf(0) }
    var tick by remember { mutableIntStateOf(0) }
    var state by remember { mutableStateOf<AlarmStore.State?>(null) }
    val refresh: () -> Unit = { tick++ }
    LaunchedEffect(tick) {
        withContext(Dispatchers.IO) { AlarmEngine.runNow(ctx, if (tick == 0) "app opened" else "update"); AlarmEngine.scheduleNext(ctx) }
        while (true) { state = withContext(Dispatchers.IO) { AlarmStore.load(ctx) }; delay(20_000L) }
    }
    val tabs = listOf("◉" to "Now", "＋" to "Log", "☰" to "History", "⚙" to "Models")
    Scaffold(bottomBar = {
        NavigationBar {
            tabs.forEachIndexed { i, (g, l) ->
                NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = { Text(g, fontSize = 20.sp) }, label = { Text(l) })
            }
        }
    }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(14.dp)) {
            Text("Glucose Alarms", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Text("Research only · never for dosing", color = RED, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            val s = state
            if (s == null) Text("Loading…") else when (tab) {
                0 -> NowTab(ctx, s, refresh) { tab = 1 }
                1 -> LogTab(ctx, s, refresh)
                2 -> HistoryTab(ctx, s, refresh)
                else -> ModelsTab(ctx, s, refresh)
            }
            Spacer(Modifier.height(12.dp))
            Text("Version 1.3 · models trained 14–15 Nov 2026", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        }
    }
}

/* ---------- small building blocks ---------- */

@Composable
private fun Stepper(value: String, onMinus: () -> Unit, onPlus: () -> Unit, big: Boolean = false) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        FilledTonalButton(onClick = onMinus) { Text("−", fontSize = 20.sp) }
        Text(value, modifier = Modifier.width(if (big) 120.dp else 84.dp), textAlign = TextAlign.Center,
            style = if (big) MaterialTheme.typography.headlineMedium else MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        FilledTonalButton(onClick = onPlus) { Text("+", fontSize = 20.sp) }
    }
}

/** Big reading stepper: ±1 and ±10, 20–600 mg/dL. */
@Composable
private fun ReadingStepper(v: Int, set: (Int) -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        Stepper("$v", { set((v - 1).coerceIn(20, 600)) }, { set((v + 1).coerceIn(20, 600)) }, big = true)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { set((v - 10).coerceIn(20, 600)) }) { Text("−10") }
            Text("mg/dL", modifier = Modifier.align(Alignment.CenterVertically), color = Color.Gray)
            OutlinedButton(onClick = { set((v + 10).coerceIn(20, 600)) }) { Text("+10") }
        }
    }
}

private val WHEN = listOf(-30 to "−30m", -15 to "−15m", 0 to "Now", 30 to "+30m", 60 to "+1h", 120 to "+2h")

@Composable
private fun WhenChips(offset: Int, set: (Int) -> Unit) {
    Text("When", style = MaterialTheme.typography.labelLarge)
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        WHEN.forEach { (m, l) -> FilterChip(selected = offset == m, onClick = { set(m) }, label = { Text(l) }) }
    }
}

private fun lastStick(s: AlarmStore.State): Int =
    s.events.filter { it.type == "stick" && it.units != null && it.t <= System.currentTimeMillis() }.maxByOrNull { it.t }?.units?.toInt() ?: 120

private fun bg(f: () -> Unit) = Thread(f).start()

/* ---------- Now ---------- */

@Composable
private fun NowTab(ctx: Context, s: AlarmStore.State, refresh: () -> Unit, goLog: () -> Unit) {
    val now = System.currentTimeMillis()
    val open = s.prompts.filter { it.doneMs == null && it.dueMs <= now + 60_000 && it.dueMs > now - 60 * 60_000L }
    open.forEach { p ->
        Card(Modifier.fillMaxWidth().padding(vertical = 6.dp), colors = CardDefaults.cardColors(containerColor = AMBER)) {
            Column(Modifier.padding(12.dp)) {
                Text("Check your blood glucose now", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(p.text, style = MaterialTheme.typography.bodySmall)
                var v by remember(p.id) { mutableIntStateOf(lastStick(s)) }
                ReadingStepper(v) { v = it }
                Row {
                    Button(onClick = { bg { AlarmEngine.log(ctx, "stick", "", v.toDouble(), System.currentTimeMillis()); AlarmEngine.runNow(ctx, "fingerstick"); refresh() } },
                        modifier = Modifier.weight(1f)) { Text("Save $v") }
                    TextButton(onClick = { bg { AlarmEngine.dismissPrompt(ctx, p.id); refresh() } }) { Text("Skip") }
                }
            }
        }
    }
    s.prompts.filter { it.doneMs == null && it.dueMs > now + 60_000 }
        .forEach { Text("Reminder at ${hhmm(it.dueMs)}: check your blood glucose", style = MaterialTheme.typography.bodySmall) }

    val shown = ModelCatalog.shown(ctx)
    if (shown.isEmpty()) Text("No models switched on. Turn some on in the Models tab.", color = RED)
    val cards = listOf<Pair<String, @Composable () -> Unit>>("low1" to { LowCard(s) }, "spike1" to { SpikeCard(s) },
        "spike2" to { TimedCard(s, "spike2") }, "rise40" to { TimedCard(s, "rise40") },
        "night1" to { NightCard(s) { v -> bg { AlarmEngine.runNight(ctx, v.toDouble()); refresh() } } })
    cards.filter { it.first in shown }.forEach { (_, c) ->
        Card(Modifier.fillMaxWidth().padding(vertical = 5.dp)) { Column(Modifier.padding(12.dp)) { c() } }
    }
    Spacer(Modifier.height(6.dp))
    Text("⌚ " + WatchReader.status.ifEmpty { "Watch: not read yet." }, style = MaterialTheme.typography.bodySmall,
        color = if (WatchReader.hasAccess()) Color.Gray else RED)
    Spacer(Modifier.height(8.dp))
    Button(onClick = goLog, modifier = Modifier.fillMaxWidth()) { Text("＋ Log meal / insulin / reading") }
}

/* ---------- Log ---------- */

@Composable
private fun LogTab(ctx: Context, s: AlarmStore.State, refresh: () -> Unit) {
    var kind by remember { mutableStateOf("meal") }
    var status by remember { mutableStateOf("") }
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf("meal" to "Meal", "insulin" to "Insulin", "reading" to "Reading", "treat" to "Low treat").forEach { (k, l) ->
            FilterChip(selected = kind == k, onClick = { kind = k; status = "" }, label = { Text(l) })
        }
    }
    Spacer(Modifier.height(6.dp))
    val done: (String) -> Unit = { status = it; refresh() }
    when (kind) {
        "meal" -> MealForm(ctx, false, done)
        "treat" -> MealForm(ctx, true, done)
        "insulin" -> InsulinForm(ctx, done)
        else -> ReadingForm(ctx, lastStick(s), done)
    }
    if (status.isNotEmpty()) Text(status, color = Color(0xFF2E7D32), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
    Text("Saved on this phone only (no cloud).", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}

@Composable
private fun MealForm(ctx: Context, treat: Boolean, done: (String) -> Unit) {
    val qty = remember(treat) { mutableStateMapOf<String, Double>() }
    var more by remember { mutableStateOf(false) }
    var units by remember(treat) { mutableDoubleStateOf(0.0) }
    var offset by remember(treat) { mutableIntStateOf(0) }
    Text(if (treat) "What did you take for the low?" else "What are you eating?", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val list = FoodChoices.ALL.filter { more || it.common || (qty[it.key] ?: 0.0) > 0.0 }
    list.forEach { f ->
        val q = qty[f.key] ?: 0.0
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(vertical = 1.dp)) {
            Text(f.label, modifier = Modifier.weight(1f), fontWeight = if (q > 0) FontWeight.Bold else FontWeight.Normal)
            Stepper(FoodChoices.show(f, q), { qty[f.key] = FoodChoices.down(f, q) }, { qty[f.key] = FoodChoices.up(f, q) })
        }
    }
    TextButton(onClick = { more = !more }) { Text(if (more) "Fewer foods" else "More foods…") }
    val text = FoodChoices.text(qty)
    if (text.isNotEmpty()) {
        val tot = MealParser.parse(text).second
        Text("≈ ${tot.avail.toInt()} g carbs · ${tot.fat.toInt()} g fat · ${tot.protein.toInt()} g protein", style = MaterialTheme.typography.bodyMedium)
    }
    if (!treat) {
        HorizontalDivider(Modifier.padding(vertical = 6.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("NovoRapid with meal", modifier = Modifier.weight(1f))
            Stepper(if (units == 0.0) "0 u" else "${FoodChoices.num(units)} u", { units = (units - 0.5).coerceAtLeast(0.0) }, { units = (units + 0.5).coerceAtMost(30.0) })
        }
    }
    WhenChips(offset) { offset = it }
    Button(enabled = text.isNotEmpty(), modifier = Modifier.fillMaxWidth(), onClick = {
        val t = System.currentTimeMillis() + offset * 60_000L
        val u = units; val tx = text; val type = if (treat) "low_treat" else "food"
        qty.clear(); units = 0.0; offset = 0
        bg {
            AlarmEngine.log(ctx, type, tx, null, t)
            if (u > 0) AlarmEngine.log(ctx, "rapid", "", u, t)
            AlarmEngine.runNow(ctx, "new log")
            done("Saved ${if (treat) "low treat" else "meal"} at ${hhmm(t)}${if (u > 0) " + ${FoodChoices.num(u)} u" else ""}${if (t > System.currentTimeMillis()) " (planned)" else ""}. See Now tab.")
        }
    }) { Text(if (offset > 0) "Save planned" else "Save") }
}

@Composable
private fun InsulinForm(ctx: Context, done: (String) -> Unit) {
    var type by remember { mutableStateOf("rapid") }
    var units by remember { mutableDoubleStateOf(2.0) }
    var offset by remember { mutableIntStateOf(0) }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf("rapid" to "NovoRapid", "correction" to "Correction", "lantus" to "Lantus").forEach { (k, l) ->
            FilterChip(selected = type == k, onClick = { type = k; units = if (k == "lantus") 28.0 else 2.0 }, label = { Text(l) })
        }
    }
    val step = if (type == "lantus") 1.0 else 0.5
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Stepper("${FoodChoices.num(units)} u", { units = (units - step).coerceAtLeast(step) }, { units = (units + step).coerceAtMost(60.0) }, big = true)
    }
    WhenChips(offset) { offset = it }
    Button(modifier = Modifier.fillMaxWidth(), onClick = {
        val t = System.currentTimeMillis() + offset * 60_000L; val u = units; val ty = type
        offset = 0
        bg { AlarmEngine.log(ctx, ty, "", u, t); AlarmEngine.runNow(ctx, "new log")
            done("Saved ${FoodChoices.num(u)} u ${if (ty == "rapid") "NovoRapid" else if (ty == "lantus") "Lantus" else "correction"} at ${hhmm(t)}.") }
    }) { Text("Save") }
}

@Composable
private fun ReadingForm(ctx: Context, start: Int, done: (String) -> Unit) {
    var v by remember { mutableIntStateOf(start) }
    var offset by remember { mutableIntStateOf(0) }
    Text("Fingerstick reading", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    ReadingStepper(v) { v = it }
    Text("A reading up to 20 min after you start eating still counts as your pre-meal reading.", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    Text("When", style = MaterialTheme.typography.labelLarge)
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf(-30 to "−30m", -15 to "−15m", 0 to "Now").forEach { (m, l) -> FilterChip(selected = offset == m, onClick = { offset = m }, label = { Text(l) }) }
    }
    Button(modifier = Modifier.fillMaxWidth(), onClick = {
        val t = System.currentTimeMillis() + offset * 60_000L; val r = v
        bg { AlarmEngine.log(ctx, "stick", "", r.toDouble(), t); AlarmEngine.runNow(ctx, "fingerstick"); done("Saved $r mg/dL at ${hhmm(t)}.") }
    }) { Text("Save $v") }
}

/* ---------- History ---------- */

@Composable
private fun HistoryTab(ctx: Context, s: AlarmStore.State, refresh: () -> Unit) {
    val now = System.currentTimeMillis()
    val names = mapOf("food" to "Meal", "rapid" to "NovoRapid", "correction" to "Correction", "lantus" to "Lantus",
        "low_treat" to "Low treat", "stick" to "Reading", "note" to "Note")
    Text("Last 12 h and planned", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val recent = s.events.filter { it.t > now - 12 * 3600_000L }.sortedByDescending { it.t }
    if (recent.isEmpty()) Text("Nothing logged.", color = Color.Gray)
    recent.forEach { e ->
        Row(verticalAlignment = Alignment.CenterVertically) {
            val what = listOfNotNull(names[e.type] ?: e.type, e.text.takeIf { it.isNotBlank() },
                e.units?.let { if (e.type == "stick") "${it.toInt()} mg/dL" else "${FoodChoices.num(it)} u" }).joinToString(" · ")
            Text("${hhmm(e.t)}  $what${if (e.t > now) "  (planned)" else ""}", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            TextButton(onClick = { bg { AlarmEngine.delete(ctx, e); AlarmEngine.runNow(ctx, "deleted"); refresh() } }) { Text("Delete") }
        }
    }
    HorizontalDivider(Modifier.padding(vertical = 8.dp))
    History(s)
    Spacer(Modifier.height(6.dp))
    OutlinedButton(onClick = { share(ctx) }) { Text("Share log (CSV)") }
}

/* ---------- Models / settings ---------- */

@Composable
private fun ModelsTab(ctx: Context, s: AlarmStore.State, refresh: () -> Unit) {
    val p = ctx.getSharedPreferences(AlarmEngine.PREFS, Context.MODE_PRIVATE)
    var shown by remember { mutableStateOf(ModelCatalog.shown(ctx)) }
    Text("Models to show", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    Text("Off = no card and no notification (it still runs quietly for scoring).", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    ModelCatalog.ALL.forEach { m ->
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
            Column(Modifier.weight(1f)) {
                Text("${m.short} · ${m.title}", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
                Text(m.what, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
            }
            Switch(m.key in shown, { on -> shown = if (on) shown + m.key else shown - m.key; ModelCatalog.setShown(ctx, shown) })
        }
    }
    HorizontalDivider(Modifier.padding(vertical = 8.dp))
    var low by remember { mutableStateOf(p.getBoolean("lowNotify", true)) }
    var gate by remember { mutableStateOf(p.getBoolean("lowGate", true)) }
    var spike by remember { mutableStateOf(p.getBoolean("spikeNotify", true)) }
    SwitchRow("Low alarm notifications", low) { low = it; p.edit().putBoolean("lowNotify", it).apply() }
    SwitchRow("Low alarm only when ≥0.5 u NovoRapid is on board (fewer false alarms)", gate) { gate = it; p.edit().putBoolean("lowGate", it).apply() }
    SwitchRow("Meal notifications (only models switched on)", spike) { spike = it; p.edit().putBoolean("spikeNotify", it).apply() }
    HorizontalDivider(Modifier.padding(vertical = 8.dp))
    Text("Watch data (Noise HR Logger)", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
    val access = WatchReader.hasAccess()
    Text(if (access) "✓ Can read Documents/NoiseHR (read only). Files found: ${WatchReader.filesFound}" +
            (if (WatchReader.firstFileDay.isNotEmpty()) ", from ${WatchReader.firstFileDay}" else "") + "."
         else "Needs \"All files access\" to read the Noise app's files. It only reads; it never changes them.",
        style = MaterialTheme.typography.bodySmall, color = if (access) Color(0xFF2E7D32) else RED)
    Text(WatchReader.status, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    if (!access && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) Button(onClick = {
        val i = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${ctx.packageName}")).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        try { ctx.startActivity(i) } catch (e: Exception) {
            ctx.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
    }) { Text("Allow All files access") }
    HorizontalDivider(Modifier.padding(vertical = 8.dp))
    Text("${s.lastRunNote}. Every alarm is logged even with notifications off.", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    OutlinedButton(onClick = refresh) { Text("Check now") }
    Text("Models: ${ModelParams.LOW_ID.substringBefore('_')}, ${ModelParams.SPIKE_ID.substringBefore('_')} (14 Nov), ${TimingParams.SPIKE2_ID.substringBefore('_')} v2 and " +
        "${TimingParams.RISE40_ID.substringBefore('_')} (15 Nov). Files: Documents/GlucoseAlarms/", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
}

@Composable
private fun SwitchRow(label: String, on: Boolean, set: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f)); Switch(on, set)
    }
}
