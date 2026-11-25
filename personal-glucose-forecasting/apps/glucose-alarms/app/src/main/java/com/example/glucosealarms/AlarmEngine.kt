package com.example.glucosealarms

import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Runs the research alarms (LOW-1, SPIKE-1, SPIKE-2, RISE40) on the phone, fully offline (no cloud at all).
 * Food, insulin and fingersticks: this app's own log. Watch HR / steps / calories: read-only from the Noise app's CSV files
 * in Documents/NoiseHR (needs All files access).
 * If the Noise app is not installed / not recording, the models run without watch data (HR filled like in training, steps and calories 0).
 *
 * Runs: every ~15 min (own AlarmManager wake-up, survives reboot) and right after anything is logged. Never throws.
 */
object AlarmEngine {
    private const val TAG = "AlarmEngine"
    private const val MIN = 60_000L
    private const val CHANNEL = "glucose_research_alarms"
    const val ACTION_TICK = "com.example.glucosealarms.ACTION_TICK"
    const val PREFS = "GlucoseAlarmPrefs"
    private const val LOW_HORIZON = 120                 // best tested warning time (8/10 lows caught, 25% false alarms)
    private const val LOW_REPEAT_MIN = 90L              // do not repeat a low alarm sooner than this
    private const val LOW_MIN_IOB = 0.5                 // units of NovoRapid on board needed for a low alarm (gate)
    private val busy = AtomicBoolean(false)
    private val idFmt = SimpleDateFormat("yyyyMMddHHmmss", Locale.US)
    private val hm = SimpleDateFormat("HH:mm", Locale.getDefault())

    // ------------------------------------------------------------------ entry points
    fun runAsync(ctx: Context, why: String, done: (() -> Unit)? = null) {
        val app = ctx.applicationContext
        thread(name = "glucose-alarms", isDaemon = true) { try { runNow(app, why) } finally { done?.invoke() } }
    }

    /** Add (or replace) an item in the on-phone log. */
    fun log(ctx: Context, type: String, text: String, units: Double?, atMs: Long): Event {
        val e = Event("phone_${type}_${idFmt.format(Date(atMs))}", atMs, type, text.trim(), units)
        AlarmStore.edit(ctx) { s ->
            s.events.removeAll { it.id == e.id }; s.events.add(e)
            if (type == "stick") s.prompts.filter { it.doneMs == null && it.dueMs <= atMs + 10 * MIN }.forEach { it.doneMs = atMs }
        }
        return e
    }

    fun delete(ctx: Context, e: Event) {
        AlarmStore.edit(ctx) { s -> s.events.removeAll { it.id == e.id } }
    }

    /** NIGHTLOW-1: "Going to sleep" button. Logs the bedtime reading, then predicts the night low + wake-up value (research only). */
    fun runNight(ctx: Context, gBed: Double): AlarmRec {
        val now = System.currentTimeMillis()
        log(ctx, "stick", "", gBed, now)
        val w = WatchReader.at(ctx, now)
        val clk = SimpleDateFormat("h:mm a", Locale.getDefault())
        val rec = AlarmStore.edit(ctx) { s ->
            val i = NightLow.inputs(s.events, now, gBed, w)
            val r = NightLow.predict(i)
            val (l1, l2) = NightLow.text(now, r) { clk.format(Date(it)) }
            val inputs = "g_bed=${gBed.toInt()};iob=${"%.2f".format(Locale.US, i.iob)};carbs4h=${i.carbs4h.toInt()};fpu=${"%.1f".format(Locale.US, i.fpu)};" +
                "milk_ml=${i.milkMl.toInt()};hr15=${i.hr15?.toInt() ?: ""};steps3h=${i.steps3h.toInt()};wake=${r.wake.toInt()}"
            val a = AlarmRec("night_${idFmt.format(Date(now))}", "night", now, now + 9 * 3600_000L, r.alarm, r.pMax, r.wake, 0, 0, "$l1\n$l2", inputs)
            s.alarms.add(a); a
        }
        runNow(ctx, "going to sleep")
        return rec
    }

    fun dismissPrompt(ctx: Context, id: String) = AlarmStore.edit(ctx) { s -> s.prompts.find { it.id == id }?.doneMs = -1L }

    // ------------------------------------------------------------------ main run
    fun runNow(ctx: Context, why: String) {
        if (!busy.compareAndSet(false, true)) return
        try {
            val now = System.currentTimeMillis()
            val w = WatchReader.at(ctx, now)
            val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val toNotify = ArrayList<Triple<Int, String, String>>()
            AlarmStore.edit(ctx) { s ->
                s.events.removeAll { it.t < now - 3 * 24 * 3600_000L }            // keep 3 days for the models
                val ev = s.events.toList()
                val shown = ModelCatalog.shown(ctx)   // models the user picked to see; the others still run and are logged, but stay quiet
                runLow(ctx, s, ev, now, w, prefs.getBoolean("lowNotify", true) && "low1" in shown, prefs.getBoolean("lowGate", true), toNotify)
                runSpike(s, ev, now, w, prefs.getBoolean("spikeNotify", true) && "spike1" in shown, toNotify, promptsOn = "spike1" in shown)
                runTiming(s, ev, now, w, prefs.getBoolean("spikeNotify", true), shown, toNotify)
                // prompts: notify when due, drop old ones
                s.prompts.filter { it.doneMs == null && !it.notified && it.dueMs <= now + MIN && it.dueMs > now - 30 * MIN }.forEach {
                    it.notified = true; toNotify.add(Triple(7000 + (it.dueMs / MIN % 1000).toInt(), "Check your blood glucose now", it.text))
                }
                s.prompts.removeAll { it.dueMs < now - 24 * 3600_000L }
                scoreOutcomes(s, ev, now)
                s.alarms.removeAll { it.madeMs < now - 60L * 24 * 3600_000L }   // keep 60 days
                s.lastRunMs = now; s.lastRunNote = "Checked at ${hm.format(Date(now))} ($why)" +
                    (if (WatchReader.status.isNotEmpty() && !WatchReader.status.startsWith("Filled")) ". " + WatchReader.status else if (w.hr15 == null) ". No heart rate in the last 15 min." else ". Watch data OK. " + WatchReader.status.trim())
            }
            toNotify.forEach { (id, title, text) -> notify(ctx, id, title, text) }
            scheduleNext(ctx)
        } catch (e: Exception) {
            Log.w(TAG, "alarm run failed", e)
        } finally { busy.set(false) }
    }

    private fun pct(p: Double) = "${(p * 100).toInt()}%"

    private fun runLow(ctx: Context, s: AlarmStore.State, ev: List<Event>, now: Long, w: Watch, notifyOn: Boolean, gateOn: Boolean, out: MutableList<Triple<Int, String, String>>) {
        val rs = listOf(60, LOW_HORIZON, 180).map { LowModel.predict(ev, now, it, w) }
        val r = rs[1]
        s.lastLow = JSONObject().put("t", now).put("p60", rs[0].pNoGlucose).put("p120", r.pNoGlucose).put("p180", rs[2].pNoGlucose)
            .put("p120_stick", r.pOneStick ?: JSONObject.NULL).put("iob", (r.inputs["ins0"] ?: 0.0) + (r.inputs["iob0"] ?: 0.0)).put("stick", r.stickValue ?: JSONObject.NULL).put("stick_age", r.stickAgeMin ?: JSONObject.NULL)
        if (now - s.lastCheckRowMs >= 14 * MIN) { AlarmStore.appendLowCheck(ctx, now, rs, w); s.lastCheckRowMs = now }
        if (r.pNoGlucose <= ModelParams.LOW_THRESHOLD) return
        // Gate: with nothing logged the model still says ~60-70% (it learned "no fat/protein = low"). On the Nov 6-12 readings,
        // asking for >= 0.5 u NovoRapid on board (incl. planned) kept 9/10 lows and cut false alarms 42% -> 29%.
        val insOnBoard = (r.inputs["ins0"] ?: 0.0) + (r.inputs["iob0"] ?: 0.0)
        if (gateOn && insOnBoard < LOW_MIN_IOB) return
        val recent = s.alarms.any { it.kind == "low" && it.alarm && now - it.madeMs < LOW_REPEAT_MIN * MIN }
        if (recent) return
        val inputs = r.inputs.entries.joinToString(";") { "${it.key}=${"%.2f".format(Locale.US, it.value)}" }
        val stickTxt = r.pOneStick?.let { " With your ${r.stickValue?.toInt()} reading from ${r.stickAgeMin?.toInt()} min ago: ${pct(it)}." } ?: ""
        s.alarms.add(AlarmRec("low_${idFmt.format(Date(now))}", "low", now, now + LOW_HORIZON * MIN, true, r.pNoGlucose, r.pOneStick, 0, 0,
            "Low (<70) risk by ${hm.format(Date(now + LOW_HORIZON * MIN))}", inputs, notified = "low"))
        s.prompts.add(Prompt("p_low_$now", now, now, "Low (<70) risk in about 2 h: ${pct(r.pNoGlucose)}.$stickTxt", notified = true))
        if (notifyOn) out.add(Triple(7101, "Low risk in ~2 h: ${pct(r.pNoGlucose)}",
            "Check your blood glucose now.$stickTxt Research only, not for dosing."))
    }

    private fun runSpike(s: AlarmStore.State, ev: List<Event>, now: Long, w: Watch, notifyOn: Boolean, out: MutableList<Triple<Int, String, String>>,
                         promptsOn: Boolean = true) {
        for (m in SpikeModel.meals(ev)) {
            if (m.startMs < now - 75 * MIN || m.startMs > now + 30 * MIN) continue      // only meals happening now
            val pre = SpikeModel.preStick(ev, m)?.units
            val s45 = SpikeModel.stick45(ev, m)?.units
            val r = SpikeModel.alarm(m.avail, m.dose, m.iob, w.hr15, w.steps60, pre, if (pre != null && s45 != null) s45 - pre else null)
            val id = "spike_${idFmt.format(Date(m.startMs))}"
            val old = s.alarms.find { it.id == id }
            if (old != null && old.step > r.step) continue                                 // keep the best-informed result
            val inputs = "avail=${"%.1f".format(Locale.US, m.avail)};dose=${m.dose};iob=${"%.2f".format(Locale.US, m.iob)};hr15=${w.hr15?.toInt() ?: ""};" +
                "steps60=${w.steps60.toInt()};pre=${pre ?: ""};d45=${if (pre != null && s45 != null) s45 - pre else ""}"
            val rec = AlarmRec(id, "spike", old?.madeMs ?: now, m.startMs, r.band >= 1, r.p30, r.p60, r.band, r.step, m.text, inputs,
                notified = old?.notified ?: "")
            s.alarms.removeAll { it.id == id }; s.alarms.add(rec)
            // 1) no pre-meal reading yet and the meal is just starting
            if (promptsOn && r.step == 0 && now <= m.startMs + SpikeModel.PRE_AFTER_MIN * MIN && "pre" !in rec.notified) {
                rec.notified += "pre,"
                s.prompts.add(Prompt("p_pre_${m.startMs}", now, now, "Take a reading now (counts as pre-meal up to 20 min after you start eating). Meal: ${m.text.take(60)}", notified = true))
                if (notifyOn) out.add(Triple(7201, "Check your blood glucose now", "For the spike alarm (counts as pre-meal up to 20 min after you start eating). Research only, not for dosing."))
            }
            // 2) borderline after the pre-meal reading: ask again 45 min after the meal started
            if (promptsOn && r.step == 1 && r.ask != null && "45" !in rec.notified) {
                rec.notified += "45,"
                s.prompts.add(Prompt("p_45_${m.startMs}", now, m.startMs + 45 * MIN, "45 min after your meal (borderline spike forecast). Meal: ${m.text.take(60)}"))
            }
            // 3) the warning itself (once per band and step)
            val key = "b${r.band}s${r.step},"
            if (r.band >= 1 && key !in rec.notified) {
                rec.notified += key
                if (notifyOn) out.add(Triple(7202, "Spike likely: ${ModelParams.SPIKE_BANDS[r.band]}",
                    "Rise ≥30: ${pct(r.p30)}, ≥60: ${pct(r.p60)}. ${m.text.take(40)}. Research only, not for dosing."))
            }
        }
    }

    /** SPIKE-2 and RISE40: one answer per meal at meal time, no reading after the meal needed. Updated while the meal is current. */
    private fun runTiming(s: AlarmStore.State, ev: List<Event>, now: Long, w: Watch, notifyOn: Boolean, shown: Set<String>,
                          out: MutableList<Triple<Int, String, String>>) {
        for (m in SpikeModel.meals(ev)) {
            if (m.startMs < now - 75 * MIN || m.startMs > now + 30 * MIN) continue
            val pre = SpikeModel.preStick(ev, m)?.units
            val x = TimingModels.inputsFor(ev, m, w, pre)
            val inputs = "meal=${m.text.take(80)};sugar=${"%.1f".format(Locale.US, x.sugar)};starch=${"%.1f".format(Locale.US, x.starch)};" +
                "fiber=${"%.1f".format(Locale.US, x.fiber)};fat=${"%.1f".format(Locale.US, x.fat)};protein=${"%.1f".format(Locale.US, x.protein)};" +
                "dose=${m.dose};iob=${"%.2f".format(Locale.US, m.iob)};hr15=${w.hr15?.toInt() ?: ""};steps60=${w.steps60.toInt()};pre=${pre ?: ""}"
            // SPIKE-2
            val r2 = TimingModels.spike2(x); val id2 = "spike2_${idFmt.format(Date(m.startMs))}"
            val old2 = s.alarms.find { it.id == id2 }
            val rec2 = AlarmRec(id2, "spike2", old2?.madeMs ?: now, m.startMs, r2.band >= 1, r2.p30, r2.p60, r2.band, if (r2.usedReading) 1 else 0,
                TimingModels.spike2Text(m.startMs, r2), inputs, notified = old2?.notified ?: "")
            s.alarms.removeAll { it.id == id2 }; s.alarms.add(rec2)
            val k2 = "b${r2.band}r${rec2.step},"
            if (r2.band >= 1 && k2 !in rec2.notified) {
                rec2.notified += k2
                if (notifyOn && "spike2" in shown) out.add(Triple(7301, "SPIKE-2: ${ModelParams.SPIKE_BANDS[r2.band]}", rec2.text + " Research only, not for dosing."))
            }
            // RISE40
            val r4 = TimingModels.rise40(x); val id4 = "rise40_${idFmt.format(Date(m.startMs))}"
            val old4 = s.alarms.find { it.id == id4 }
            val rec4 = AlarmRec(id4, "rise40", old4?.madeMs ?: now, m.startMs, r4.alert, r4.p40, null, if (r4.alert) 1 else 0, if (r4.usedReading) 1 else 0,
                TimingModels.rise40Text(m.startMs, r4), inputs, notified = old4?.notified ?: "")
            s.alarms.removeAll { it.id == id4 }; s.alarms.add(rec4)
            val k4 = "a${rec4.step},"
            if (r4.alert && k4 !in rec4.notified) {
                rec4.notified += k4
                if (notifyOn && "rise40" in shown) out.add(Triple(7401, "Rise of 40+ coming", rec4.text + " Research only, not for dosing."))
            }
        }
    }

    /** Fills outcomes from fingersticks so each alarm can be scored (and lows with no alarm before are counted). */
    private fun scoreOutcomes(s: AlarmStore.State, ev: List<Event>, now: Long) {
        val sticks = ev.filter { it.type == "stick" && it.units != null }
        val treats = ev.filter { it.type == "low_treat" }
        // NIGHTLOW-1: any reading under 70 during the 9 h night = low; else lowest reading; wake value = last reading in hours 7-10
        s.alarms.filter { it.kind == "night" && it.outcome == null && now > it.targetMs + 60 * MIN }.forEach { a ->
            val win = sticks.filter { it.t > a.madeMs + 15 * MIN && it.t <= a.targetMs + 60 * MIN }
            a.actual = win.filter { it.t >= a.madeMs + 7 * 3600_000L }.maxByOrNull { it.t }?.units
            a.outcome = when {
                win.any { it.units!! < 70 } || treats.any { it.t > a.madeMs && it.t <= a.targetMs } -> if (a.alarm) "low (alarm right)" else "low (missed)"
                win.isEmpty() -> "no reading"
                a.alarm -> "no low (false alarm?)"
                else -> "no low (right)"
            }
        }
        s.alarms.filter { it.kind == "low" && it.outcome == null && now > it.targetMs + 30 * MIN }.forEach { a ->
            val win = sticks.filter { it.t >= a.madeMs && it.t <= a.targetMs + 30 * MIN }
            a.actual = win.minOfOrNull { it.units!! }
            a.outcome = when {
                win.any { it.units!! < 70 } -> "low (alarm right)"
                win.isNotEmpty() -> "no low (false alarm?)"
                treats.any { it.t >= a.madeMs && it.t <= a.targetMs + 30 * MIN } -> "treated, no reading"
                else -> "no reading"
            }
        }
        // every real low: was there an alarm 30 min - 4 h before?
        sticks.filter { it.units!! < 70 && it.t > now - 3 * 24 * 3600_000L }.forEach { st ->
            val id = "lowseen_${st.id}"
            if (s.alarms.any { it.id == id }) return@forEach
            val warned = s.alarms.any { it.kind == "low" && it.alarm && it.madeMs in (st.t - 240 * MIN)..(st.t - 30 * MIN) }
            s.alarms.add(AlarmRec(id, "low_seen", st.t, st.t, false, 0.0, null, 0, 0, "Real low ${st.units!!.toInt()} mg/dL", "",
                outcome = if (warned) "warned before" else "missed (no alarm 30 min-4 h before)", actual = st.units))
        }
        val meals = SpikeModel.meals(ev)
        s.alarms.filter { it.kind == "spike" && it.outcome == null }.forEach { a ->
            val m = meals.find { it.startMs == a.targetMs } ?: return@forEach
            if (now < m.windowEndMs + 15 * MIN) return@forEach
            val rise = SpikeModel.actualRise(ev, m)
            a.actual = rise
            a.outcome = if (rise == null) "no readings to score" else {
                val real = (if (rise >= 30) 1 else 0) + (if (rise >= 60) 1 else 0)
                if (real == a.band) "band right" else "band wrong (real: ${ModelParams.SPIKE_BANDS[real]})"
            }
        }
        // SPIKE-2 (band) and RISE40 (yes/no 40+), scored from the sticks you happened to take (timing needs a sensor to score)
        s.alarms.filter { (it.kind == "spike2" || it.kind == "rise40") && it.outcome == null }.forEach { a ->
            val m = meals.find { it.startMs == a.targetMs } ?: return@forEach
            if (now < m.windowEndMs + 15 * MIN) return@forEach
            val rise = SpikeModel.actualRise(ev, m)
            a.actual = rise
            a.outcome = when {
                rise == null -> "no readings to score"
                a.kind == "rise40" -> if ((rise >= 40) == a.alarm) "right" else if (a.alarm) "false alarm" else "missed rise"
                else -> { val real = (if (rise >= 30) 1 else 0) + (if (rise >= 60) 1 else 0)
                    if (real == a.band) "band right" else "band wrong (real: ${ModelParams.SPIKE_BANDS[real]})" }
            }
        }
    }

    // ------------------------------------------------------------------ notifications + wake-ups
    private fun notify(ctx: Context, id: Int, title: String, text: String) {
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Low / spike research alarms", NotificationManager.IMPORTANCE_HIGH))
        val open = PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val n = NotificationCompat.Builder(ctx, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentTitle(title).setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(open).setAutoCancel(true).build()
        try { nm.notify(id, n) } catch (e: SecurityException) { Log.w(TAG, "no notification permission") }
    }

    fun scheduleNext(ctx: Context) {
        val now = System.currentTimeMillis()
        val due = AlarmStore.load(ctx).prompts.filter { it.doneMs == null && !it.notified && it.dueMs > now }.minOfOrNull { it.dueMs }
        val at = minOf(now + 15 * MIN, due ?: Long.MAX_VALUE)
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val pi = PendingIntent.getBroadcast(ctx, 4711, Intent(ctx, AlarmReceiver::class.java).setAction(ACTION_TICK),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        try {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S || am.canScheduleExactAlarms()) am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, pi)
            else am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, pi)
        } catch (e: SecurityException) { am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, pi) }
    }
}

/**
 * Watch data, READ-ONLY, from the Noise HR Logger CSV files in the phone's Documents/NoiseHR folder
 * (the Noise app saves every session there as well as in its own folder). Needs "All files access" for this app.
 * Reads every past session file and the live one as it is written (every 30 s). Never writes to that folder.
 * Short gaps (up to 10 min) are filled with a straight line; longer gaps are left empty (see WatchCsv).
 */
object WatchReader {
    @Volatile var status: String = ""
    @Volatile var filesFound: Int = 0
    @Volatile var firstFileDay: String = ""
    fun folder(): java.io.File = java.io.File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS), "NoiseHR")
    fun hasAccess(): Boolean = Build.VERSION.SDK_INT < Build.VERSION_CODES.R || android.os.Environment.isExternalStorageManager()

    /** All rows from every session file (past + live), oldest first. */
    fun allRows(): List<WatchCsv.Row> {
        val files = folder().listFiles { f -> f.isFile && f.name.startsWith("NoiseHR_Session") && f.name.endsWith(".csv") }?.toList() ?: emptyList()
        return files.flatMap { f -> try { f.useLines { WatchCsv.parse(it) } } catch (e: Exception) { emptyList() } }.sortedBy { it.t }
    }

    fun at(ctx: Context, now: Long): Watch {
        return try {
            if (!hasAccess()) { status = "Allow \"All files access\" (Models tab) so the app can read the Noise watch files"; return Watch(null, 0.0, 0.0) }
            val dir = folder()
            val files = dir.listFiles { f -> f.isFile && f.name.startsWith("NoiseHR_Session") && f.name.endsWith(".csv") }?.toList() ?: emptyList()
            filesFound = files.size
            if (files.isEmpty()) { status = "No Noise files in Documents/NoiseHR yet (open the updated Noise app once)"; return Watch(null, 0.0, 0.0) }
            firstFileDay = java.text.SimpleDateFormat("d MMM", Locale.getDefault()).format(Date(files.minOf { it.lastModified() }))
            // live calculation only needs files touched in the last 4 h
            val recent = files.filter { it.lastModified() > now - 4 * 3600_000L }
            val rows = recent.flatMap { f -> try { f.useLines { WatchCsv.parse(it) } } catch (e: Exception) { emptyList() } }
            val sm = WatchCsv.summarize(rows, now)
            status = when {
                sm.lastRowMs == null || now - sm.lastRowMs > 15 * 60_000L -> "No watch rows in the last 15 min (is the Noise app recording?)"
                else -> "" + (if (sm.filledSlots > 0) "Filled ${sm.filledSlots / 2} min of short gaps. " else "")
            }
            sm.watch
        } catch (e: SecurityException) {
            status = "No permission to read Documents/NoiseHR. Allow \"All files access\" (Models tab)."; Watch(null, 0.0, 0.0)
        } catch (e: Exception) { Log.w("WatchReader", "watch read failed", e); status = "Watch read failed: ${e.message}"; Watch(null, 0.0, 0.0) }
    }
}

/** Wakes the alarms every ~15 min and at "check at +45 min" times; re-arms after reboot. */
class AlarmReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val pending = goAsync()
        AlarmEngine.runAsync(context, if (intent.action == AlarmEngine.ACTION_TICK) "15-min check" else "after reboot") {
            try { AlarmEngine.scheduleNext(context.applicationContext) } catch (_: Exception) {}
            pending.finish()
        }
    }
}
