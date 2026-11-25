package com.example.glucosealarms

import android.content.Context
import android.os.Environment
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** One alarm / prediction record, kept so every alarm can be scored later. */
data class AlarmRec(
    val id: String, val kind: String,          // low | spike | low_seen
    val madeMs: Long, val targetMs: Long,      // low: made + 2 h; spike: meal start
    val alarm: Boolean,                         // true = user was warned
    val p: Double, val p2: Double?,             // low: p no-glucose, p one-stick; spike: P(rise>=30), P(rise>=60)
    val band: Int, val step: Int,               // spike only
    val text: String,                           // meal text or short reason
    val inputs: String,                         // model inputs, key=value;...
    var outcome: String? = null,                // filled later (low / no low / no reading / band right ...)
    var actual: Double? = null,                 // lowest stick (low) or real rise (spike)
    var notified: String = ""                   // which notifications were already sent
)

/** A "check your blood glucose now" request. Done once a fingerstick is logged after it was made. */
data class Prompt(val id: String, val madeMs: Long, val dueMs: Long, val text: String, var notified: Boolean = false, var doneMs: Long? = null)

/** App-private JSON state + CSV copies in Documents/GlucoseAlarms (same folder the session CSVs live in). */
object AlarmStore {
    private const val FILE = "glucose_alarms_state.json"
    val lock = Any()

    class State(
        val events: MutableList<Event> = mutableListOf(),
        val alarms: MutableList<AlarmRec> = mutableListOf(),
        val prompts: MutableList<Prompt> = mutableListOf(),
        var lastLow: JSONObject? = null,
        var lastCheckRowMs: Long = 0L,
        var lastRemotePullMs: Long = 0L,
        var lastRunMs: Long = 0L,
        var lastRunNote: String = ""
    )

    fun load(ctx: Context): State = synchronized(lock) {
        val f = File(ctx.filesDir, FILE)
        if (!f.exists()) return State()
        return try {
            val o = JSONObject(f.readText())
            State(
                events = o.optJSONArray("events").objs().map { Event(it.getString("id"), it.getLong("t"), it.getString("type"), it.optString("text", ""),
                    if (it.isNull("units")) null else it.optDouble("units")) }.toMutableList(),
                alarms = o.optJSONArray("alarms").objs().map { a -> AlarmRec(a.getString("id"), a.getString("kind"), a.getLong("madeMs"), a.getLong("targetMs"),
                    a.getBoolean("alarm"), a.getDouble("p"), if (a.isNull("p2")) null else a.optDouble("p2"), a.optInt("band"), a.optInt("step"),
                    a.optString("text", ""), a.optString("inputs", ""), if (a.isNull("outcome")) null else a.optString("outcome"),
                    if (a.isNull("actual")) null else a.optDouble("actual"), a.optString("notified", "")) }.toMutableList(),
                prompts = o.optJSONArray("prompts").objs().map { p -> Prompt(p.getString("id"), p.getLong("madeMs"), p.getLong("dueMs"), p.getString("text"),
                    p.optBoolean("notified"), if (p.isNull("doneMs")) null else p.optLong("doneMs")) }.toMutableList(),
                lastLow = o.optJSONObject("lastLow"),
                lastCheckRowMs = o.optLong("lastCheckRowMs"), lastRemotePullMs = o.optLong("lastRemotePullMs"),
                lastRunMs = o.optLong("lastRunMs"), lastRunNote = o.optString("lastRunNote", "")
            )
        } catch (e: Exception) { State() }
    }

    fun save(ctx: Context, s: State) = synchronized(lock) {
        val o = JSONObject()
        o.put("events", JSONArray(s.events.map { JSONObject().put("id", it.id).put("t", it.t).put("type", it.type).put("text", it.text).put("units", it.units ?: JSONObject.NULL) }))
        o.put("alarms", JSONArray(s.alarms.map { a -> JSONObject().put("id", a.id).put("kind", a.kind).put("madeMs", a.madeMs).put("targetMs", a.targetMs)
            .put("alarm", a.alarm).put("p", a.p).put("p2", a.p2 ?: JSONObject.NULL).put("band", a.band).put("step", a.step).put("text", a.text)
            .put("inputs", a.inputs).put("outcome", a.outcome ?: JSONObject.NULL).put("actual", a.actual ?: JSONObject.NULL).put("notified", a.notified) }))
        o.put("prompts", JSONArray(s.prompts.map { p -> JSONObject().put("id", p.id).put("madeMs", p.madeMs).put("dueMs", p.dueMs).put("text", p.text)
            .put("notified", p.notified).put("doneMs", p.doneMs ?: JSONObject.NULL) }))
        o.put("lastLow", s.lastLow ?: JSONObject.NULL).put("lastCheckRowMs", s.lastCheckRowMs).put("lastRemotePullMs", s.lastRemotePullMs)
            .put("lastRunMs", s.lastRunMs).put("lastRunNote", s.lastRunNote)
        val tmp = File(ctx.filesDir, "$FILE.tmp"); tmp.writeText(o.toString()); tmp.renameTo(File(ctx.filesDir, FILE))
        try { writeCsvs(ctx, s) } catch (_: Exception) {}
    }

    /** Small edit helper: load, change, save, all under one lock. */
    fun <T> edit(ctx: Context, f: (State) -> T): T = synchronized(lock) { val s = load(ctx); val r = f(s); save(ctx, s); r }

    private fun JSONArray?.objs(): List<JSONObject> = if (this == null) emptyList() else (0 until length()).map { getJSONObject(it) }

    // ---------------------------------------------------------------- CSV copies (for scoring on the laptop)
    private val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    fun t(ms: Long?): String = if (ms == null) "" else ts.format(Date(ms))
    fun dir(ctx: Context): File? = ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)?.let { File(it, "GlucoseAlarms").apply { mkdirs() } }
    private fun q(s: String) = "\"" + s.replace("\"", "'") + "\""

    private fun writeCsvs(ctx: Context, s: State) {
        val d = dir(ctx) ?: return
        File(d, "alarms.csv").writeText(buildString {
            append("id,kind,made,target,alarm,p,p2,band,step,text,inputs,outcome,actual\n")
            s.alarms.sortedBy { it.madeMs }.forEach { a ->
                append("${a.id},${a.kind},${t(a.madeMs)},${t(a.targetMs)},${if (a.alarm) 1 else 0},${"%.3f".format(Locale.US, a.p)},${a.p2?.let { "%.3f".format(Locale.US, it) } ?: ""}," +
                    "${a.band},${a.step},${q(a.text)},${q(a.inputs)},${a.outcome ?: ""},${a.actual ?: ""}\n")
            }
        })
        File(d, "events.csv").writeText(buildString {
            append("time,type,text,units,id\n")
            s.events.sortedBy { it.t }.forEach { e -> append("${t(e.t)},${e.type},${q(e.text)},${e.units ?: ""},${e.id}\n") }
        })
    }

    /** One row every ~15 min with the low model's numbers, so missed lows can be found later. */
    fun appendLowCheck(ctx: Context, nowMs: Long, rs: List<LowResult>, w: Watch) {
        val d = dir(ctx) ?: return
        val f = File(d, "low_checks.csv")
        if (!f.exists()) f.writeText("time,horizon_min,p_no_glucose,p_one_stick,last_stick,stick_age_min,hr15,steps60,cal60,ins0,carb0,fpu0,iob0\n")
        f.appendText(rs.joinToString("") { r ->
            "${t(nowMs)},${r.horizonMin},${"%.3f".format(Locale.US, r.pNoGlucose)},${r.pOneStick?.let { "%.3f".format(Locale.US, it) } ?: ""},${r.stickValue ?: ""}," +
                "${r.stickAgeMin?.let { "%.0f".format(Locale.US, it) } ?: ""},${w.hr15?.let { "%.0f".format(Locale.US, it) } ?: ""},${w.steps60.toInt()},${w.cal60.toInt()}," +
                listOf("ins0", "carb0", "fpu0", "iob0").joinToString(",") { k -> "%.2f".format(Locale.US, r.inputs[k] ?: 0.0) } + "\n"
        })
    }
}
