package com.example.glucosealarms

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/*
 * Pure Kotlin (no Android) port of the two research alarms, so they run on the phone with no internet.
 *   LOW-1  : chance a glucometer reading is under 70 at time X (1-3 h ahead), planned food + insulin known up to X.
 *   SPIKE-1: after a meal, rise band (<30 / 30-60 / >=60) from uncovered carbs, insulin on board, HR, steps, sticks.
 * Same maths as the Python training code (glucoforecast8/9, parse_meal, lowfeat, spike_alarm). Checked against the
 * Python outputs in app/src/test (AlarmCoreTest). RESEARCH ONLY: never use for dosing.
 */

/** One logged item. type: food, rapid, correction, lantus, low_treat, stick, note. t = epoch ms (may be in the future = planned). */
data class Event(val id: String, val t: Long, val type: String, val text: String = "", val units: Double? = null)

/** Watch summary at a moment: HR mean over the last 15 min, raw step and calorie totals over the last 60 min. */
data class Watch(val hr15: Double?, val steps60: Double, val cal60: Double, val steps180: Double = 0.0)

data class Macros(val carbs: Double, val protein: Double, val fat: Double, val fiber: Double, val sugar: Double,
                  val starch: Double, val gl: Double, val freeSugar: Double) {
    val avail get() = max(carbs - fiber, 0.0)
    companion object { val ZERO = Macros(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }
    operator fun plus(o: Macros) = Macros(carbs + o.carbs, protein + o.protein, fat + o.fat, fiber + o.fiber, sugar + o.sugar,
        starch + o.starch, gl + o.gl, freeSugar + o.freeSugar)
}

/** Port of parse_meal.py. Items split by ';' or new line. Unknown foods count as 0 (shown to the user). */
object MealParser {
    private val DAIRY = setOf("paneer_from_milk", "milk_buffalo", "milk_toned", "milk_cow", "milk", "paneer", "curd")
    private const val TBSP_G = 15.0
    /** Same as Python round(x, 1): exact binary value, halves to even. */
    private fun r1(x: Double) = java.math.BigDecimal(x).setScale(1, java.math.RoundingMode.HALF_EVEN).toDouble()
    private fun num(s: String) = Regex("(\\d+(?:\\.\\d+)?)").find(s)?.groupValues?.get(1)?.toDouble()

    data class Item(val text: String, val food: String, val grams: Double?, val m: Macros)

    fun parseItem(text: String): Item? {
        val t = text.lowercase().trim()
        if (t.isEmpty() || t == "coffee") return null
        Regex("(\\d+(?:\\.\\d+)?)\\s*gm?\\s*carb").find(t)?.let {
            val c = it.groupValues[1].toDouble()
            return Item(t, "explicit_carbs", null, Macros(c, 0.0, 0.0, 0.0, c / 2, c / 2, r1(0.55 * c), c / 2))
        }
        val row = FoodDb.ROWS.firstOrNull { it.pattern.containsMatchIn(t) } ?: return Item(t, "UNKNOWN", null, Macros.ZERO)
        val n = num(t)
        val grams = when {
            row.unit == "ml_milk" -> (n ?: row.defaultQty) * row.gramsPerUnit
            Regex("\\bml\\b").containsMatchIn(t) -> n ?: 0.0
            Regex("\\d\\s*(gm|g|gms|grams)\\b").containsMatchIn(t) -> n ?: 0.0
            "cup" in t -> (n ?: 1.0) * (if (row.key == "curd") 240.0 else if (row.unit == "cup") row.gramsPerUnit else 150.0)
            Regex("tbsp|table\\s*spoon").containsMatchIn(t) -> (n ?: 1.0) * (if (row.unit == "tbsp") row.gramsPerUnit else TBSP_G)
            row.unit == "piece" -> (n ?: row.defaultQty) * row.gramsPerUnit
            else -> (if (n != null && n != 0.0 && row.unit != "g") n else row.defaultQty) * (if (row.unit !in setOf("g", "ml")) row.gramsPerUnit else 1.0)
        }
        val carbs = r1(grams * row.carbs / 100); val protein = r1(grams * row.protein / 100); val fat = r1(grams * row.fat / 100)
        val fiber = r1(grams * row.fiber / 100); var sugar = r1(grams * row.sugar / 100)
        val avail = max(carbs - fiber, 0.0)
        sugar = if (avail != 0.0) min(sugar, avail) else 0.0
        val starch = r1(max(avail - sugar, 0.0))
        val gl = r1(row.gi * avail / 100)
        val free = if (row.key in DAIRY) 0.0 else r1(sugar * min(row.gi / 55, 1.0))
        return Item(t, row.key, r1(grams), Macros(carbs, protein, fat, fiber, sugar, starch, gl, free))
    }

    fun parse(text: String): Pair<List<Item>, Macros> {
        if (text.isBlank()) return emptyList<Item>() to Macros.ZERO
        val items = text.split(Regex("[;\\n]")).mapNotNull { parseItem(it) }
        return items to items.fold(Macros.ZERO) { a, i -> a + i.m }
    }
}

/** Absorption curves (cumulative share absorbed m minutes after the event), same as glucoforecast.py. */
object Kernels {
    fun insulinCum(m: Double, peak: Double = 75.0, dur: Double = 300.0): Double {
        if (m < 0) return 0.0
        val tau = peak * (1 - peak / dur) / (1 - 2 * peak / dur)
        val a = 2 * tau / dur; val s = 1 / (1 - a + (1 + a) * exp(-dur / tau))
        val t = min(m, dur)
        val iob = 1 - s * (1 - a) * ((t * t / (tau * dur * (1 - a)) - t / tau - 1) * exp(-t / tau) + 1)
        return 1 - iob.coerceIn(0.0, 1.0)
    }
    fun triCum(m: Double, dur: Double, peakFrac: Double = 1.0 / 3): Double {
        if (m < 0) return 0.0
        val p = dur * peakFrac; val t = m.coerceIn(0.0, dur)
        return if (t < p) t * t / (p * dur) else 1 - (dur - t) * (dur - t) / ((dur - p) * dur)
    }
}

/** One event turned into the model channels (load_log4 + load_log9 fiber split). */
data class Channels(val t: Long, val rapid: Double, val freeSugar: Double, val gl: Double, val slow: Double, val fpu: Double, val avail: Double)

object Features {
    private const val FIB_CAP = 0.6
    fun isInsulin(e: Event) = e.type == "rapid" || e.type == "correction"
    fun isFood(e: Event) = (e.type == "food" || e.type == "low_treat") && e.text.isNotBlank()

    fun channels(events: List<Event>): List<Channels> = events.mapNotNull { e ->
        val rapid = if (isInsulin(e)) (e.units ?: 0.0) else 0.0
        if (!isFood(e) && rapid == 0.0) return@mapNotNull null
        val m = if (isFood(e)) MealParser.parse(e.text).second else Macros.ZERO
        val avail = m.avail
        var slow = max(avail - m.gl, 0.0)
        val s = if (avail + m.fiber > 0) min(m.fiber / (avail + m.fiber), FIB_CAP) else 0.0
        val mvS = m.freeSugar * s; val mvG = m.gl * s
        slow += mvS + mvG
        Channels(e.t, rapid, m.freeSugar - mvS, m.gl - mvG, slow, (9 * m.fat + 4 * m.protein) / 100, avail)
    }

    data class Window(val insNext: Double, val carbNext: Double, val fpuNext: Double, val insOb: Double)

    /** event_features4 at time [atMs] for horizon [h] minutes, with PLAN = h - 1 (things logged before the target are known). */
    fun window(ch: List<Channels>, atMs: Long, h: Int): Window {
        val plan = h - 1
        var insNext = 0.0; var carbNext = 0.0; var fpuNext = 0.0; var insOb = 0.0
        val atMin = Math.floorDiv(atMs, 60_000L)
        for (c in ch) {
            val m = (atMin - Math.floorDiv(c.t, 60_000L)).toDouble()
            if (m < -plan) continue
            if (c.rapid != 0.0) {
                val cc = Kernels.insulinCum(m)
                insOb += c.rapid * (1 - cc); insNext += c.rapid * (Kernels.insulinCum(m + h) - cc)
            }
            carbNext += c.freeSugar * (Kernels.triCum(m + h, 90.0) - Kernels.triCum(m, 90.0)) +
                c.gl * (Kernels.triCum(m + h, 150.0) - Kernels.triCum(m, 150.0)) +
                c.slow * (Kernels.triCum(m + h, 240.0, 0.4) - Kernels.triCum(m, 240.0, 0.4))
            fpuNext += c.fpu * (Kernels.triCum(m + h, 300.0, 0.5) - Kernels.triCum(m, 300.0, 0.5))
        }
        return Window(insNext, carbNext, fpuNext, insOb)
    }
}

/** StandardScaler + LogisticRegression with the exported numbers. */
class Logistic(val cols: Array<String>, val mean: DoubleArray, val scale: DoubleArray, val coef: DoubleArray, val intercept: Double) {
    fun prob(x: Map<String, Double>): Double {
        var z = intercept
        for (i in cols.indices) z += coef[i] * ((x.getValue(cols[i]) - mean[i]) / scale[i])
        return 1 / (1 + exp(-z))
    }
}

data class LowResult(val atMs: Long, val horizonMin: Int, val pNoGlucose: Double, val pOneStick: Double?,
                     val stickValue: Double?, val stickAgeMin: Double?, val inputs: Map<String, Double>)

object LowModel {
    private const val MAX_STICK_AGE_MIN = 480.0
    fun lastStick(events: List<Event>, atMs: Long) =
        events.filter { it.type == "stick" && it.units != null && it.t <= atMs }.maxByOrNull { it.t }

    /** P(reading < 70) [h] minutes after [atMs]. One-stick version when a fingerstick <= 8 h old exists. */
    fun predict(events: List<Event>, atMs: Long, h: Int, w: Watch): LowResult {
        val ch = Features.channels(events)
        val hr = w.hr15 ?: 70.0
        val f = Features.window(ch, atMs, h)
        val xn = mapOf("ins0" to f.insNext, "carb0" to f.carbNext, "fpu0" to f.fpuNext, "iob0" to f.insOb - f.insNext,
            "L" to h.toDouble(), "n_hr_15" to hr, "n_steps_60" to w.steps60, "n_cal_60" to w.cal60)
        val pn = ModelParams.LOW_NO_GLUCOSE.prob(xn)
        val s = lastStick(events, atMs)
        var po: Double? = null; var age: Double? = null
        if (s != null) {
            age = (atMs - s.t) / 60_000.0
            if (age <= MAX_STICK_AGE_MIN) {
                val hh = (age + h).toInt()
                val o = Features.window(ch, s.t, hh)
                po = ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to s.units!!, "ins" to o.insNext, "carb" to o.carbNext, "fpu" to o.fpuNext,
                    "H" to hh.toDouble(), "o_age" to age, "n_hr_15" to hr, "n_steps_60" to w.steps60, "n_cal_60" to w.cal60))
            }
        }
        return LowResult(atMs, h, pn, po, s?.units, age, xn)
    }
}

data class Meal(val startMs: Long, val endFoodMs: Long, val windowEndMs: Long, val text: String, val avail: Double,
                val dose: Double, val iob: Double)

data class SpikeResult(val band: Int, val p30: Double, val p60: Double, val step: Int, val ask: String?, val uncov: Double)

object SpikeModel {
    private const val MIN = 60_000L

    /** Groups food logs into meals like build.py: foods within 45 min of the first one are one meal. */
    fun meals(events: List<Event>): List<Meal> {
        val foods = events.filter { Features.isFood(it) }.sortedBy { it.t }
        val ins = events.filter { Features.isInsulin(it) && (it.units ?: 0.0) > 0 }
        val out = ArrayList<Meal>(); var i = 0
        while (i < foods.size) {
            val t0 = foods[i].t; val grp = ArrayList<Event>()
            while (i < foods.size && foods[i].t <= t0 + 45 * MIN) { grp.add(foods[i]); i++ }
            val end = grp.last().t
            val avail = grp.sumOf { MealParser.parse(it.text).second.avail }
            val dose = ins.filter { it.t >= t0 - 50 * MIN && it.t <= end + 30 * MIN }.sumOf { it.units!! }
            val iob = ins.filter { it.t < t0 - 50 * MIN }.sumOf { it.units!! * (1 - Kernels.insulinCum((t0 - it.t) / MIN.toDouble())) }
            out.add(Meal(t0, end, 0L, grp.joinToString("; ") { it.text }, avail, dose, iob))
        }
        return out.mapIndexed { k, m ->
            val next = out.getOrNull(k + 1)?.startMs
            m.copy(windowEndMs = if (next != null) min(next, m.startMs + 300 * MIN) else m.startMs + 300 * MIN)
        }
    }

    /** Minutes after the meal start a reading still counts as "pre-meal" (you started eating at 8:26 and tested at 8:41 = OK).
     *  Your rises start ~25-55 min after eating, so a reading up to 20 min in is still about the pre-meal level. */
    const val PRE_AFTER_MIN = 20L
    /** Pre-meal reading: from 45 min before the meal to 20 min after it started. Prefers the one closest to the meal start. */
    fun preStick(events: List<Event>, m: Meal) = events.filter { it.type == "stick" && it.units != null && it.t >= m.startMs - 45 * MIN && it.t <= m.startMs + PRE_AFTER_MIN * MIN }
        .minByOrNull { abs(it.t - m.startMs) }
    fun stick45(events: List<Event>, m: Meal) = events.filter { it.type == "stick" && it.units != null && it.t >= m.startMs + 35 * MIN && it.t <= m.startMs + 60 * MIN }
        .minByOrNull { abs(it.t - (m.startMs + 45 * MIN)) }

    /** Same as spike_alarm.alarm(). */
    fun alarm(availCarbs: Double, dose: Double, iob: Double, hr15: Double?, steps60: Double, preBg: Double?, d45: Double?): SpikeResult {
        val uncov = availCarbs - ModelParams.CARBS_PER_UNIT * dose
        val x = HashMap<String, Double>(mapOf("uncov" to uncov, "iob" to iob, "hr15" to (hr15 ?: ModelParams.SPIKE_HR15_FALLBACK), "steps60" to steps60))
        var step = 0
        if (preBg != null) {
            x["prec"] = preBg.coerceIn(40.0, 250.0); x["low"] = if (preBg < 70) 1.0 else 0.0; step = 1
            if (d45 != null) { x["d45"] = d45; step = 2 }
        }
        val (m30, m60) = when (step) {
            0 -> ModelParams.SPIKE_STEP0_30 to ModelParams.SPIKE_STEP0_60
            1 -> ModelParams.SPIKE_STEP1_30 to ModelParams.SPIKE_STEP1_60
            else -> ModelParams.SPIKE_STEP2_30 to ModelParams.SPIKE_STEP2_60
        }
        val p30 = m30.prob(x); val p60 = min(m60.prob(x), p30)
        val band = (if (p30 > .5) 1 else 0) + (if (p60 > .5) 1 else 0)
        val w = ModelParams.SPIKE_ASK_WITHIN
        val ask = when {
            step == 0 -> "Check your blood glucose now (before eating) for a better spike forecast."
            step == 1 && (abs(p30 - .5) < w || abs(p60 - .5) < w) -> "Borderline. Check your blood glucose 45 min after starting the meal."
            else -> null
        }
        return SpikeResult(band, p30, p60, step, ask, uncov)
    }

    /** Actual rise for scoring: highest stick from +15 min to window end (+10) minus the pre-meal stick. Null if not measurable. */
    fun actualRise(events: List<Event>, m: Meal): Double? {
        val pre = preStick(events, m)?.units ?: return null
        val preT = preStick(events, m)!!.t
        val post = events.filter { it.type == "stick" && it.units != null && it.t > preT && it.t >= m.startMs + 15 * MIN && it.t <= m.windowEndMs + 10 * MIN }
        if (post.isEmpty()) return null
        return post.maxOf { it.units!! } - pre
    }
}
