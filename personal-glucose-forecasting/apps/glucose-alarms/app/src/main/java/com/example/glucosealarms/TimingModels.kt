package com.example.glucosealarms

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.exp
import kotlin.math.min

/*
 * SPIKE-2 v2 and RISE40-ALERT (2026-11-15): answers at meal time with NO reading after the meal.
 *   SPIKE-2  : rise size band (<30 / 30-60 / >=60, same maths as SPIKE-1 first step) + when the rise starts (+30) and peaks, as RANGES.
 *   RISE40   : will glucose rise 40 or more above the pre-meal level (fast or slow)? + ranges for: starts rising (+20), passes +40, highest.
 * Ranges hold the real time 7 times out of 10 on days the model never saw. Same maths as export2.py (checked in AlarmCoreTest).
 * RESEARCH ONLY. Never use for dosing.
 */

/** StandardScaler + Ridge (linear) with exported numbers. The time models predict log(minutes). */
class Linear(val cols: Array<String>, val mean: DoubleArray, val scale: DoubleArray, val coef: DoubleArray, val intercept: Double) {
    fun value(x: Map<String, Double>): Double {
        var z = intercept
        for (i in cols.indices) z += coef[i] * ((x.getValue(cols[i]) - mean[i]) / scale[i])
        return z
    }
}

/** A time range in minutes after the meal start. */
data class MinRange(val mid: Double, val lo: Double, val hi: Double)

data class MealInputs(val sugar: Double, val starch: Double, val fiber: Double, val fat: Double, val protein: Double, val milk: Boolean,
                      val dose: Double, val iob: Double, val hr15: Double?, val steps60: Double, val hour: Int, val pre: Double?)

data class Spike2Result(val band: Int, val p30: Double, val p60: Double, val usedReading: Boolean, val start: MinRange, val peak: MinRange,
                        val lateRisk: Double)

data class Rise40Result(val p40: Double, val alert: Boolean, val usedReading: Boolean, val start: MinRange, val pass40: MinRange, val peak: MinRange)

object TimingModels {
    private const val MIN = 60_000L

    /** The model inputs, built the same way as ranges.py / export2.py. */
    fun features(m: MealInputs): Pair<HashMap<String, Double>, Boolean> {
        val avail = m.sugar + m.starch
        val x = hashMapOf("uncov" to avail - ModelParams.CARBS_PER_UNIT * m.dose, "iob" to m.iob,
            "hr15" to (m.hr15 ?: ModelParams.SPIKE_HR15_FALLBACK), "steps60" to m.steps60, "sug_share" to m.sugar / (avail + 1),
            "starch" to m.starch, "fiber" to m.fiber, "fpu" to (9 * m.fat + 4 * m.protein) / 100, "milk" to if (m.milk) 1.0 else 0.0,
            "dose" to m.dose, "night" to if (m.hour >= 21 || m.hour < 6) 1.0 else 0.0)
        val pre = m.pre
        if (pre != null) { x["prec"] = pre.coerceIn(40.0, 250.0); x["low"] = if (pre < 70) 1.0 else 0.0; x["pre"] = pre }
        return x to (pre != null)
    }

    private fun range(lin: Linear, f: DoubleArray, x: Map<String, Double>): MinRange {
        val mid = exp(lin.value(x)); return MinRange(mid, mid * f[0], mid * f[1])
    }

    fun spike2(m: MealInputs): Spike2Result {
        val (x, pre) = features(m)
        val (m30, m60) = if (pre) ModelParams.SPIKE_STEP1_30 to ModelParams.SPIKE_STEP1_60 else ModelParams.SPIKE_STEP0_30 to ModelParams.SPIKE_STEP0_60
        val p30 = m30.prob(x); val p60 = min(m60.prob(x), p30)
        val band = (if (p30 > .5) 1 else 0) + (if (p60 > .5) 1 else 0)
        val P = TimingParams
        val start = if (pre) range(P.S2_START_PRE, P.S2_START_PRE_RANGE, x) else range(P.S2_START_NONE, P.S2_START_NONE_RANGE, x)
        val peak = if (pre) range(P.S2_PEAK_PRE, P.S2_PEAK_PRE_RANGE, x) else range(P.S2_PEAK_NONE, P.S2_PEAK_NONE_RANGE, x)
        val late = (if (pre) P.LATE_PRE else P.LATE_NONE).prob(x)
        return Spike2Result(band, p30, p60, pre, start, peak, late)
    }

    fun rise40(m: MealInputs): Rise40Result {
        val (x, pre) = features(m)
        val P = TimingParams
        val p = (if (pre) P.P40_PRE else P.P40_NONE).prob(x)
        val start = if (pre) range(P.R_START_PRE, P.R_START_PRE_RANGE, x) else range(P.R_START_NONE, P.R_START_NONE_RANGE, x)
        val p40 = if (pre) range(P.R_40_PRE, P.R_40_PRE_RANGE, x) else range(P.R_40_NONE, P.R_40_NONE_RANGE, x)
        val peak = if (pre) range(P.R_PEAK_PRE, P.R_PEAK_PRE_RANGE, x) else range(P.R_PEAK_NONE, P.R_PEAK_NONE_RANGE, x)
        return Rise40Result(p, p > 0.5, pre, start, p40, peak)
    }

    /** Inputs for a meal from the log (food grouped like SpikeModel.meals). */
    fun inputsFor(events: List<Event>, meal: Meal, w: Watch, pre: Double?): MealInputs {
        val foods = events.filter { Features.isFood(it) && it.t >= meal.startMs && it.t <= meal.endFoodMs }
        val mac = foods.fold(Macros.ZERO) { a, e -> a + MealParser.parse(e.text).second }
        val milk = foods.any { it.text.lowercase().contains("milk") } && !foods.all { it.text.lowercase().contains("paneer from") }
        val hour = java.util.Calendar.getInstance().apply { timeInMillis = meal.startMs }.get(java.util.Calendar.HOUR_OF_DAY)
        return MealInputs(mac.sugar, mac.starch, mac.fiber, mac.fat, mac.protein, milk, meal.dose, meal.iob, w.hr15, w.steps60, hour, pre)
    }

    // ---------------------------------------------------------------- wording (rounded to 5 min)
    private val hm = SimpleDateFormat("h:mm a", Locale.getDefault())
    private fun at(t0: Long, mins: Double): String { val ms = t0 + (Math.round(mins / 5.0) * 5) * MIN; return hm.format(Date(ms)) }
    fun span(t0: Long, r: MinRange) = "${at(t0, r.lo)} – ${at(t0, r.hi)}"

    fun spike2Text(t0: Long, r: Spike2Result): String {
        val size = arrayOf("No rise of 30+ expected", "Rise of about +30 to 60 likely", "Big rise of +60 or more likely")[r.band]
        if (r.band == 0) return "$size (chance ${(r.p30 * 100).toInt()}%)." + if (r.lateRisk > .5) " It may still creep up late, after ${at(t0, 120.0)}." else ""
        return "$size. Starts rising ${span(t0, r.start)}. Highest ${span(t0, r.peak)}." + if (r.lateRisk > .5) " May start late (after 2 h)." else ""
    }

    fun rise40Text(t0: Long, r: Rise40Result): String {
        val pc = (r.p40 * 100).toInt()
        if (!r.alert) return "No rise of 40+ expected (chance $pc%)."
        return "Rise of 40+ likely ($pc%). Starts rising ${span(t0, r.start)}. Passes +40 ${span(t0, r.pass40)}. Highest ${span(t0, r.peak)}."
    }
}

