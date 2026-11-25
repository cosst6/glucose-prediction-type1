package com.example.glucosealarms

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong
import kotlin.math.sqrt

/**
 * NIGHTLOW-1 (15 Nov 2026): at bedtime, predicts glucose for each of the next 9 hours (blood scale),
 * the chance of a night low (<70) with a 1-3 h time range, and the wake-up value (mean of hours 8-9).
 * Ported from NIGHTLOW-1_glucometer-bedtime-food-macros-insulin-watch15-steps-PLANNED90M_v1_2026-11-15.json (phys.py / feats.py). Trained on only 18 nights (2 real lows): research only, never for dosing.
 */
object NightLow {
    const val ID = "NIGHTLOW-1_glucometer-bedtime-food-macros-insulin-watch15-steps-PLANNED90M_v1_2026-11-15"
    const val PLANNED_MIN = 90L          // food / NovoRapid planned up to 90 min after bedtime count
    const val CUT = 0.4
    const val HI = 0.7
    /** Features: one, gdev, iob, carbs, milk, fpu, hr, steps. Rows = hours 1..9 after bedtime. */
    val COEF = arrayOf(
        doubleArrayOf(24.0935, -0.0125, -10.4951, 0.1897, 52.5634, 0.3653, 1.1682, -3.3992),
        doubleArrayOf(-7.3605, -0.0, -18.337, 0.232, 74.3601, 2.8238, 1.6119, 0.3273),
        doubleArrayOf(-45.1163, -0.0127, -20.2549, 0.3118, 93.7838, 5.614, 2.1199, 2.9331),
        doubleArrayOf(-33.496, -0.0, -20.0205, 0.0, 105.3055, 3.2327, 2.7327, 3.0698),
        doubleArrayOf(-81.6082, -0.0, -21.5517, 0.0, 129.8403, 10.4603, 1.8575, 5.1874),
        doubleArrayOf(-93.4316, -0.0, -22.8098, 0.0, 113.2091, 12.0938, 2.0095, 7.2751),
        doubleArrayOf(-103.9212, -0.0, -23.8197, 0.0, 107.7527, 13.1379, 1.6171, 6.5964),
        doubleArrayOf(-107.2416, -0.0, -17.0903, 0.0, 90.0215, 11.4176, -2.0137, 4.723),
        doubleArrayOf(-30.5172, -0.3339, -35.5349, 0.0, 46.6597, 15.9215, -0.5085, -0.0508))
    val SD = doubleArrayOf(27.7, 29.8, 40.5, 47.8, 54.5, 54.3, 51.8, 51.6, 45.5)

    data class Inputs(val gBed: Double, val iob: Double, val carbs4h: Double, val fpu: Double, val milkMl: Double, val hr15: Double?, val steps3h: Double)
    data class Result(val pred: DoubleArray, val pLow: DoubleArray, val rangeH: Pair<Double, Double>?, val wake: Double, val wakeLo: Double, val wakeHi: Double) {
        val pMax get() = pLow.max()
        val alarm get() = pMax > CUT
    }

    private val MILK = Regex("(\\d+(?:\\.\\d+)?)\\s*ml[^;]*milk", RegexOption.IGNORE_CASE)
    fun milkMl(text: String) = MILK.findAll(text).sumOf { it.groupValues[1].toDouble() }

    /** Inputs from the log at bedtime [bedMs] (same windows as feats.py). NovoRapid = "rapid" and "correction" events. */
    fun inputs(events: List<Event>, bedMs: Long, gBed: Double, w: Watch): Inputs {
        var iob = 0.0; var carbs = 0.0; var fat = 0.0; var prot = 0.0; var milk = 0.0
        for (e in events) {
            val m = (bedMs - e.t) / 60_000.0                // minutes before bedtime (negative = after, planned)
            if (m < -PLANNED_MIN) continue
            if ((e.type == "rapid" || e.type == "correction") && e.units != null && m > -PLANNED_MIN && m < 300)
                iob += e.units * (1 - Kernels.insulinCum(max(m, 0.0)))
            if (e.type == "food" || e.type == "low_treat") {
                if (m <= 240) { val t = MealParser.parse(e.text).second; carbs += t.avail; fat += t.fat; prot += t.protein }
                if (m <= 60) milk += milkMl(e.text)
            }
        }
        return Inputs(gBed, iob, carbs, (9 * fat + 4 * prot) / 100, milk, w.hr15, w.steps180)
    }

    fun x(i: Inputs) = doubleArrayOf(1.0, i.gBed - 120, min(i.iob, 5.5), min(i.carbs4h, 80.0), min(i.milkMl, 500.0) / 250, min(i.fpu, 8.0),
        if (i.hr15 == null) 0.0 else (i.hr15 - 70).coerceIn(-15.0, 25.0) / 10, ln(1 + max(i.steps3h, 0.0)))

    fun predict(i: Inputs): Result {
        val x = x(i)
        val pred = DoubleArray(9) { h -> i.gBed + COEF[h].indices.sumOf { COEF[h][it] * x[it] } }
        val p = DoubleArray(9) { h -> normCdf((70 - pred[h]) / SD[h]) }
        val wake = (pred[7] + pred[8]) / 2; val wsd = (SD[7] + SD[8]) / 2
        return Result(pred, p, lowRange(p), wake, wake - 1.28 * wsd, wake + 1.28 * wsd)
    }

    /** Hours after bedtime: start = first hour P>0.4 minus 0.5 h, end = first hour P>0.7 (1-3 h wide). */
    fun lowRange(p: DoubleArray): Pair<Double, Double>? {
        val s = p.indexOfFirst { it > CUT }; if (s < 0) return null
        val s1 = s + 1; val b = p.indexOfFirst { it > HI }
        var e = if (b >= 0) b + 1 else s1 + 2
        e = minOf(max(e, s1 + 1), s1 + 3, 9)
        return Pair(s1 - 0.5, e.toDouble())
    }

    /** Standard normal CDF (Abramowitz-Stegun 7.1.26 erf, error < 2e-7). */
    fun normCdf(z: Double): Double {
        val x = abs(z) / sqrt(2.0); val t = 1 / (1 + 0.3275911 * x)
        val erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
        return if (z >= 0) 0.5 * (1 + erf) else 0.5 * (1 - erf)
    }

    private fun v(g: Double) = if (g < 40) "under 40" else "${(g / 5).roundToLong() * 5}"
    private fun clock(ms: Long, fmt: (Long) -> String) = fmt((ms / 300_000.0).roundToLong() * 300_000L)

    /** Two lines, e.g. "Night low: likely 2:00 am – 4:00 am (55%)" / "Wake-up: about 110 (50–170)". */
    fun text(bedMs: Long, r: Result, fmt: (Long) -> String): Pair<String, String> {
        val pct = "${(r.pMax * 100).toInt()}%"
        val low = if (r.alarm && r.rangeH != null)
            "Night low: likely ${clock(bedMs + (r.rangeH.first * 3600_000).toLong(), fmt)} – ${clock(bedMs + (r.rangeH.second * 3600_000).toLong(), fmt)} ($pct)"
        else "Night low: not likely (highest chance $pct)"
        return Pair(low, "Wake-up: about ${v(r.wake)} (${v(max(r.wakeLo, 0.0))}–${v(r.wakeHi)})")
    }
}
