package com.example.glucosealarms

/** The models this app can show. The user picks which ones to see (and get notifications from). */
object ModelCatalog {
    data class Info(val key: String, val short: String, val title: String, val what: String)
    val ALL = listOf(
        Info("low1", "LOW-1", "Low alarm (under 70 in ~2 h)", "Checks every 15 min. Needs ≥0.5 u NovoRapid on board (switch below)."),
        Info("spike1", "SPIKE-1", "Meal spike size (asks for readings)", "Size band after a meal. Uses your pre-meal reading; may ask for one at 45 min."),
        Info("spike2", "SPIKE-2", "Spike size + time ranges (no reading after the meal)", "Size band and when the rise starts and peaks, as ranges."),
        Info("rise40", "RISE40", "Rise 40+ alert with start & peak ranges", "Any rise of 40 or more, fast or slow: when it starts, passes +40 and peaks."),
        Info("night1", "NIGHTLOW-1", "Night low + wake-up value", "Tap \"Going to sleep\" with a bedtime reading: chance of a low overnight (time range) and your wake-up value."))
    const val PREF = "shownModels"
    /** Picked models. A model added in a later version (e.g. NIGHTLOW-1) starts switched ON even if the user picked before. */
    fun shown(ctx: android.content.Context): Set<String> {
        val p = ctx.getSharedPreferences(AlarmEngine.PREFS, android.content.Context.MODE_PRIVATE)
        val all = ALL.map { it.key }.toSet()
        val saved = p.getStringSet(PREF, null) ?: return all
        val known = p.getStringSet("knownModels", null) ?: setOf("low1", "spike1", "spike2", "rise40")
        return saved + (all - known)
    }
    fun setShown(ctx: android.content.Context, s: Set<String>) =
        ctx.getSharedPreferences(AlarmEngine.PREFS, android.content.Context.MODE_PRIVATE).edit().putStringSet(PREF, HashSet(s)).putStringSet("knownModels", HashSet(ALL.map { it.key })).apply()
}
