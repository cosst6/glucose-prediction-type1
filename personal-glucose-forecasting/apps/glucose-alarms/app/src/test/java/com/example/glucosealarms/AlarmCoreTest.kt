package com.example.glucosealarms

import org.junit.Assert.assertEquals
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Locale

/** Golden values produced by the Python training code (parse_meal, glucoforecast9, sklearn joblib files). */
class AlarmCoreTest {
    private val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)
    private fun t(s: String) = fmt.parse(s)!!.time

    @Test fun mealParserMatchesPython() {
        MealParser.parse("250 ml milk").second.let { assertEquals(11.8, it.carbs, 1e-9); assertEquals(15.0, it.fat, 1e-9); assertEquals(8.0, it.protein, 1e-9); assertEquals(4.6, it.gl, 1e-9); assertEquals(0.0, it.freeSugar, 1e-9); assertEquals(0.0, it.starch, 1e-9) }
        MealParser.parse("3 boiled egg; 1 cup moong").second.let { assertEquals(39.6, it.carbs, 1e-9); assertEquals(20.0, it.fat, 1e-9); assertEquals(32.9, it.protein, 1e-9); assertEquals(7.4, it.gl, 1e-9); assertEquals(2.3, it.freeSugar, 1e-9); assertEquals(20.0, it.starch, 1e-9) }
        MealParser.parse("2 roti; dal; veg").second.let { assertEquals(75.8, it.carbs, 1e-9); assertEquals(18.6, it.fat, 1e-9); assertEquals(21.9, it.protein, 1e-9); assertEquals(28.2, it.gl, 1e-9); assertEquals(6.2, it.freeSugar, 1e-9); assertEquals(48.8, it.starch, 1e-9) }
        MealParser.parse("180 g apple").second.let { assertEquals(24.8, it.carbs, 1e-9); assertEquals(0.4, it.fat, 1e-9); assertEquals(0.5, it.protein, 1e-9); assertEquals(7.4, it.gl, 1e-9); assertEquals(12.2, it.freeSugar, 1e-9); assertEquals(1.8, it.starch, 1e-9) }
        MealParser.parse("30 gm carb apple").second.let { assertEquals(30.0, it.carbs, 1e-9); assertEquals(0.0, it.fat, 1e-9); assertEquals(0.0, it.protein, 1e-9); assertEquals(16.5, it.gl, 1e-9); assertEquals(15.0, it.freeSugar, 1e-9); assertEquals(15.0, it.starch, 1e-9) }
        MealParser.parse("paneer from 750 ml milk").second.let { assertEquals(4.5, it.carbs, 1e-9); assertEquals(33.0, it.fat, 1e-9); assertEquals(27.0, it.protein, 1e-9); assertEquals(1.4, it.gl, 1e-9); assertEquals(0.0, it.freeSugar, 1e-9); assertEquals(1.5, it.starch, 1e-9) }
        MealParser.parse("1 banana").second.let { assertEquals(20.7, it.carbs, 1e-9); assertEquals(0.3, it.fat, 1e-9); assertEquals(1.0, it.protein, 1e-9); assertEquals(9.4, it.gl, 1e-9); assertEquals(10.2, it.freeSugar, 1e-9); assertEquals(7.4, it.starch, 1e-9) }
        MealParser.parse("2 tbsp besan").second.let { assertEquals(11.6, it.carbs, 1e-9); assertEquals(1.4, it.fat, 1e-9); assertEquals(4.4, it.protein, 1e-9); assertEquals(3.3, it.gl, 1e-9); assertEquals(1.4, it.freeSugar, 1e-9); assertEquals(7.2, it.starch, 1e-9) }
        MealParser.parse("100 g paneer").second.let { assertEquals(3.6, it.carbs, 1e-9); assertEquals(20.0, it.fat, 1e-9); assertEquals(18.0, it.protein, 1e-9); assertEquals(1.1, it.gl, 1e-9); assertEquals(0.0, it.freeSugar, 1e-9); assertEquals(1.0, it.starch, 1e-9) }
        MealParser.parse("unknown thing").second.let { assertEquals(0.0, it.carbs, 1e-9); assertEquals(0.0, it.fat, 1e-9); assertEquals(0.0, it.protein, 1e-9); assertEquals(0.0, it.gl, 1e-9); assertEquals(0.0, it.freeSugar, 1e-9); assertEquals(0.0, it.starch, 1e-9) }
        MealParser.parse("1 cup rice; 1 cup curd").second.let { assertEquals(56.1, it.carbs, 1e-9); assertEquals(10.1, it.fat, 1e-9); assertEquals(12.7, it.protein, 1e-9); assertEquals(36.3, it.gl, 1e-9); assertEquals(0.2, it.freeSugar, 1e-9); assertEquals(44.0, it.starch, 1e-9) }
        MealParser.parse("10 almond").second.let { assertEquals(2.6, it.carbs, 1e-9); assertEquals(6.0, it.fat, 1e-9); assertEquals(2.5, it.protein, 1e-9); assertEquals(0.1, it.gl, 1e-9); assertEquals(0.1, it.freeSugar, 1e-9); assertEquals(0.7, it.starch, 1e-9) }
        MealParser.parse("5 grape").second.let { assertEquals(7.2, it.carbs, 1e-9); assertEquals(0.1, it.fat, 1e-9); assertEquals(0.3, it.protein, 1e-9); assertEquals(3.8, it.gl, 1e-9); assertEquals(6.2, it.freeSugar, 1e-9); assertEquals(0.6, it.starch, 1e-9) }
        MealParser.parse("melody 2").second.let { assertEquals(7.5, it.carbs, 1e-9); assertEquals(0.0, it.fat, 1e-9); assertEquals(0.0, it.protein, 1e-9); assertEquals(5.2, it.gl, 1e-9); assertEquals(5.2, it.freeSugar, 1e-9); assertEquals(2.3, it.starch, 1e-9) }
    }

    private val events = listOf(
        Event("a", t("2026-11-14 08:00"), "food", "250 ml milk"),
        Event("a2", t("2026-11-14 08:00"), "stick", units = 110.0),
        Event("b", t("2026-11-14 08:05"), "rapid", units = 2.0),
        Event("c", t("2026-11-14 13:00"), "food", "2 roti; dal; veg"),
        Event("d", t("2026-11-14 12:55"), "rapid", units = 4.0),
        Event("e", t("2026-11-14 14:30"), "food", "180 g apple"),
        Event("f", t("2026-11-14 15:00"), "food", "3 boiled egg; 100 g paneer"),
        Event("f2", t("2026-11-14 15:00"), "correction", units = 1.5),
        Event("g", t("2026-11-14 22:00"), "lantus", units = 28.0))

    @Test fun windowFeaturesMatchPython() {
        val ch = Features.channels(events)
        Features.window(ch, t("2026-11-14 08:30"), 60).let { assertEquals(0.6686542141579217, it.insNext, 1e-6); assertEquals(5.194000000000001, it.carbNext, 1e-6); assertEquals(0.2672, it.fpuNext, 1e-6); assertEquals(1.8915020276235333, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 08:30"), 120).let { assertEquals(1.3123081896092363, it.insNext, 1e-6); assertEquals(9.279250000000001, it.carbNext, 1e-6); assertEquals(0.8016, it.fpuNext, 1e-6); assertEquals(1.8915020276235333, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 08:30"), 180).let { assertEquals(1.706590339508987, it.insNext, 1e-6); assertEquals(10.779250000000001, it.carbNext, 1e-6); assertEquals(1.336, it.fpuNext, 1e-6); assertEquals(1.8915020276235333, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 12:30"), 60).let { assertEquals(0.41292269565816286, it.insNext, 1e-6); assertEquals(5.696929831794195, it.carbNext, 1e-6); assertEquals(0.08440000000000003, it.fpuNext, 1e-6); assertEquals(4.020548191804322, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 12:30"), 120).let { assertEquals(1.8154748551948228, it.insNext, 1e-6); assertEquals(34.73801492414248, it.carbNext, 1e-6); assertEquals(0.4924, it.fpuNext, 1e-6); assertEquals(4.020548191804322, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 12:30"), 180).let { assertEquals(3.142844008882917, it.insNext, 1e-6); assertEquals(69.8460328742517, it.carbNext, 1e-6); assertEquals(1.4054, it.fpuNext, 1e-6); assertEquals(5.520548191804322, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 13:40"), 60).let { assertEquals(1.429348930014768, it.insNext, 1e-6); assertEquals(30.100651721393547, it.carbNext, 1e-6); assertEquals(0.47612444444444435, it.fpuNext, 1e-6); assertEquals(3.401423258401236, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 13:40"), 120).let { assertEquals(2.74895214302051, it.insNext, 1e-6); assertEquals(64.64578892455978, it.carbNext, 1e-6); assertEquals(1.5192444444444444, it.fpuNext, 1e-6); assertEquals(4.901423258401236, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 13:40"), 180).let { assertEquals(3.9023266743259404, it.insNext, 1e-6); assertEquals(83.84836309601212, it.carbNext, 1e-6); assertEquals(3.1456977777777775, it.fpuNext, 1e-6); assertEquals(4.901423258401236, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 14:10"), 60).let { assertEquals(1.3639336184624367, it.insNext, 1e-6); assertEquals(33.63906240129184, it.carbNext, 1e-6); assertEquals(0.6922711111111111, it.fpuNext, 1e-6); assertEquals(4.19055956183243, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 14:10"), 120).let { assertEquals(2.669900481321914, it.insNext, 1e-6); assertEquals(60.74442424414371, it.carbNext, 1e-6); assertEquals(2.1028311111111107, it.fpuNext, 1e-6); assertEquals(4.19055956183243, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 14:10"), 180).let { assertEquals(3.580868331923367, it.insNext, 1e-6); assertEquals(73.05772361920873, it.carbNext, 1e-6); assertEquals(3.8995955555555555, it.fpuNext, 1e-6); assertEquals(4.19055956183243, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 16:00"), 60).let { assertEquals(0.9965006922821205, it.insNext, 1e-6); assertEquals(14.55255160412588, it.carbNext, 1e-6); assertEquals(1.7401600000000004, it.fpuNext, 1e-6); assertEquals(1.7211696381830037, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 16:00"), 120).let { assertEquals(1.482975015123746, it.insNext, 1e-6); assertEquals(19.336965582620508, it.carbNext, 1e-6); assertEquals(3.6274400000000004, it.fpuNext, 1e-6); assertEquals(1.7211696381830037, it.insOb, 1e-6) }
        Features.window(ch, t("2026-11-14 16:00"), 180).let { assertEquals(1.6717823349880598, it.insNext, 1e-6); assertEquals(20.193694413265668, it.carbNext, 1e-6); assertEquals(4.746640000000001, it.fpuNext, 1e-6); assertEquals(1.7211696381830037, it.insOb, 1e-6) }
    }

    @Test fun lowModelMatchesSklearn() {
        assertEquals(0.9274575423210673, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 2.047286498801027, "carb0" to 38.01854785303741, "fpu0" to 0.4324788381589012, "iob0" to 1.8972988942744877, "L" to 60.0, "n_hr_15" to 81.16632244862879, "n_steps_60" to 2483.1077814613254, "n_cal_60" to 61.37987045537419)), 1e-9)
        assertEquals(0.8157251556184439, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 164.4228006578813, "ins" to 2.047286498801027, "carb" to 38.01854785303741, "fpu" to 0.4324788381589012, "H" to 90.0, "o_age" to 30.0, "n_hr_15" to 81.16632244862879, "n_steps_60" to 2483.1077814613254, "n_cal_60" to 61.37987045537419)), 1e-9)
        assertEquals(0.8506180609157721, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 0.11023645297227347, "carb0" to 30.140524346992265, "fpu0" to 1.6144299396578345, "iob0" to 0.6594634329981843, "L" to 60.0, "n_hr_15" to 99.42143517142021, "n_steps_60" to 909.584487874935, "n_cal_60" to 68.02468342209772)), 1e-9)
        assertEquals(0.758464365672186, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 85.4679224769613, "ins" to 0.11023645297227347, "carb" to 30.140524346992265, "fpu" to 1.6144299396578345, "H" to 90.0, "o_age" to 30.0, "n_hr_15" to 99.42143517142021, "n_steps_60" to 909.584487874935, "n_cal_60" to 68.02468342209772)), 1e-9)
        assertEquals(0.5138217987169912, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 1.612451945788517, "carb0" to 8.138209627045985, "fpu0" to 0.7869400213255485, "iob0" to 1.5007293452601052, "L" to 60.0, "n_hr_15" to 84.25954872158175, "n_steps_60" to 2942.211599403716, "n_cal_60" to 144.248579049568)), 1e-9)
        assertEquals(0.31619151142680657, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 197.7100887469714, "ins" to 1.612451945788517, "carb" to 8.138209627045985, "fpu" to 0.7869400213255485, "H" to 90.0, "o_age" to 30.0, "n_hr_15" to 84.25954872158175, "n_steps_60" to 2942.211599403716, "n_cal_60" to 144.248579049568)), 1e-9)
        assertEquals(0.7798430603257848, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 2.164907422189737, "carb0" to 11.075648161814833, "fpu0" to 0.4819560263253806, "iob0" to 1.9398508264322651, "L" to 60.0, "n_hr_15" to 85.80342927739393, "n_steps_60" to 347.59683741231095, "n_cal_60" to 93.52346333062506)), 1e-9)
        assertEquals(0.46050054487293335, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 207.56979172503662, "ins" to 2.164907422189737, "carb" to 11.075648161814833, "fpu" to 0.4819560263253806, "H" to 90.0, "o_age" to 30.0, "n_hr_15" to 85.80342927739393, "n_steps_60" to 347.59683741231095, "n_cal_60" to 93.52346333062506)), 1e-9)
        assertEquals(0.9036642941095939, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 2.452013204212162, "carb0" to 36.69190819163611, "fpu0" to 0.11877862999260858, "iob0" to 1.0571785265200433, "L" to 180.0, "n_hr_15" to 63.11747895749378, "n_steps_60" to 1923.9845074181248, "n_cal_60" to 127.89492577209852)), 1e-9)
        assertEquals(0.8433082399044911, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 172.65879343981396, "ins" to 2.452013204212162, "carb" to 36.69190819163611, "fpu" to 0.11877862999260858, "H" to 210.0, "o_age" to 30.0, "n_hr_15" to 63.11747895749378, "n_steps_60" to 1923.9845074181248, "n_cal_60" to 127.89492577209852)), 1e-9)
        assertEquals(0.9002843920535127, ModelParams.LOW_NO_GLUCOSE.prob(mapOf("ins0" to 1.0403897909488928, "carb0" to 33.59526084125635, "fpu0" to 1.5284876445645281, "iob0" to 1.021777768933066, "L" to 120.0, "n_hr_15" to 97.65151038510889, "n_steps_60" to 443.76610735486963, "n_cal_60" to 122.94400786789154)), 1e-9)
        assertEquals(0.6049483533863401, ModelParams.LOW_ONE_STICK.prob(mapOf("o_g0" to 189.82451214061885, "ins" to 1.0403897909488928, "carb" to 33.59526084125635, "fpu" to 1.5284876445645281, "H" to 150.0, "o_age" to 30.0, "n_hr_15" to 97.65151038510889, "n_steps_60" to 443.76610735486963, "n_cal_60" to 122.94400786789154)), 1e-9)
    }

    @Test fun spikeModelMatchesPython() {
        SpikeModel.alarm(12.0, 1.0, 0.0, 80.0, 0.0, null, null).let { assertEquals(1, it.band); assertEquals(0.78, it.p30, 0.006); assertEquals(0.47, it.p60, 0.006); assertEquals(true, it.ask != null) }
        SpikeModel.alarm(12.0, 2.0, 0.0, 80.0, 0.0, 110.0, null).let { assertEquals(1, it.band); assertEquals(0.71, it.p30, 0.006); assertEquals(0.36, it.p60, 0.006); assertEquals(false, it.ask != null) }
        SpikeModel.alarm(5.0, 3.0, 1.0, 75.0, 500.0, 150.0, null).let { assertEquals(0, it.band); assertEquals(0.25, it.p30, 0.006); assertEquals(0.09, it.p60, 0.006); assertEquals(false, it.ask != null) }
        SpikeModel.alarm(12.0, 1.0, 0.0, 90.0, 200.0, 110.0, 35.0).let { assertEquals(1, it.band); assertEquals(0.93, it.p30, 0.006); assertEquals(0.35, it.p60, 0.006); assertEquals(false, it.ask != null) }
        SpikeModel.alarm(40.0, 2.0, 0.5, 85.0, 1500.0, 65.0, null).let { assertEquals(1, it.band); assertEquals(0.66, it.p30, 0.006); assertEquals(0.48, it.p60, 0.006); assertEquals(true, it.ask != null) }
        SpikeModel.alarm(25.0, 1.0, 0.0, 70.0, 0.0, 180.0, 10.0).let { assertEquals(1, it.band); assertEquals(0.64, it.p30, 0.006); assertEquals(0.14, it.p60, 0.006); assertEquals(false, it.ask != null) }
    }

    @Test fun mealGrouping() {
        val m = SpikeModel.meals(events)
        assertEquals(3, m.size)             // 08:00 milk, 13:00 roti, 14:30 apple + 15:00 eggs (within 45 min = one meal)
        assertEquals(2.0, m[0].dose, 1e-9)  // NovoRapid 08:05 belongs to the milk
        assertEquals(4.0, m[1].dose, 1e-9)  // 12:55 dose (5 min before) belongs to lunch
        assertEquals(110.0, SpikeModel.preStick(events, m[0])!!.units!!, 1e-9)
    }

    /** SPIKE-2 v2 + RISE40 (export2.py golden values). */
    @Test fun timingModelsMatchPython() {
        TimingModels.run { val m = MealInputs(5.6, 0.0, 0.0, 15.0, 8.0, true, 0.0, 0.0, 70.0, 0.0, 23, null); val r2 = spike2(m); val r4 = rise40(m)
            assertEquals(0.7231065541041725, r4.p40, 1e-6); assertEquals(0.16726669461750804, r2.lateRisk, 1e-6); assertEquals(55.906780360802756, r2.start.mid, 1e-6); assertEquals(133.89627631114672, r2.peak.mid, 1e-6)
            assertEquals(42.39773115832871, r4.start.mid, 1e-6); assertEquals(63.23014909971153, r4.pass40.mid, 1e-6); assertEquals(139.52606541693441, r4.peak.mid, 1e-6) }
        TimingModels.run { val m = MealInputs(14.0, 20.0, 6.0, 30.0, 30.0, true, 4.0, 0.5, 85.0, 400.0, 13, 110.0); val r2 = spike2(m); val r4 = rise40(m)
            assertEquals(0.6573726597778257, r4.p40, 1e-6); assertEquals(0.5829153609190622, r2.lateRisk, 1e-6); assertEquals(96.82322309101885, r2.start.mid, 1e-6); assertEquals(154.94141643016505, r2.peak.mid, 1e-6)
            assertEquals(62.33290633739445, r4.start.mid, 1e-6); assertEquals(108.3025555128285, r4.pass40.mid, 1e-6); assertEquals(135.1472958945389, r4.peak.mid, 1e-6) }
        TimingModels.run { val m = MealInputs(3.0, 3.0, 3.0, 35.0, 36.0, false, 3.0, 1.2, 76.8, 1200.0, 21, 160.0); val r2 = spike2(m); val r4 = rise40(m)
            assertEquals(0.12623659324349815, r4.p40, 1e-6); assertEquals(0.9522392254121478, r2.lateRisk, 1e-6); assertEquals(137.29611732520686, r2.start.mid, 1e-6); assertEquals(179.55830076024193, r2.peak.mid, 1e-6)
            assertEquals(69.51351298652637, r4.start.mid, 1e-6); assertEquals(145.160722238599, r4.pass40.mid, 1e-6); assertEquals(254.65056128163823, r4.peak.mid, 1e-6) }
        TimingModels.run { val m = MealInputs(12.2, 7.4, 2.6, 0.3, 1.0, false, 0.0, 0.0, 90.0, 50.0, 10, 65.0); val r2 = spike2(m); val r4 = rise40(m)
            assertEquals(0.6381178883065607, r4.p40, 1e-6); assertEquals(0.08196122874548627, r2.lateRisk, 1e-6); assertEquals(33.0972942960892, r2.start.mid, 1e-6); assertEquals(66.18515021158979, r2.peak.mid, 1e-6)
            assertEquals(37.38296498476553, r4.start.mid, 1e-6); assertEquals(52.99181532036246, r4.pass40.mid, 1e-6); assertEquals(57.20805983848925, r4.peak.mid, 1e-6) }
    }
    /** Pre-meal reading: a stick up to 20 min AFTER the meal starts still counts (e.g. eat 8:26, stick 8:41). */
    @Test fun lateStickCountsAsPreMeal() {
        val ev = listOf(Event("f", t("2026-11-15 08:26"), "food", "250 ml milk"), Event("s", t("2026-11-15 08:41"), "stick", units = 140.0))
        val m = SpikeModel.meals(ev)[0]
        assertEquals(140.0, SpikeModel.preStick(ev, m)!!.units!!, 1e-9)
        val late = listOf(ev[0], Event("s2", t("2026-11-15 08:50"), "stick", units = 180.0))   // 24 min after: not pre-meal
        assertEquals(null, SpikeModel.preStick(late, SpikeModel.meals(late)[0]))
    }

    /** Watch gaps: <=10 min filled with a straight line, longer gaps left empty. */
    @Test fun watchGapFill() {
        val rows = WatchCsv.parse(sequenceOf("timestamp,heart_rate,steps,distance_m,calories",
            "2026-11-15 08:00:00,80,10,7,1.0", "2026-11-15 08:02:00,100,10,7,1.0",      // 2-min gap -> 3 filled slots
            "2026-11-15 08:30:00,90,0,0,0.5"))                                            // 28-min gap -> not filled
        val f = WatchCsv.fill(rows)
        assertEquals(3, f.count { it.filled })
        assertEquals(90.0, f.first { it.t == t("2026-11-15 08:01") }.hr!!, 1e-9)
        assertEquals(0, f.count { it.filled && it.t > t("2026-11-15 08:02") })
    }

    /** Food buttons build the same text the parser and training logs use. */
    @Test fun foodChoicesParse() {
        FoodChoices.ALL.forEach { f -> assert(MealParser.parse(f.template.format(FoodChoices.num(f.first))).first.none { it.food == "UNKNOWN" }) { f.key } }
        assertEquals("250 ml milk; 2 roti", FoodChoices.text(mapOf("roti" to 2.0, "milk" to 250.0)))
    }

    /** NIGHTLOW-1 golden cases from golden.json (phys.py). */
    @Test fun nightLowMatchesPython() {
        val cases = listOf(
        Triple(NightLow.Inputs(110.0, 3.0, 10.0, 4.0, 0.0, 65.0, 200.0),
            doubleArrayOf(87.5, 62.2, 44.3, 44.3, 32.2, 34.1, 21.3, 23.2, 39.9) to doubleArrayOf(0.264, 0.604, 0.737, 0.705, 0.756, 0.746, 0.826, 0.818, 0.746), Pair(1.5, 3.0) to 31.6),
        Triple(NightLow.Inputs(110.0, 1.0, 10.0, 4.0, 250.0, 65.0, 200.0),
            doubleArrayOf(161.0, 173.2, 178.6, 189.6, 205.1, 192.9, 176.7, 147.4, 157.6) to doubleArrayOf(0.001, 0.0, 0.004, 0.006, 0.007, 0.012, 0.02, 0.067, 0.027), null to 152.5),
        Triple(NightLow.Inputs(180.0, 2.0, 30.0, 6.0, 0.0, 70.0, 1500.0),
            doubleArrayOf(165.4, 162.3, 158.1, 148.3, 156.0, 166.7, 155.5, 141.6, 153.5) to doubleArrayOf(0.0, 0.001, 0.015, 0.051, 0.057, 0.037, 0.049, 0.082, 0.033), null to 147.6))
        for ((inp, exp, rw) in cases) {
            val r = NightLow.predict(inp)
            for (h in 0 until 9) { assertEquals(exp.first[h], r.pred[h], 0.06); assertEquals(exp.second[h], r.pLow[h], 0.0015) }
            assertEquals(rw.first, r.rangeH); assertEquals(rw.second, r.wake, 0.06)
        }
        // features from the log: 2 u NovoRapid 1 h before bed + 1 u planned 30 min after, 250 ml milk at bed, dinner 3 h before
        val bed = t("2026-11-15 22:30")
        val ev = listOf(Event("a", t("2026-11-15 19:30"), "food", "2 roti; 1 cup dal"), Event("b", t("2026-11-15 21:30"), "rapid", units = 2.0),
            Event("c", t("2026-11-15 22:30"), "food", "250 ml milk"), Event("d", t("2026-11-15 23:00"), "rapid", units = 1.0),
            Event("e", t("2026-11-16 00:30"), "food", "250 ml milk"))                     // 2 h after bed: not counted
        val i = NightLow.inputs(ev, bed, 110.0, Watch(65.0, 0.0, 0.0, 200.0))
        assertEquals(250.0, i.milkMl, 1e-9)
        assertEquals(2 * (1 - Kernels.insulinCum(60.0)) + 1.0, i.iob, 1e-9)
        val p = MealParser.parse("2 roti; 1 cup dal").second; val m = MealParser.parse("250 ml milk").second
        assertEquals(p.avail + m.avail, i.carbs4h, 1e-9)
    }

}
