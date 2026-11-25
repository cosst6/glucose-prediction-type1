package com.example.glucosealarms

/**
 * Tap-and-step food list (no typing). Each choice builds the same text the meal parser and the training logs use,
 * e.g. "250 ml milk; 100 gm paneer; 1 cup veg cooked in indian style", so the models get exactly the same numbers.
 */
object FoodChoices {
    data class Food(val key: String, val label: String, val unit: String, val step: Double, val first: Double, val template: String, val common: Boolean)
    val ALL = listOf(
        Food("milk", "Milk (Gold)", "ml", 50.0, 250.0, "%s ml milk", true),
        Food("paneer", "Paneer", "g", 25.0, 100.0, "%s gm paneer", true),
        Food("egg", "Boiled egg", "", 1.0, 2.0, "%s boiled egg", true),
        Food("veg", "Cooked veg", "cup", 0.5, 1.0, "%s cup veg cooked in indian style", true),
        Food("roti", "Roti", "", 1.0, 1.0, "%s roti", true),
        Food("dal", "Dal", "cup", 0.5, 1.0, "%s cup dal", true),
        Food("moong", "Moong", "cup", 0.5, 1.0, "%s cup moong", true),
        Food("rice", "Rice", "cup", 0.5, 1.0, "%s cup rice", true),
        Food("banana", "Banana", "g", 30.0, 90.0, "%s gm banana", true),
        Food("apple", "Apple", "g", 50.0, 150.0, "%s gm apple", true),
        Food("milk_cow", "Cow milk", "ml", 50.0, 250.0, "%s ml cow milk", false),
        Food("milk_toned", "Toned milk", "ml", 50.0, 250.0, "%s ml toned milk", false),
        Food("milk_buffalo", "Buffalo milk", "ml", 50.0, 250.0, "%s ml buffalo milk", false),
        Food("curd", "Curd / dahi", "cup", 0.5, 1.0, "%s cup curd", false),
        Food("raw_veg", "Salad / raw veg", "cup", 0.5, 1.0, "%s cup raw veg", false),
        Food("peas", "Peas", "cup", 0.5, 1.0, "%s cup peas", false),
        Food("pumpkin", "Pumpkin", "cup", 0.5, 1.0, "%s cup pumpkin", false),
        Food("soy", "Soy chunks", "g", 25.0, 50.0, "%s gm soy", false),
        Food("besan", "Besan", "tbsp", 1.0, 2.0, "%s tbsp besan", false),
        Food("almond", "Almonds", "", 5.0, 10.0, "%s almond", false),
        Food("guava", "Guava", "g", 50.0, 100.0, "%s gm guava", false),
        Food("pomegranate", "Pomegranate", "g", 50.0, 100.0, "%s gm pomegranate", false),
        Food("grape", "Grapes", "", 5.0, 10.0, "%s grape", false),
        Food("cucumber", "Cucumber", "g", 50.0, 100.0, "%s gm cucumber", false),
        Food("melody", "Melody (sweet)", "", 1.0, 1.0, "%s melody", false),
        Food("other_carb", "Other food (carbs)", "g carbs", 5.0, 15.0, "%s gm carb other", false))

    fun num(q: Double): String = if (q == Math.floor(q)) q.toLong().toString() else q.toString()
    fun up(f: Food, q: Double) = if (q <= 0.0) f.first else q + f.step
    fun down(f: Food, q: Double) = if (q - f.step < f.step / 2) 0.0 else q - f.step
    fun show(f: Food, q: Double) = if (q <= 0.0) "0" else "${num(q)}${if (f.unit.isEmpty()) "" else " " + f.unit}"

    /** The meal text saved in the log (same words as the old typed logs). Empty when nothing is picked. */
    fun text(qty: Map<String, Double>): String = ALL.filter { (qty[it.key] ?: 0.0) > 0.0 }.joinToString("; ") { it.template.format(num(qty[it.key]!!)) }
}
