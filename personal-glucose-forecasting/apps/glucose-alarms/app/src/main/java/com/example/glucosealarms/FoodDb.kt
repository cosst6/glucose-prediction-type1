package com.example.glucosealarms

// AUTO-GENERATED from food_db.csv (same table the laptop models use). Values per 100 g (per 100 ml for milk).
data class Food(val key: String, val pattern: Regex, val unit: String, val defaultQty: Double, val gramsPerUnit: Double,
                val carbs: Double, val protein: Double, val fat: Double, val fiber: Double, val sugar: Double, val gi: Double)

object FoodDb {
    val ROWS = listOf(
        Food("paneer_from_milk", Regex("paneer from"), "ml_milk", 750.0, 0.2, 3.0, 18.0, 22.0, 0.0, 2.0, 30.0),
        Food("milk_buffalo", Regex("buffalo"), "ml", 250.0, 1.0, 5.0, 3.8, 6.5, 0.0, 5.0, 39.0),
        Food("milk_toned", Regex("taza|toned"), "ml", 250.0, 1.0, 4.7, 3.1, 3.0, 0.0, 4.7, 32.0),
        Food("milk_cow", Regex("cow milk|cow"), "ml", 250.0, 1.0, 4.8, 3.2, 3.5, 0.0, 4.8, 39.0),
        Food("milk", Regex("milk|shakti|gold"), "ml", 250.0, 1.0, 4.7, 3.2, 6.0, 0.0, 4.7, 39.0),
        Food("paneer", Regex("paneer"), "g", 100.0, 1.0, 3.6, 18.0, 20.0, 0.0, 2.6, 30.0),
        Food("egg", Regex("\\beggs?\\b|omelette|omlete"), "piece", 1.0, 50.0, 1.1, 12.6, 10.0, 0.0, 1.1, 0.0),
        Food("soy", Regex("soy"), "piece", 10.0, 2.0, 33.0, 52.0, 0.5, 13.0, 7.0, 18.0),
        Food("rice", Regex("rice"), "cup", 1.0, 160.0, 28.0, 2.7, 0.3, 0.4, 0.1, 73.0),
        Food("roti", Regex("roti|chapati"), "piece", 1.0, 40.0, 46.0, 9.0, 8.0, 6.0, 2.0, 62.0),
        Food("moong", Regex("moong"), "cup", 1.0, 200.0, 19.0, 7.0, 2.5, 7.0, 2.0, 31.0),
        Food("dal", Regex("dal"), "cup", 1.0, 200.0, 15.0, 6.0, 2.5, 4.0, 1.8, 32.0),
        Food("peas", Regex("peas"), "cup", 1.0, 160.0, 15.0, 5.5, 3.0, 5.5, 5.7, 51.0),
        Food("pumpkin", Regex("pumpkin"), "cup", 1.0, 200.0, 7.0, 1.0, 2.5, 1.0, 2.8, 64.0),
        Food("besan", Regex("besan"), "tbsp", 1.0, 10.0, 58.0, 22.0, 7.0, 11.0, 10.8, 35.0),
        Food("sauce", Regex("tomato onion sauce|sauce"), "tbsp", 1.0, 15.0, 8.0, 1.5, 6.0, 1.5, 5.0, 40.0),
        Food("curd", Regex("curd|dahi"), "g", 80.0, 1.0, 4.7, 3.5, 4.0, 0.0, 4.7, 35.0),
        Food("almond", Regex("almond"), "piece", 10.0, 1.2, 22.0, 21.0, 50.0, 12.0, 4.4, 10.0),
        Food("apple", Regex("apple"), "g", 180.0, 1.0, 13.8, 0.3, 0.2, 2.4, 10.4, 36.0),
        Food("banana", Regex("banana"), "g", 90.0, 1.0, 23.0, 1.1, 0.3, 2.6, 12.2, 51.0),
        Food("guava", Regex("guava"), "g", 200.0, 1.0, 14.0, 2.6, 1.0, 5.4, 8.9, 30.0),
        Food("pomegranate", Regex("pomegranate"), "cup", 1.0, 174.0, 19.0, 1.7, 1.2, 4.0, 13.7, 35.0),
        Food("grape", Regex("grape"), "piece", 1.0, 8.0, 18.0, 0.7, 0.2, 0.9, 15.5, 56.0),
        Food("cucumber", Regex("cucumber"), "g", 150.0, 1.0, 3.6, 0.7, 0.1, 0.5, 1.7, 15.0),
        Food("melody", Regex("melody"), "piece", 1.0, 1.0, 375.0, 0.0, 0.0, 0.0, 260.0, 70.0),
        Food("fish_oil", Regex("fish oil"), "piece", 1.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
        Food("raw_veg", Regex("raw veg"), "cup", 1.0, 100.0, 5.0, 1.5, 0.2, 2.0, 2.5, 15.0),
        Food("veg", Regex("veg|subji|bottlegourd|bottlegoired|parval|okra|bhindi|cabbage|carrot|bean|bittergourd|spinach|gourd|valor|boro"), "cup", 1.0, 180.0, 5.0, 1.5, 4.0, 2.5, 2.5, 30.0),
    )
}
