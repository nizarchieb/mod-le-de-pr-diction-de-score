from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract

# Initialiser une session Spark
spark = SparkSession.builder.appName("PredictionMatchParisSG").getOrCreate()

# Charger le fichier texte contenant les résultats des matchs
file_path = "C:/Users/21651/Desktop/resultats.txt"
df = spark.read.text(file_path)

# Afficher les premières lignes pour vérifier
df.show(truncate=False)

# Extraire les informations nécessaires : équipe adverse et score
df_extracted = df.select(
    regexp_extract('value', r'Paris SG vs (\w+)', 1).alias('adversaire'),  # Extraire le nom de l'adversaire
    regexp_extract('value', r'(\d+)-(\d+)', 1).cast('int').alias('score_paris'),  # Score de Paris SG
    regexp_extract('value', r'(\d+)-(\d+)', 2).cast('int').alias('score_adverse')  # Score de l'adversaire
)

# Afficher le DataFrame extrait
df_extracted.show()

# Calculer le résultat de Paris SG (1 si victoire, 0 si défaite, 0.5 pour match nul)
df_with_result = df_extracted.withColumn(
    'resultat_paris', 
    when(col('score_paris') > col('score_adverse'), 1)  # Victoire
    .when(col('score_paris') < col('score_adverse'), 0)  # Défaite
    .otherwise(0.5)  # Match nul
)

# Afficher les résultats avec le calcul
df_with_result.show()

# Calculer la moyenne des résultats des 15 derniers matchs pour prédire le prochain match
# Supposons que nous avons besoin de la moyenne des 15 derniers résultats
df_last_15_results = df_with_result.orderBy("value", ascending=False).limit(15)  # Derniers 15 matchs
average_result = df_last_15_results.agg({'resultat_paris': 'avg'}).collect()[0][0]

# Prédiction basée sur la moyenne : Si la moyenne est supérieure à 0.5, prédire victoire
predicted_result = "Victoire" if average_result > 0.5 else "Défaite" if average_result < 0.5 else "Match nul"

print(f"Prédiction du résultat du prochain match : {predicted_result}")
