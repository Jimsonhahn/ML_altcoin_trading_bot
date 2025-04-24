# ML-Clustering Konfiguration

Diese Dateien enthalten die Konfiguration für das Asset-Clustering im Trading-Bot.

## Clustering-Konfiguration (clustering_config.json)

- `n_clusters`: Anzahl der Cluster (wurde angepasst, um das 'n_samples=1 should be >= n_clusters=3' Problem zu beheben)
- `algorithm`: Clustering-Algorithmus (kmeans)
- `random_state`: Zufallszahlengenerator-Seed für reproduzierbare Ergebnisse
- `key_features`: Features, die für das Clustering verwendet werden
- `scaling`: Skalierungsmethode für Features

## ML-Konfiguration (ml_config.json)

- `use_clustering`: Ob Clustering verwendet werden soll
- `clustering_file`: Pfad zur Datei mit Clustering-Ergebnissen
- `min_samples_per_cluster`: Mindestanzahl an Samples pro Cluster
- `validation_size`: Anteil der Daten für Validierung
- `test_size`: Anteil der Daten für Test
- `random_state`: Zufallszahlengenerator-Seed
- `training_algorithms`: ML-Algorithmen für Training
- `feature_importance_threshold`: Schwellenwert für Feature-Wichtigkeit

## Verwendung

Diese Konfigurationen sollten das Problem 'n_samples=1 should be >= n_clusters=3' beheben,
indem die Cluster-Anzahl auf eine angemessene Größe reduziert wurde und sichergestellt wird,
dass genügend Daten für jedes Symbol verfügbar sind.

Die Visualisierungen im 'visualizations'-Verzeichnis zeigen die Clustering-Ergebnisse.
