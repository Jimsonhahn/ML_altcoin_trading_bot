# ML_altcoin_trading_bot/model_builder.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import Sequential # Deaktiviert, da Fokus auf RF
# from tensorflow.keras.layers import LSTM, Dense, Dropout # Deaktiviert
# from tensorflow.keras.callbacks import EarlyStopping # Deaktiviert
import joblib  # For saving/loading sklearn models and scalers
import logging

import config
from data_handler import DataHandler  # Zum Holen der skalierten Daten und des Scalers

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, model_type=config.MODEL_TYPE, model_path=config.MODEL_SAVE_PATH,
                 scaler_path=config.SCALER_SAVE_PATH):
        self.model_type = model_type
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None  # Scaler wird jetzt vom DataHandler geladen/gespeichert

    def build_and_train_model(self, X_train, y_train, X_test, y_test, data_handler_instance):
        if X_train.empty or y_train.empty:
            logger.error("Training data (X_train or y_train) is empty. Cannot train model.")
            return None

        logger.info(f"Starting model training with type: {self.model_type}")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        if self.model_type == 'RandomForest':
            self.model = self._train_random_forest(X_train, y_train)
        elif self.model_type == 'LSTM':
            # LSTM erfordert spezielle Vorbereitung der Sequenzdaten
            # X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, config.LSTM_SEQUENCE_LENGTH)
            # X_test_seq, y_test_seq = self._create_sequences(X_test.values, y_test.values, config.LSTM_SEQUENCE_LENGTH)
            # if X_train_seq.size == 0 or X_test_seq.size == 0:
            #     logger.error("Not enough data to create LSTM sequences after train/test split.")
            #     return None
            # self.model = self._train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
            logger.warning("LSTM training is currently a stub. Implement sequence creation and model fitting.")
            # Fallback or raise error if LSTM is chosen but not fully implemented
            logger.info("Falling back to RandomForest due to incomplete LSTM implementation.")
            self.model = self._train_random_forest(X_train, y_train)  # Fallback
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            return None

        if self.model:
            logger.info("Model training completed.")
            self.evaluate_model(X_test, y_test)  # Evaluate on the hold-out test set
            self.save_model(data_handler_instance.scaler)  # Speichere Modell und den Scaler vom DataHandler
        else:
            logger.error("Model training failed.")

        return self.model

    def _train_random_forest(self, X_train, y_train):
        logger.info("Training RandomForest model with GridSearchCV...")
        # TimeSeriesSplit für Cross-Validation bei Zeitreihendaten
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)

        rf_model = RandomForestClassifier(random_state=42)

        # GridSearchCV
        # Stellen Sie sicher, dass config.RF_PARAM_GRID für RandomForestClassifier gültige Parameter enthält
        grid_search = GridSearchCV(estimator=rf_model, param_grid=config.RF_PARAM_GRID,
                                   cv=tscv, n_jobs=-1, verbose=1,
                                   scoring='f1_weighted')  # f1_weighted ist gut für ungleiche Klassen

        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
            logger.info(f"Best F1-score (weighted) on validation sets: {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during RandomForest GridSearchCV: {e}")
            # Fallback auf Default-Modell, wenn GridSearch fehlschlägt
            logger.warning("Falling back to RandomForest with default parameters.")
            default_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            default_rf.fit(X_train, y_train)
            return default_rf

    # def _create_sequences(self, X_data, y_data, sequence_length):
    #     logger.info(f"Creating sequences with length {sequence_length}...")
    #     Xs, ys = [], []
    #     if len(X_data) <= sequence_length:
    #         logger.warning(f"Data length {len(X_data)} is less than or equal to sequence length {sequence_length}. Cannot create sequences.")
    #         return np.array(Xs), np.array(ys)

    #     for i in range(len(X_data) - sequence_length):
    #         Xs.append(X_data[i:(i + sequence_length)])
    #         ys.append(y_data[i + sequence_length]) # Ziel ist die Kerze direkt nach der Sequenz
    #     logger.info(f"Created {len(Xs)} sequences.")
    #     return np.array(Xs), np.array(ys)

    # def _train_lstm(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    #     # Diese Funktion ist ein STUB und muss implementiert werden, wenn LSTM genutzt wird.
    #     logger.info("Training LSTM model...")
    #     if X_train_seq.ndim != 3:
    #         logger.error(f"X_train_seq must be 3-dimensional (samples, timesteps, features), got {X_train_seq.ndim}")
    #         return None

    #     model = Sequential()
    #     # Erste LSTM Schicht
    #     model.add(LSTM(units=config.LSTM_UNITS[0], return_sequences=len(config.LSTM_UNITS) > 1,
    #                    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    #     model.add(Dropout(config.LSTM_DROPOUT))

    #     # Weitere LSTM Schichten, falls definiert
    #     for i in range(1, len(config.LSTM_UNITS)):
    #         model.add(LSTM(units=config.LSTM_UNITS[i],
    #                        return_sequences=(i < len(config.LSTM_UNITS) - 1))) # True außer für die letzte LSTM Schicht
    #         model.add(Dropout(config.LSTM_DROPOUT))

    #     # Output Layer - Annahme: Target ist 0, 1, 2 -> 3 Klassen
    #     # Passen Sie die Anzahl der Units im Dense Layer und die Aktivierungsfunktion an Ihre Zielvariable an
    #     num_classes = len(np.unique(y_train_seq)) # Ermittelt die Anzahl der Klassen dynamisch
    #     if num_classes <= 1 :
    #         logger.error(f"LSTM training requires at least 2 classes in y_train_seq, found {num_classes}")
    #         return None
    #     activation_func = 'softmax' if num_classes > 2 else 'sigmoid' # Sigmoid für binär, Softmax für multi-class
    #     output_units = num_classes if num_classes > 2 else 1

    #     model.add(Dense(units=output_units, activation=activation_func))

    #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
    #                   metrics=['accuracy']) # sparse_categorical_crossentropy, wenn y nicht one-hot encoded ist

    #     logger.info(model.summary())

    #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #     model.fit(X_train_seq, y_train_seq,
    #               epochs=config.LSTM_EPOCHS,
    #               batch_size=config.LSTM_BATCH_SIZE,
    #               validation_data=(X_val_seq, y_val_seq),
    #               callbacks=[early_stopping],
    #               verbose=1)
    #     return model

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            logger.error("Model is not trained. Cannot evaluate.")
            return

        if X_test.empty or y_test.empty:
            logger.warning("Test data is empty. Skipping evaluation.")
            return

        logger.info("Evaluating model on the test set...")

        # Für LSTM: X_test muss auch in Sequenzen umgewandelt werden
        # if self.model_type == 'LSTM':
        #     if X_test.ndim == 2 : # Wenn X_test noch nicht sequenziert ist
        #         X_test_eval, y_test_eval = self._create_sequences(X_test.values, y_test.values, config.LSTM_SEQUENCE_LENGTH)
        #     else: # Annahme: X_test ist bereits sequenziert (z.B. X_test_seq von oben)
        #         X_test_eval, y_test_eval = X_test, y_test # Hier ist Vorsicht geboten, y_test muss passen

        #     if X_test_eval.size == 0:
        #         logger.warning("Not enough test data to create LSTM sequences for evaluation.")
        #         return
        #     # LSTM gibt Wahrscheinlichkeiten aus, muss in Klassen umgewandelt werden
        #     y_pred_proba = self.model.predict(X_test_eval)
        #     if y_pred_proba.shape[1] > 1: # Softmax (Multi-Class)
        #         y_pred = np.argmax(y_pred_proba, axis=1)
        #     else: # Sigmoid (Binär)
        #         y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        #     target_names = [f'class {i}' for i in sorted(y_test_eval.unique())] # y_test_eval statt y_test
        #     print(classification_report(y_test_eval, y_pred, target_names=target_names, zero_division=0))
        #     print(confusion_matrix(y_test_eval, y_pred))

        # elif self.model_type == 'RandomForest':
        if self.model_type == 'RandomForest':  # Angepasst, da LSTM-Teil auskommentiert ist
            y_pred = self.model.predict(X_test)
            # Dynamische target_names basierend auf den einzigartigen Werten in y_test
            unique_labels = sorted(y_test.unique())
            target_names = [f'class {i}' for i in unique_labels]
            if len(unique_labels) < 2:
                logger.warning(
                    f"Only one class ({unique_labels}) present in y_test. Classification report might be uninformative.")

            print("\n--- Model Evaluation on Test Set ---")
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred, labels=unique_labels))
        else:
            logger.warning(f"Evaluation logic not implemented for model type: {self.model_type}")

    def save_model(self, scaler_instance):
        if self.model is None:
            logger.error("No model to save.")
            return
        if scaler_instance is None:
            logger.error("Scaler instance is None. Cannot save scaler.")
            return

        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            joblib.dump(scaler_instance, self.scaler_path)  # Speichere den Scaler vom DataHandler
            logger.info(f"Scaler saved to {self.scaler_path}")
        except Exception as e:
            logger.error(f"Error saving model or scaler: {e}")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            self.scaler = joblib.load(self.scaler_path)  # Lade den Scaler
            logger.info(f"Scaler loaded from {self.scaler_path}")
            return self.model, self.scaler
        except FileNotFoundError:
            logger.error(f"Model file {self.model_path} or scaler file {self.scaler_path} not found.")
            return None, None
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            return None, None

    def predict(self, X_live_features):
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not loaded. Cannot predict.")
            # Versuche zu laden, falls noch nicht geschehen
            if self.load_model()[0] is None:  # load_model gibt (model, scaler) zurück
                return None

        if X_live_features.empty:
            logger.warning("Live features are empty. Cannot predict.")
            return None

        # Sicherstellen, dass X_live_features die gleiche Anzahl an Spalten hat wie beim Training
        # Das Skalieren sollte bereits in data_handler.get_latest_features erfolgen.
        # X_live_features sollte bereits skaliert sein und als DataFrame oder Numpy-Array ankommen.

        # if self.model_type == 'LSTM':
        #     # Für LSTM: X_live_features muss eine Sequenz sein
        #     # Annahme: X_live_features ist bereits als Sequenz vorbereitet (samples, timesteps, features)
        #     if X_live_features.ndim != 3:
        #         logger.error(f"LSTM prediction expects 3D input, got {X_live_features.ndim}D. Ensure data is sequenced.")
        #         return None
        #     prediction_proba = self.model.predict(X_live_features)
        #     if prediction_proba.shape[1] > 1: # Softmax
        #         prediction = np.argmax(prediction_proba, axis=1)
        #     else: # Sigmoid
        #         prediction = (prediction_proba > 0.5).astype(int).flatten()
        #     return prediction[0] # Return single prediction

        # elif self.model_type == 'RandomForest':
        if self.model_type == 'RandomForest':
            # RandomForest erwartet 2D Input (samples, features)
            if X_live_features.ndim == 1:  # Falls es ein einzelner Sample als 1D Array ist
                X_live_features = X_live_features.reshape(1, -1)
            elif isinstance(X_live_features, pd.DataFrame) and len(X_live_features) == 1:
                X_live_features = X_live_features.values  # Konvertiere zu numpy array
            # else: Annahme es ist bereits korrekt als 2D numpy array

            # Feature-Namen Konsistenz prüfen (optional, aber gut für Debugging)
            # if hasattr(self.model, 'feature_names_in_') and isinstance(X_live_features, pd.DataFrame):
            #     if not all(col in self.model.feature_names_in_ for col in X_live_features.columns):
            #         logger.error("Feature mismatch between training and live data.")
            #         # return None # Oder versuchen, Spalten anzupassen

            prediction = self.model.predict(X_live_features)
            prediction_proba = self.model.predict_proba(X_live_features)
            logger.info(f"Prediction: {prediction[0]}, Probabilities: {prediction_proba[0]}")
            return prediction[0], prediction_proba[0]  # Einzelne Vorhersage und Wahrscheinlichkeiten
        else:
            logger.error(f"Prediction logic not implemented for model type {self.model_type}")
            return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Daten vorbereiten
    data_handler = DataHandler()
    logger.info("Fetching and preprocessing data...")
    # Reduziere limit_days für schnellen Test, für echtes Training config.HISTORICAL_DATA_DAYS verwenden
    raw_data = data_handler.fetch_ohlcv(limit_days=150)  # z.B. 150 Tage für Test

    if raw_data.empty:
        logger.error("Failed to fetch data. Exiting.")
        exit()

    # Hier wird der Scaler im data_handler gefittet
    X_processed, y_processed = data_handler.preprocess_data(fit_scaler=True)

    if X_processed.empty or y_processed.empty:
        logger.error("Data preprocessing resulted in empty X or y. Exiting.")
        exit()

    # 2. Train/Test Split (Zeitreihenfreundlich)
    # Manuelle Aufteilung, da TimeSeriesSplit in GridSearchCV verwendet wird
    # Wir nehmen z.B. die letzten 20% der Daten als Hold-Out Test Set
    split_ratio = 0.8
    split_index = int(len(X_processed) * split_ratio)

    X_train_df = X_processed.iloc[:split_index]
    y_train_series = y_processed.iloc[:split_index]
    X_test_df = X_processed.iloc[split_index:]
    y_test_series = y_processed.iloc[split_index:]

    if X_train_df.empty or X_test_df.empty:
        logger.error("Train or test set is empty after split. Adjust split_ratio or data size.")
        exit()

    logger.info(f"Train set size: X={X_train_df.shape}, y={y_train_series.shape}")
    logger.info(f"Test set size: X={X_test_df.shape}, y={y_test_series.shape}")
    logger.info(f"y_train class distribution:\n{y_train_series.value_counts(normalize=True)}")
    logger.info(f"y_test class distribution:\n{y_test_series.value_counts(normalize=True)}")

    # 3. Modell bauen und trainieren
    # Der ModelBuilder verwendet jetzt den Scaler vom DataHandler
    model_builder = ModelBuilder(model_type=config.MODEL_TYPE)
    trained_model = model_builder.build_and_train_model(X_train_df, y_train_series, X_test_df, y_test_series,
                                                        data_handler)

    if trained_model:
        logger.info("Model training and saving process finished.")

        # 4. Test der Vorhersage mit geladenem Modell
        logger.info("\n--- Testing prediction with loaded model ---")
        loaded_model_builder = ModelBuilder(model_type=config.MODEL_TYPE)
        model, scaler = loaded_model_builder.load_model()  # Scaler wird auch geladen

        if model and scaler:
            # Aktualisiere den Scaler im DataHandler mit dem geladenen Scaler
            data_handler.scaler = scaler
            data_handler.fitted_scaler = True  # Wichtig!

            logger.info("Getting latest features for a test prediction...")
            # Die Anzahl der Kerzen für get_latest_features muss ausreichend sein
            # Für RF reichen die MA-Windows + Puffer. Für LSTM: sequence_length + MA-Windows + Puffer
            min_candles_needed = max(config.MA_WINDOWS) + 20  # Puffer für andere Indikatoren
            # if config.MODEL_TYPE == 'LSTM':
            #     min_candles_needed = max(min_candles_needed, config.LSTM_SEQUENCE_LENGTH + max(config.MA_WINDOWS) + 10)

            # X_live, current_price = data_handler.get_latest_features(n_candles=min_candles_needed)

            # Für Testzwecke nehmen wir einfach die letzte Zeile der Testdaten als "live"
            X_live_test = X_test_df.iloc[[-1]]  # Nehmen die letzte Zeile des Test-Sets als "live"
            current_price_test = data_handler.data['close'].loc[X_live_test.index].iloc[0]

            if not X_live_test.empty:
                logger.info(f"Using features for timestamp {X_live_test.index[0]} for prediction test.")
                # X_live_test ist bereits skaliert und ein DataFrame
                prediction_result, prediction_proba_result = loaded_model_builder.predict(
                    X_live_test)  # predict erwartet DataFrame
                if prediction_result is not None:
                    logger.info(f"Prediction for latest data (simulated): {prediction_result}")
                    logger.info(f"Probabilities: {prediction_proba_result}")
                    logger.info(f"Actual current price (simulated): {current_price_test}")
                else:
                    logger.error("Failed to get a prediction for live data.")
            else:
                logger.error("Could not get latest features for prediction test.")
        else:
            logger.error("Failed to load model or scaler for prediction test.")
    else:
        logger.error("Model training failed.")