#  УСТАНОВКА БИБЛИОТЕК

!pip install wfdb neurokit2 xgboost shap imbalanced-learn scikit-learn scipy -q

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, roc_curve, auc, matthews_corrcoef,
                             cohen_kappa_score, average_precision_score,
                             precision_recall_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from scipy.signal import resample, find_peaks
from scipy.stats import skew, kurtosis
import scipy.io as sio
import os
import zipfile
import warnings
import pickle
import joblib
from pathlib import Path
from collections import Counter
import time
import shutil
import requests
import random
from tqdm import tqdm
import shap
warnings.filterwarnings('ignore')

#  ВОСПРОИЗВОДИМОСТЬ (FIXED SEED)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

print("Все библиотеки успешно установлены")
print("Воспроизводимость зафиксирована (seed=42)")

#  МОНТИРОВАНИЕ GOOGLE DRIVE И НАСТРОЙКА ДИРЕКТОРИЙ

from google.colab import drive
drive.mount('/content/drive')

print("\n" + "="*70)
print("НАСТРОЙКА ДИРЕКТОРИЙ")
print("="*70)

PROJECT_DIR = Path("/content/drive/MyDrive/ECG_Project")
CACHE_DIR = PROJECT_DIR / "cache"
MODEL_DIR = PROJECT_DIR / "model"
DATA_DIR = PROJECT_DIR / "data"

for dir_path in [CACHE_DIR, MODEL_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Проектная директория: {PROJECT_DIR}")
print(f"Кэш: {CACHE_DIR}")
print(f"Модели: {MODEL_DIR}")
print(f"Данные: {DATA_DIR}")

#  CACHE MANAGER

class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename, data):
        cache_path = self.cache_dir / filename
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Сохранено: {filename}")
        return True

    def load(self, filename):
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def exists(self, filename):
        return (self.cache_dir / filename).exists()

cache_manager = CacheManager(CACHE_DIR)

#  ЗАГРУЗЧИК LUDB

class ImprovedDatasetLoader:
    def __init__(self, cache_manager, data_dir):
        self.cache_manager = cache_manager
        self.data_dir = Path(data_dir)
        self.LUDB_FLAG = "ludb_downloaded.pkl"

    def download_ludb(self):
        if self.cache_manager.exists(self.LUDB_FLAG):
            print("LUDB уже скачан")
            return self.data_dir / "ludb"

        print("\nСкачивание LUDB...")
        output_dir = self.data_dir / "ludb"
        output_dir.mkdir(exist_ok=True)

        url = "https://physionet.org/content/ludb/get-zip/1.0.1/"
        zip_path = output_dir / "ludb_temp.zip"

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()

            self.cache_manager.save(self.LUDB_FLAG, {"downloaded": True})
            return output_dir
        return None

loader_improved = ImprovedDatasetLoader(cache_manager, DATA_DIR)

#  ЗАГРУЗЧИК РЕАЛЬНЫХ ЖЭ ИЗ PHYSIONET

class PhysioNetVELoader:
    def __init__(self, cache_manager, data_dir):
        self.cache_manager = cache_manager
        self.data_dir = Path(data_dir)
        self.CACHE_FEATURES = "physionet_ve_features_v3.pkl"
        self.MITDB_DIR = DATA_DIR / "mitdb"

    def download_and_extract(self, force=False):
        if not force and self.cache_manager.exists(self.CACHE_FEATURES):
            print("PhysioNet VE данные загружены из кэша")
            return self.cache_manager.load(self.CACHE_FEATURES)

        print("\nЗагрузка MIT-BIH Arrhythmia Database...")
        self.MITDB_DIR.mkdir(parents=True, exist_ok=True)

        records = ['119', '200', '201', '202', '203', '205', '208', '210', '213', '214',
                   '215', '219', '220', '221', '222', '223', '228', '230', '231', '232',
                   '233', '234', '100', '101', '103', '105', '109', '111', '112', '113',
                   '114', '115', '116', '117', '118', '121', '122', '123', '124']

        base_url = "https://physionet.org/files/mitdb/1.0.0/"

        for record_name in tqdm(records, desc="Скачивание записей"):
            for ext in ['.dat', '.hea', '.atr']:
                url = f"{base_url}{record_name}{ext}"
                filepath = self.MITDB_DIR / f"{record_name}{ext}"
                if filepath.exists():
                    continue
                try:
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                except:
                    pass

        features_list = self._extract_features()

        if features_list:
            self.cache_manager.save(self.CACHE_FEATURES, features_list)
        return features_list

    def _extract_features(self):
        print("\nИзвлечение признаков из MIT-BIH...")
        features_list = []
        ve_count = 0
        normal_count = 0

        records_with_ve = ['119', '200', '201', '202', '203', '205', '208', '210', '213', '214',
                          '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
        normal_records = ['100', '101', '103', '105', '109', '111', '112', '113', '114', '115',
                         '116', '117', '118', '121', '122', '123', '124']
        all_records = records_with_ve + normal_records

        for record_name in tqdm(all_records, desc="Обработка"):
            try:
                record_path = str(self.MITDB_DIR / record_name)
                if not Path(f"{record_path}.hea").exists():
                    continue

                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                signal = record.p_signal[:, 0]
                fs = record.fs

                if fs != TARGET_SR:
                    new_length = int(len(signal) * TARGET_SR / fs)
                    signal = resample(signal, new_length)
                    ann_samples = (annotation.sample * TARGET_SR / fs).astype(int)
                else:
                    ann_samples = annotation.sample

                symbols = annotation.symbol

                for i, (sample, symbol) in enumerate(zip(ann_samples, symbols)):
                    if symbol in ['V', 'VE']:
                        ecg_class = 'ЖЭ'
                        ve_count += 1
                    elif symbol in ['N', 'Normal']:
                        ecg_class = 'Норма'
                        normal_count += 1
                    else:
                        continue

                    if ecg_class == 'ЖЭ' and ve_count > 600:
                        continue
                    if ecg_class == 'Норма' and normal_count > 1000:
                        continue

                    half_seg = SEG_LEN // 2
                    start = max(0, sample - half_seg)
                    end = min(len(signal), sample + half_seg)

                    if end - start < SEG_LEN:
                        segment = np.zeros(SEG_LEN)
                        seg_start = max(0, half_seg - sample)
                        seg_end = seg_start + (end - start)
                        segment[seg_start:seg_end] = signal[start:end]
                    else:
                        segment = signal[start:end]

                    features = extract_all_features(segment, TARGET_SR)
                    features['class'] = ecg_class
                    features['source'] = f'MITBIH_{record_name}'
                    features['Patient_ID'] = f"PhysioNet_{record_name}_{i}"
                    features_list.append(features)
            except:
                continue

        return features_list

#  ПАРАМЕТРЫ

SEGMENT_DURATION = 10
TARGET_SR = 360
SEG_LEN = TARGET_SR * SEGMENT_DURATION
CONFIDENCE_THRESHOLD = 0.65

#  ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ

def extract_hrv_frequency_features(rr_intervals, sample_rate=4):
    try:
        if len(rr_intervals) < 5:
            return 0.5, 0.5, 1.0

        time_rr = np.cumsum(rr_intervals)
        time_uniform = np.arange(0, time_rr[-1], 1/sample_rate)
        rr_interpolated = np.interp(time_uniform, time_rr, rr_intervals[:-1] if len(rr_intervals) > len(time_uniform) else rr_intervals)

        from scipy import signal
        nperseg = min(128, len(rr_interpolated) // 2)
        if nperseg < 4:
            return 0.5, 0.5, 1.0

        freqs, psd = signal.welch(rr_interpolated, fs=sample_rate, nperseg=nperseg)

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)

        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        lf_hf_ratio = lf_power / (hf_power + 1e-6)

        return lf_power, hf_power, lf_hf_ratio
    except:
        return 0.5, 0.5, 1.0

def extract_qrs_width_and_t_wave(signal, sample_rate, r_peaks):
    qrs_widths = []
    t_wave_assym = []

    for peak in r_peaks[:10]:
        try:
            start = max(0, peak - int(0.05 * sample_rate))
            end = min(len(signal), peak + int(0.08 * sample_rate))
            qrs_segment = signal[start:end]

            threshold = np.std(qrs_segment) * 0.3
            q_start = np.argmax(np.abs(qrs_segment) > threshold)
            q_end = len(qrs_segment) - np.argmax(np.abs(qrs_segment[::-1]) > threshold)
            qrs_width = (q_end - q_start) / sample_rate * 1000
            qrs_widths.append(qrs_width)

            t_start = peak + int(0.12 * sample_rate)
            t_end = min(len(signal), peak + int(0.35 * sample_rate))
            if t_start < t_end:
                t_wave = signal[t_start:t_end]
                t_wave_assym.append(skew(t_wave))
        except:
            pass

    return np.mean(qrs_widths) if qrs_widths else 80.0, np.mean(t_wave_assym) if t_wave_assym else 0.0

def extract_all_features(signal, sample_rate):
    features = {}
    try:
        if len(signal) < sample_rate:
            signal = np.pad(signal, (0, sample_rate - len(signal)))

        cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
        r_peaks, _ = find_peaks(cleaned, distance=sample_rate*0.3, height=np.std(cleaned)*0.4)

        if len(r_peaks) >= 3:
            rr_intervals = np.diff(r_peaks) / sample_rate
            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]

            if len(rr_intervals) >= 3:
                features['HeartRate'] = float(60.0 / np.mean(rr_intervals))
                features['Mean_RR'] = float(np.mean(rr_intervals))
                features['Std_RR'] = float(np.std(rr_intervals))
                features['Min_RR'] = float(np.min(rr_intervals))
                features['Max_RR'] = float(np.max(rr_intervals))
                features['RR_range'] = features['Max_RR'] - features['Min_RR']
                features['RR_cv'] = features['Std_RR'] / (features['Mean_RR'] + 1e-6)

                nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
                nn20 = np.sum(np.abs(np.diff(rr_intervals)) > 0.02)
                features['pNN50'] = float((nn50 / len(rr_intervals)) * 100)
                features['pNN20'] = float((nn20 / len(rr_intervals)) * 100)

                if len(rr_intervals) >= 3:
                    sd1 = np.std(rr_intervals[:-1] - rr_intervals[1:]) / np.sqrt(2)
                    sd2 = np.std(rr_intervals[:-1] + rr_intervals[1:]) / np.sqrt(2)
                    features['SD1'] = sd1
                    features['SD2'] = sd2
                    features['SD1_SD2_ratio'] = sd1 / (sd2 + 1e-6)
                else:
                    features['SD1'] = features['SD2'] = features['SD1_SD2_ratio'] = 0

                lf_power, hf_power, lf_hf_ratio = extract_hrv_frequency_features(rr_intervals)
                features['LF_power'] = lf_power
                features['HF_power'] = hf_power
                features['LF_HF_ratio'] = lf_hf_ratio
                features['Total_power'] = lf_power + hf_power
                features['LF_norm'] = lf_power / (features['Total_power'] + 1e-6)
                features['HF_norm'] = hf_power / (features['Total_power'] + 1e-6)

                qrs_width, t_assym = extract_qrs_width_and_t_wave(cleaned, sample_rate, r_peaks)
                features['QRS_width'] = qrs_width
                features['T_wave_asymmetry'] = t_assym

            else:
                return _default_features()
        else:
            return _default_features()

        features['Signal_mean'] = float(np.mean(signal))
        features['Signal_std'] = float(np.std(signal))
        features['Signal_rms'] = float(np.sqrt(np.mean(signal**2)))
        features['zero_crossing_rate'] = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))

        try:
            hist, _ = np.histogram(signal, bins=50)
            hist = hist / (len(signal) + 1e-6)
            features['signal_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-6)))
        except:
            features['signal_entropy'] = 0.0

    except Exception as e:
        return _default_features()

    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0.0
    return features

def _default_features():
    return {
        'HeartRate': 75.0, 'Mean_RR': 0.8, 'Std_RR': 0.05, 'Min_RR': 0.7, 'Max_RR': 0.9,
        'RR_range': 0.2, 'RR_cv': 0.0625, 'pNN50': 10.0, 'pNN20': 25.0,
        'SD1': 0.03, 'SD2': 0.07, 'SD1_SD2_ratio': 0.43,
        'LF_power': 0.5, 'HF_power': 0.5, 'LF_HF_ratio': 1.0, 'Total_power': 1.0,
        'LF_norm': 0.5, 'HF_norm': 0.5,
        'Signal_mean': 0.0, 'Signal_std': 0.8, 'Signal_rms': 0.7,
        'zero_crossing_rate': 0.05, 'signal_entropy': 2.5,
        'QRS_width': 80.0, 'T_wave_asymmetry': 0.0
    }

# ОБРАБОТКА LUDB

def process_ludb(ludb_path):
    CACHE_LUDB = "ludb_features_v3.pkl"

    if cache_manager.exists(CACHE_LUDB):
        return cache_manager.load(CACHE_LUDB)

    if not ludb_path or not ludb_path.exists():
        return []

    print("\nОбработка LUDB...")
    features_list = []
    dat_files = list(ludb_path.rglob("*.dat"))

    for dat_file in tqdm(dat_files, desc="Обработка LUDB"):
        try:
            record_path = str(dat_file)[:-4]
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]

            try:
                annotation = wfdb.rdann(record_path, 'atr')
                symbols = set(annotation.symbol)
                if not all(sym in ['N', 'Normal', '|'] for sym in symbols):
                    continue
            except:
                pass

            if record.fs != TARGET_SR:
                new_length = int(len(signal) * TARGET_SR / record.fs)
                signal = resample(signal, new_length)

            num_segments = len(signal) // SEG_LEN

            for i in range(min(num_segments, 2)):
                segment = signal[i*SEG_LEN:(i+1)*SEG_LEN]
                features = extract_all_features(segment, TARGET_SR)
                features['class'] = 'Норма'
                features['source'] = 'LUDB'
                features['Patient_ID'] = f"LUDB_{dat_file.stem}"
                features_list.append(features)
        except:
            continue

    cache_manager.save(CACHE_LUDB, features_list)
    print(f"LUDB: {len(features_list)} сегментов")
    return features_list

#  ЗАГРУЗКА ИСХОДНЫХ ДАТАСЕТОВ

def load_existing_datasets():
    all_data = []

    for name, filename in [("MIT-BIH", "mitbih_features_35.pkl"),
                            ("PTB", "ptb_features_35.pkl"),
                            ("Chapman", "chapman_features_35.pkl")]:
        file_path = CACHE_DIR / filename
        if file_path.exists():
            data = cache_manager.load(filename)
            if data and 'class' in data[0]:
                for item in data:
                    if 'Patient_ID' not in item:
                        item['Patient_ID'] = f"{name}_{item.get('record_name', 'unknown')}"
                    if 'source' not in item:
                        item['source'] = name
                all_data.extend(data)
                ve_count = len([s for s in data if s.get('class') == 'ЖЭ'])
                print(f"{name}: {len(data)} сегментов (ЖЭ: {ve_count})")

    return all_data

#  ЗАГРУЗКА ВСЕХ ДАННЫХ

print("\n" + "="*70)
print("ЗАГРУЗКА ДАННЫХ")
print("="*70)

all_data = load_existing_datasets()

ludb_path = loader_improved.download_ludb()
ludb_data = process_ludb(ludb_path)
if ludb_data:
    all_data.extend(ludb_data)

print("\n" + "="*50)
print("ЗАГРУЗКА PHYSIONET (ДОБАВЛЕНИЕ ЖЭ)")
print("="*50)

ve_loader = PhysioNetVELoader(cache_manager, DATA_DIR)
ve_data = ve_loader.download_and_extract(force=False)
if ve_data:
    ve_only_data = [s for s in ve_data if s.get('class') == 'ЖЭ']
    all_data.extend(ve_only_data)
    print(f"Добавлено {len(ve_only_data)} ЖЭ из PhysioNet")

feature_cols = [
    'HeartRate', 'Mean_RR', 'Std_RR', 'Min_RR', 'Max_RR',
    'RR_range', 'RR_cv', 'pNN50', 'pNN20',
    'SD1', 'SD2', 'SD1_SD2_ratio',
    'LF_power', 'HF_power', 'LF_HF_ratio', 'Total_power',
    'LF_norm', 'HF_norm',
    'Signal_mean', 'Signal_std', 'Signal_rms',
    'zero_crossing_rate', 'signal_entropy',
    'QRS_width', 'T_wave_asymmetry'
]

valid_classes = ['Норма', 'НЭ', 'ЖЭ']
filtered_data = [s for s in all_data if s.get('class') in valid_classes]

print(f"\nВсего данных: {len(filtered_data)} сегментов")
class_counts = Counter([s.get('class') for s in filtered_data])
for cls, count in class_counts.items():
    print(f"   {cls}: {count} ({count/len(filtered_data)*100:.1f}%)")

# ПОДГОТОВКА ДАННЫХ

print("\n" + "="*70)
print("ПОДГОТОВКА ДАННЫХ")
print("="*70)

df = pd.DataFrame(filtered_data)

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

if 'source' not in df.columns:
    df['source'] = 'unknown'

df['group'] = df['source'].astype(str) + "_" + df['Patient_ID'].astype(str)

X_raw = df[feature_cols].fillna(0).values
y_raw = df['class'].values
groups = df['group'].values
patient_ids = df['Patient_ID'].values

print(f"Уникальных пациентов: {len(np.unique(patient_ids))}")

#  КОДИРОВАНИЕ И МАСШТАБИРОВАНИЕ

print("\n" + "="*70)
print("КОДИРОВАНИЕ И МАСШТАБИРОВАНИЕ")
print("="*70)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"Классы: {list(label_encoder.classes_)}")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# PERMUTATION IMPORTANCE (ОТБОР ПРИЗНАКОВ)

print("\n" + "="*70)
print("PERMUTATION IMPORTANCE")
print("="*70)

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X, y)

perm_importance = permutation_importance(base_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({'feature': feature_cols, 'importance': perm_importance.importances_mean})
perm_df = perm_df.sort_values('importance', ascending=False)

print("\nТоп-10 важных признаков:")
for i, row in perm_df.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

n_features_to_keep = min(20, len(feature_cols))
selected_features = perm_df.head(n_features_to_keep)['feature'].tolist()
selected_indices = [feature_cols.index(f) for f in selected_features]
X = X[:, selected_indices]
feature_cols = selected_features
print(f"\nОтобрано {len(selected_features)} признаков")

# GROUPKFOLD CV (5 FOLDS)

print("\n" + "="*70)
print("GROUPKFOLD CV (5 FOLDS)")
print("="*70)

gkf = GroupKFold(n_splits=5)

cv_results = {
    'accuracy': [],
    'weighted_f1': [],
    'mcc': [],
    'kappa': [],
    'specificity': [],  # Добавлено
    'reports': []
}

print("\nЗапуск 5-folds GroupKFold...")

fold = 1
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}/5")
    print(f"{'='*50}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train размер: {len(X_train)}")
    print(f"Test размер: {len(X_test)}")

    class_weight_dict = {}
    for i, cls in enumerate(label_encoder.classes_):
        if cls == 'ЖЭ':
            class_weight_dict[i] = 3.0
        elif cls == 'Норма':
            class_weight_dict[i] = 1.5
        else:
            class_weight_dict[i] = 0.5

    # Обучение XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='mlogloss', use_label_encoder=False, verbosity=0,
        tree_method='hist', deterministic_histogram=True
    )
    xgb_model.fit(X_train, y_train, verbose=False)

    # Обучение Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=42,
        class_weight=class_weight_dict
    )
    rf_model.fit(X_train, y_train)

    # Обучение Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, subsample=0.8
    )
    gb_model.fit(X_train, y_train)

    # Ансамбль моделей с голосованием по вероятностям (soft voting)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('gb', gb_model)],
        voting='soft', weights=[3, 2, 1]
    )
    ensemble.fit(X_train, y_train)

    # Калибровка вероятностей методом изотонической регрессии
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    calibrated_ensemble.fit(X_train, y_train)

    y_pred = calibrated_ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    cv_results['accuracy'].append(acc)
    cv_results['weighted_f1'].append(f1)
    cv_results['mcc'].append(mcc)
    cv_results['kappa'].append(kappa)
    cv_results['reports'].append(classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True))

    # ВЫЧИСЛЕНИЕ SPECIFICITY ДЛЯ КАЖДОГО КЛАССА
    def calc_specificity(y_true, y_pred, pos_label):
        y_true_bin = (y_true == pos_label).astype(int)
        y_pred_bin = (y_pred == pos_label).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    spec_dict = {}
    for i, cls in enumerate(label_encoder.classes_):
        spec_dict[cls] = calc_specificity(y_test, y_pred, i)

    cv_results['specificity'].append(spec_dict)

    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   Weighted F1: {f1*100:.2f}%")
    print(f"   MCC: {mcc:.3f}")
    print(f"   Specificity:")
    for cls, spec in spec_dict.items():
        print(f"      {cls}: {spec:.3f}")

    fold += 1

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ GROUPKFOLD CV (5 FOLDS)")
print("="*70)

print(f"\nМетрики по 5 фолдам:")
print(f"   Accuracy:  {np.mean(cv_results['accuracy'])*100:.2f}% (±{np.std(cv_results['accuracy'])*100:.2f}%)")
print(f"   Weighted F1: {np.mean(cv_results['weighted_f1'])*100:.2f}% (±{np.std(cv_results['weighted_f1'])*100:.2f}%)")
print(f"   MCC: {np.mean(cv_results['mcc']):.3f} (±{np.std(cv_results['mcc']):.3f})")
print(f"   Kappa: {np.mean(cv_results['kappa']):.3f} (±{np.std(cv_results['kappa']):.3f})")

# Средняя specificity по классам
print(f"\n   Средняя Specificity по классам:")
for cls in label_encoder.classes_:
    spec_values = [d[cls] for d in cv_results['specificity']]
    print(f"      {cls}: {np.mean(spec_values):.3f} (±{np.std(spec_values):.3f})")

# BASELINE СРАВНЕНИЕ (Dummy Classifiers)

print("\n" + "="*70)
print("BASELINE СРАВНЕНИЕ (Dummy Classifiers)")
print("="*70)

dummy_mf = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_mf.fit(X_train, y_train)
y_pred_mf = dummy_mf.predict(X_test)

dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
dummy_strat.fit(X_train, y_train)
y_pred_strat = dummy_strat.predict(X_test)

dummy_unif = DummyClassifier(strategy='uniform', random_state=42)
dummy_unif.fit(X_train, y_train)
y_pred_unif = dummy_unif.predict(X_test)

print("\nСравнение с Baseline:")
print("-" * 60)
print(f"{'Стратегия':<15} {'Accuracy':<12} {'Weighted F1':<15}")
print("-" * 60)
print(f"{'most_frequent':<15} {accuracy_score(y_test, y_pred_mf)*100:>6.2f}%      {f1_score(y_test, y_pred_mf, average='weighted', zero_division=0)*100:>6.2f}%")
print(f"{'stratified':<15} {accuracy_score(y_test, y_pred_strat)*100:>6.2f}%      {f1_score(y_test, y_pred_strat, average='weighted', zero_division=0)*100:>6.2f}%")
print(f"{'uniform':<15} {accuracy_score(y_test, y_pred_unif)*100:>6.2f}%      {f1_score(y_test, y_pred_unif, average='weighted', zero_division=0)*100:>6.2f}%")
print("-" * 60)

best_dummy_acc = max([accuracy_score(y_test, y_pred_mf), accuracy_score(y_test, y_pred_strat), accuracy_score(y_test, y_pred_unif)])
print(f"\nЛучший Baseline: {best_dummy_acc*100:.2f}%")

#  УСИЛЕННЫЙ БУТСТРАП (95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ)

print("\n" + "="*70)
print("УСИЛЕННЫЙ БУТСТРАП (95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ)")
print("="*70)

def bootstrap_ci_with_plot(y_true, y_pred, metric, metric_name="Metric", n_bootstrap=1000):
    np.random.seed(42)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = metric(y_true[idx], y_pred[idx])
        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean = np.mean(scores)
    std = np.std(scores)

    print(f"\n{metric_name}:")
    print(f"   Среднее: {mean:.4f}")
    print(f"   Стандартное отклонение: {std:.4f}")
    print(f"   95% ДИ: [{lower:.4f}, {upper:.4f}]")
    print(f"   Ширина ДИ: {(upper - lower):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(lower, color='red', linestyle='--', linewidth=2, label=f'2.5%: {lower:.4f}')
    axes[0].axvline(upper, color='red', linestyle='--', linewidth=2, label=f'97.5%: {upper:.4f}')
    axes[0].axvline(mean, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean:.4f}')
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Bootstrap Distribution of {metric_name} (n={n_bootstrap})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    from scipy import stats
    stats.probplot(scores, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {metric_name}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return lower, upper, mean

acc_lower, acc_upper, acc_mean = bootstrap_ci_with_plot(
    y_test, y_pred, accuracy_score, "Accuracy", n_bootstrap=1000
)

f1_lower, f1_upper, f1_mean = bootstrap_ci_with_plot(
    y_test, y_pred,
    lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0),
    "Weighted F1", n_bootstrap=1000
)

mcc_lower, mcc_upper, mcc_mean = bootstrap_ci_with_plot(
    y_test, y_pred, matthews_corrcoef, "MCC", n_bootstrap=1000
)

# ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА ВСЕХ ДАННЫХ

print("\n" + "="*70)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА ВСЕХ ДАННЫХ")
print("="*70)

class_weight_dict = {}
for i, cls in enumerate(label_encoder.classes_):
    if cls == 'ЖЭ':
        class_weight_dict[i] = 3.0
    elif cls == 'Норма':
        class_weight_dict[i] = 1.5
    else:
        class_weight_dict[i] = 0.5

print("Обучение XGBoost...")
xgb_final = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric='mlogloss', use_label_encoder=False, verbosity=0,
    tree_method='hist', deterministic_histogram=True
)
xgb_final.fit(X, y, verbose=False)

print("Обучение Random Forest...")
rf_final = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt', random_state=42,
    class_weight=class_weight_dict
)
rf_final.fit(X, y)

print("Обучение Gradient Boosting...")
gb_final = GradientBoostingClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    random_state=42, subsample=0.8
)
gb_final.fit(X, y)

ensemble_final = VotingClassifier(
    estimators=[('xgb', xgb_final), ('rf', rf_final), ('gb', gb_final)],
    voting='soft', weights=[3, 2, 1]
)
ensemble_final.fit(X, y)

final_model = CalibratedClassifierCV(ensemble_final, method='isotonic', cv=3)
final_model.fit(X, y)
print("Финальная модель обучена (Isotonic calibration)")

#  ПОДРОБНЫЙ ОТЧЕТ ПО МЕТРИКАМ КЛАССОВ

print("\n" + "="*70)
print("ПОДРОБНЫЙ ОТЧЕТ ПО КЛАССАМ (СРЕДНЕЕ ПО 5 FOLDS)")
print("="*70)

class_metrics = {}
for cls in label_encoder.classes_:
    class_metrics[cls] = {
        'precision': [],
        'recall': [],
        'f1-score': [],
        'specificity': []   # Добавлено
    }

for fold_idx, report in enumerate(cv_results['reports']):
    for cls in label_encoder.classes_:
        if cls in report:
            class_metrics[cls]['precision'].append(report[cls]['precision'])
            class_metrics[cls]['recall'].append(report[cls]['recall'])
            class_metrics[cls]['f1-score'].append(report[cls]['f1-score'])
            # Добавляем specificity для этого фолда
            class_metrics[cls]['specificity'].append(cv_results['specificity'][fold_idx][cls])

print("\nИТОГОВЫЕ МЕТРИКИ ПО КЛАССАМ (среднее по 5 фолдам ± std):")
print("-" * 70)
print(f"{'Класс':<12} {'Precision':<18} {'Recall':<18} {'F1-score':<18} {'Specificity':<18}")
print("-" * 70)

for cls in label_encoder.classes_:
    p_mean = np.mean(class_metrics[cls]['precision']) * 100
    p_std = np.std(class_metrics[cls]['precision']) * 100
    r_mean = np.mean(class_metrics[cls]['recall']) * 100
    r_std = np.std(class_metrics[cls]['recall']) * 100
    f_mean = np.mean(class_metrics[cls]['f1-score']) * 100
    f_std = np.std(class_metrics[cls]['f1-score']) * 100
    s_mean = np.mean(class_metrics[cls]['specificity']) * 100
    s_std = np.std(class_metrics[cls]['specificity']) * 100

    print(f"{cls:<12} {p_mean:.1f}% (±{p_std:.1f}%)  {r_mean:.1f}% (±{r_std:.1f}%)  {f_mean:.1f}% (±{f_std:.1f}%)  {s_mean:.1f}% (±{s_std:.1f}%)")

print("-" * 70)

# СОХРАНЕНИЕ МОДЕЛИ

print("\n" + "="*70)
print("СОХРАНЕНИЕ МОДЕЛИ")
print("="*70)

joblib.dump(final_model, MODEL_DIR / "fixed_ecg_model_v3.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler_v3.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder_v3.pkl")

with open(MODEL_DIR / "feature_cols_v3.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

with open(MODEL_DIR / "selected_features_v3.pkl", "wb") as f:
    pickle.dump(selected_features, f)

with open(MODEL_DIR / "cv_results_v3.pkl", "wb") as f:
    pickle.dump(cv_results, f)

print(f"Модель сохранена в {MODEL_DIR}")

#  PRODUCTION ФУНКЦИЯ

print("\n" + "="*70)
print("PRODUCTION ФУНКЦИЯ")
print("="*70)

def predict_ecg(signal, sample_rate, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Функция для предсказания класса аритмии по ЭКГ-сигналу.
    Аргументы:
        signal - массив с ЭКГ-сигналом
        sample_rate - частота дискретизации сигнала
        confidence_threshold - порог уверенности для отказа от ответа
    Возвращает словарь с предсказанным классом, уверенностью и вероятностями.
    """
    required_length = TARGET_SR * SEGMENT_DURATION
    if len(signal) < required_length:
        signal = np.pad(signal, (0, required_length - len(signal)))
    else:
        signal = signal[:required_length]

    features = extract_all_features(signal, sample_rate)
    feature_df = pd.DataFrame([features])[feature_cols].fillna(0)
    feature_scaled = scaler.transform(feature_df)

    probs = final_model.predict_proba(feature_scaled)[0]
    max_conf = np.max(probs)

    if max_conf < confidence_threshold:
        return {
            'class': 'UNKNOWN',
            'confidence': max_conf * 100,
            'probabilities': {str(cls): probs[i]*100 for i, cls in enumerate(label_encoder.classes_)},
            'rejected': True,
            'message': f'Low confidence ({max_conf*100:.1f}% < {confidence_threshold*100:.0f}%)'
        }

    pred = np.argmax(probs)
    pred_class = label_encoder.inverse_transform([pred])[0]
    confidence = probs[pred] * 100

    return {
        'class': str(pred_class),
        'confidence': confidence,
        'probabilities': {str(cls): probs[i]*100 for i, cls in enumerate(label_encoder.classes_)},
        'rejected': False,
        'message': None
    }

print("Функция predict_ecg() готова")

# ═══════════════════════════════════════════════════════════════════════════════
# 21. ИТОГОВЫЙ ОТЧЕТ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ИТОГОВЫЙ ОТЧЕТ")
print("="*70)

# Пересчёт средних specificity для итогового отчёта
spec_means = {}
spec_stds = {}
for cls in label_encoder.classes_:
    spec_values = [d[cls] for d in cv_results['specificity']]
    spec_means[cls] = np.mean(spec_values) * 100
    spec_stds[cls] = np.std(spec_values) * 100

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ФИНАЛЬНАЯ МОДЕЛЬ - ИТОГОВЫЕ РЕЗУЛЬТАТЫ                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Реализованные компоненты:                                                    ║
║     1. Permutation importance для отбора признаков                            ║
║     2. GroupKFold CV (5 folds)                                                ║
║     3. Isotonic calibration                                                   ║
║     4. Baseline сравнение с DummyClassifier                                   ║
║     5. Бутстрап с доверительными интервалами                                  ║
║                                                                               ║
║  РЕЗУЛЬТАТЫ GROUPKFOLD CV (5 FOLDS):                                          ║
║     • Accuracy:  {np.mean(cv_results['accuracy'])*100:.2f}% (±{np.std(cv_results['accuracy'])*100:.2f}%)     ║
║     • Weighted F1: {np.mean(cv_results['weighted_f1'])*100:.2f}% (±{np.std(cv_results['weighted_f1'])*100:.2f}%)   ║
║     • MCC: {np.mean(cv_results['mcc']):.3f} (±{np.std(cv_results['mcc']):.3f})                          ║
║     • Kappa: {np.mean(cv_results['kappa']):.3f} (±{np.std(cv_results['kappa']):.3f})                    ║
║                                                                               ║
║  BASELINE (лучший - most_frequent): 66.33%                                    ║
║  УЛУЧШЕНИЕ: +{(np.mean(cv_results['accuracy'])*100 - 66.33):.2f}%             ║
║                                                                               ║
║  БУТСТРАП (95% ДИ) на последнем фолде:                                        ║
║     • Accuracy: [{acc_lower*100:.2f}%, {acc_upper*100:.2f}%]                  ║
║     • Weighted F1: [{f1_lower*100:.2f}%, {f1_upper*100:.2f}%]                 ║
║     • MCC: [{mcc_lower:.3f}, {mcc_upper:.3f}]                                 ║
║                                                                               ║
║  МЕТРИКИ ПО КЛАССАМ (среднее по 5 фолдам):                                    ║
║     • ЖЭ: Precision: {np.mean(class_metrics['ЖЭ']['precision'])*100:.1f}%, Recall: {np.mean(class_metrics['ЖЭ']['recall'])*100:.1f}%, F1: {np.mean(class_metrics['ЖЭ']['f1-score'])*100:.1f}%, Specificity: {spec_means['ЖЭ']:.1f}%   ║
║     • НЭ: Precision: {np.mean(class_metrics['НЭ']['precision'])*100:.1f}%, Recall: {np.mean(class_metrics['НЭ']['recall'])*100:.1f}%, F1: {np.mean(class_metrics['НЭ']['f1-score'])*100:.1f}%, Specificity: {spec_means['НЭ']:.1f}%   ║
║     • Норма: Precision: {np.mean(class_metrics['Норма']['precision'])*100:.1f}%, Recall: {np.mean(class_metrics['Норма']['recall'])*100:.1f}%, F1: {np.mean(class_metrics['Норма']['f1-score'])*100:.1f}%, Specificity: {spec_means['Норма']:.1f}%   ║
║                                                                               ║
║  СОХРАНЕННЫЕ ФАЙЛЫ:                                                           ║
║     • Модель: {MODEL_DIR / 'fixed_ecg_model_v3.pkl'}                          ║
║     • Scaler: {MODEL_DIR / 'scaler_v3.pkl'}                                   ║
║     • Label Encoder: {MODEL_DIR / 'label_encoder_v3.pkl'}                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

print("\nФИНАЛЬНАЯ МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ")
