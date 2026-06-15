# -*- coding: utf-8 -*-
"""
ВЕБ-ПРИЛОЖЕНИЕ ДЛЯ КЛАССИФИКАЦИИ АРИТМИЙ ПО ЭКГ
Поддерживает: загрузку CSV/TXT с сигналом ИЛИ загрузку изображения ЭКГ
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
from pathlib import Path
import neurokit2 as nk
from scipy.signal import find_peaks, resample
from scipy.stats import skew
import cv2
from PIL import Image
import io

# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="ЭКГ Аритмия | AI Диагностика",
    page_icon="❤️",
    layout="wide"
)

# ============================================================
# ЗАГРУЗКА МОДЕЛИ (из локальной папки model/)
# ============================================================
@st.cache_resource
def load_model():
    """Загружает модель и все компоненты из папки model/"""
    model_dir = Path(__file__).parent / "model"
    
    required_files = [
        "fixed_ecg_model_v3.pkl",
        "scaler_v3.pkl",
        "label_encoder_v3.pkl",
        "feature_cols_v3.pkl"
    ]
    
    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        st.error(f"❌ Не найдены файлы: {missing}")
        return None, None, None, None
    
    model = joblib.load(model_dir / "fixed_ecg_model_v3.pkl")
    scaler = joblib.load(model_dir / "scaler_v3.pkl")
    label_encoder = joblib.load(model_dir / "label_encoder_v3.pkl")
    with open(model_dir / "feature_cols_v3.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    
    return model, scaler, label_encoder, feature_cols

# ============================================================
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (из вашего кода)
# ============================================================
def normalize_segment(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        std = 1.0
    return (signal - mean) / std

def extract_hrv_frequency_features(rr_intervals, sample_rate=4):
    try:
        if len(rr_intervals) < 5:
            return 0.5, 0.5, 1.0
        time_rr = np.cumsum(rr_intervals)
        time_uniform = np.arange(0, time_rr[-1], 1/sample_rate)
        rr_interpolated = np.interp(time_uniform, time_rr, 
                                    rr_intervals[:-1] if len(rr_intervals) > len(time_uniform) else rr_intervals)
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

def extract_all_features(signal, sample_rate):
    """Извлекает 25 признаков из ЭКГ-сигнала (ваша оригинальная функция)"""
    features = {}
    try:
        if len(signal) < sample_rate:
            signal = np.pad(signal, (0, sample_rate - len(signal)))
        
        signal = normalize_segment(signal)
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
                features['pNN50'] = float((nn50 / len(rr_intervals)) * 100)
                
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
                
                # Ширина QRS
                qrs_widths = []
                for peak in r_peaks[:5]:
                    if peak - int(0.05*sample_rate) < 0 or peak + int(0.08*sample_rate) >= len(cleaned):
                        continue
                    beat = cleaned[peak - int(0.05*sample_rate):peak + int(0.08*sample_rate)]
                    if len(beat) > 0:
                        half_max = (np.max(beat) + np.min(beat)) / 2
                        above_half = np.where(beat > half_max)[0]
                        if len(above_half) > 1:
                            qrs_widths.append((above_half[-1] - above_half[0]) / sample_rate * 1000)
                features['QRS_width'] = np.mean(qrs_widths) if qrs_widths else 80.0
                
                # Энтропия
                try:
                    hist, _ = np.histogram(signal, bins=50)
                    hist = hist / (len(signal) + 1e-6)
                    features['signal_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-6)))
                except:
                    features['signal_entropy'] = 2.5
            else:
                return _default_features()
        else:
            return _default_features()
        
        features['Signal_mean'] = float(np.mean(signal))
        features['Signal_std'] = float(np.std(signal))
        features['Signal_rms'] = float(np.sqrt(np.mean(signal**2)))
        features['zero_crossing_rate'] = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))
        features['T_wave_asymmetry'] = 0.0
        
    except Exception as e:
        return _default_features()
    
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0.0
    return features

def _default_features():
    return {
        'HeartRate': 75.0, 'Mean_RR': 0.8, 'Std_RR': 0.05, 'Min_RR': 0.7, 'Max_RR': 0.9,
        'RR_range': 0.2, 'RR_cv': 0.0625, 'pNN50': 10.0,
        'SD1': 0.03, 'SD2': 0.07, 'SD1_SD2_ratio': 0.43,
        'LF_power': 0.5, 'HF_power': 0.5, 'LF_HF_ratio': 1.0, 'Total_power': 1.0,
        'LF_norm': 0.5, 'HF_norm': 0.5, 'QRS_width': 80.0, 'signal_entropy': 2.5,
        'Signal_mean': 0.0, 'Signal_std': 0.8, 'Signal_rms': 0.7,
        'zero_crossing_rate': 0.05, 'T_wave_asymmetry': 0.0
    }

# ============================================================
# ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЯ В СИГНАЛ
# ============================================================
def image_to_ecg_signal(image, target_length=3600):
    """
    Преобразует изображение ЭКГ-ленты в числовой сигнал.
    Извлекает линию ЭКГ из изображения и возвращает массив значений.
    """
    # Конвертируем в numpy array
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = image
    
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Улучшаем контраст
    gray = cv2.equalizeHist(gray)
    
    # Бинаризация (инвертируем, чтобы линия была белой на чёрном)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Находим координаты белых пикселей (линия ЭКГ)
    y_coords, x_coords = np.where(binary > 0)
    
    if len(x_coords) == 0:
        # Если не нашли линию, используем среднее значение по столбцам
        signal = np.mean(gray, axis=0)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    else:
        # Для каждого x берём средний y (линия может быть толстой)
        signal = []
        for x in range(gray.shape[1]):
            y_vals = y_coords[x_coords == x]
            if len(y_vals) > 0:
                # Нормализуем y (инвертируем, т.к. в изображении y растёт вниз)
                y_norm = (np.mean(y_vals) - gray.shape[0]/2) / (gray.shape[0]/2)
                signal.append(y_norm)
            else:
                signal.append(0)
        signal = np.array(signal)
    
    # Изменяем размер до target_length
    if len(signal) != target_length:
        signal = resample(signal, target_length)
    
    # Нормализуем сигнал
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    
    return signal

# ============================================================
# ПРЕДСКАЗАНИЕ
# ============================================================
def predict_from_signal(signal, sample_rate, model, scaler, label_encoder, feature_cols, confidence_threshold=0.65):
    """Предсказание по сигналу (ваша оригинальная функция)"""
    required_length = 360 * 10  # 10 секунд при 360 Гц
    
    if len(signal) < required_length:
        signal = np.pad(signal, (0, required_length - len(signal)))
    else:
        signal = signal[:required_length]
    
    features = extract_all_features(signal, sample_rate)
    feature_df = pd.DataFrame([features])[feature_cols].fillna(0)
    feature_scaled = scaler.transform(feature_df)
    
    probs = model.predict_proba(feature_scaled)[0]
    max_conf = np.max(probs)
    
    if max_conf < confidence_threshold:
        return {
            'class': 'UNKNOWN',
            'confidence': max_conf * 100,
            'probabilities': {cls: probs[i]*100 for i, cls in enumerate(label_encoder.classes_)},
            'rejected': True
        }
    
    pred = np.argmax(probs)
    pred_class = label_encoder.inverse_transform([pred])[0]
    confidence = probs[pred] * 100
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'probabilities': {cls: probs[i]*100 for i, cls in enumerate(label_encoder.classes_)},
        'rejected': False
    }

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================
model, scaler, label_encoder, feature_cols = load_model()
if model is None:
    st.stop()

feature_cols_list = feature_cols if isinstance(feature_cols, list) else list(feature_cols)
st.success(f"✅ Модель загружена! Ожидается {len(feature_cols_list)} признаков.")

# ============================================================
# ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ============================================================
st.title("❤️ ECG Clinical AI V9.7")
st.markdown("**Классификация аритмий по ЭКГ** — загрузите изображение ЭКГ или файл с сигналом.")

# Выбор способа ввода
input_method = st.radio(
    "Способ ввода:",
    ["📷 Загрузить изображение ЭКГ", "📊 Загрузить CSV/TXT с сигналом"],
    horizontal=True
)

# Настройки
with st.sidebar:
    st.header("⚙️ Настройки")
    confidence_threshold = st.slider("Порог уверенности", min_value=0.5, max_value=0.95, value=0.65, step=0.05)
    st.markdown("---")
    st.markdown("### 📋 Классы:")
    st.markdown("- **Норма** — нормальный синусовый ритм")
    st.markdown("- **НЭ** — наджелудочковая экстрасистолия")
    st.markdown("- **ЖЭ** — желудочковая экстрасистолия (опасная!)")
    st.markdown("---")
    st.markdown("### 🔬 О модели")
    st.markdown(f"- Признаков: {len(feature_cols_list)}")
    st.markdown("- Алгоритм: ансамбль XGBoost + RF + GB")

# ============================================================
# РЕЖИМ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ============================================================
if input_method == "📷 Загрузить изображение ЭКГ":
    uploaded_file = st.file_uploader(
        "Выберите изображение ЭКГ (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Загрузите чёткое фото или скан ЭКГ-ленты"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Загруженное изображение ЭКГ", use_container_width=True)
        with col2:
            sample_rate = st.number_input("Частота дискретизации (Гц)", min_value=100, max_value=1000, value=360, key="img_sr")
            st.caption("Для изображений обычно используется 360 Гц")
        
        if st.button("🔍 Анализировать ЭКГ", type="primary"):
            with st.spinner("🩻 Извлечение сигнала из изображения..."):
                try:
                    # Извлекаем сигнал из изображения
                    signal = image_to_ecg_signal(image, target_length=3600)
                    
                    # Показываем извлечённый сигнал
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(signal[:1800], color='#1a1a2e', linewidth=0.8)
                    ax.set_xlabel("Отсчёты")
                    ax.set_ylabel("Амплитуда")
                    ax.set_title("Извлечённый ЭКГ-сигнал (первые 5 секунд)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    with st.spinner("🧠 Анализ сигнала..."):
                        result = predict_from_signal(
                            signal, sample_rate, model, scaler, 
                            label_encoder, feature_cols_list, confidence_threshold
                        )
                    
                    st.markdown("---")
                    st.subheader("📈 Результат классификации")
                    
                    if result['rejected']:
                        st.warning(f"⚠️ Низкая уверенность: {result['confidence']:.1f}%")
                        st.info("Попробуйте загрузить более чёткое изображение")
                    else:
                        if result['class'] == 'ЖЭ':
                            st.error(f"### 🚨 {result['class']}")
                            st.warning("⚠️ **Требуется консультация кардиолога!**")
                        elif result['class'] == 'НЭ':
                            st.warning(f"### ⚠️ {result['class']}")
                            st.info("Рекомендуется наблюдение")
                        else:
                            st.success(f"### ✅ {result['class']}")
                        
                        st.write(f"**Уверенность:** {result['confidence']:.2f}%")
                        
                        st.markdown("#### Вероятности по классам:")
                        for cls, prob in result['probabilities'].items():
                            st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                    
                except Exception as e:
                    st.error(f"Ошибка при анализе: {e}")
                    st.info("Попробуйте загрузить изображение с более чёткой линией ЭКГ")

# ============================================================
# РЕЖИМ 2: ЗАГРУЗКА CSV/TXT (оригинальный функционал)
# ============================================================
else:
    uploaded_file = st.file_uploader(
        "Выберите файл с ЭКГ-сигналом", 
        type=["csv", "txt"],
        help="Файл должен содержать числовые значения сигнала (одно значение на строку или через запятую)"
    )
    
    col1, col2 = st.columns([2, 1])
    with col2:
        sample_rate = st.number_input("Частота дискретизации (Гц)", min_value=100, max_value=1000, value=360, key="csv_sr")
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode()
            numbers = []
            for line in content.replace(',', '\n').split('\n'):
                line = line.strip()
                if line and (line.replace('.', '').replace('-', '').isdigit() or 
                            (line.startswith('-') and line[1:].replace('.', '').replace('-', '').isdigit())):
                    numbers.append(float(line))
            
            signal = np.array(numbers)
            st.success(f"✅ Загружено {len(signal)} отсчётов (~{len(signal)/sample_rate:.1f} секунд)")
            
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(signal[:min(len(signal), 3600)], color='#1a1a2e', linewidth=0.8)
            ax.set_xlabel("Отсчёты")
            ax.set_ylabel("Амплитуда")
            ax.set_title("Загруженный ЭКГ-сигнал")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            if st.button("🔍 Анализировать ЭКГ", type="primary"):
                with st.spinner("🧠 Анализ сигнала..."):
                    result = predict_from_signal(
                        signal, sample_rate, model, scaler,
                        label_encoder, feature_cols_list, confidence_threshold
                    )
                
                st.markdown("---")
                st.subheader("📈 Результат классификации")
                
                if result['rejected']:
                    st.warning(f"⚠️ Низкая уверенность: {result['confidence']:.1f}%")
                else:
                    if result['class'] == 'ЖЭ':
                        st.error(f"### 🚨 {result['class']}")
                        st.warning("⚠️ **Требуется консультация кардиолога!**")
                    elif result['class'] == 'НЭ':
                        st.warning(f"### ⚠️ {result['class']}")
                    else:
                        st.success(f"### ✅ {result['class']}")
                    
                    st.write(f"**Уверенность:** {result['confidence']:.2f}%")
                    
                    st.markdown("#### Вероятности по классам:")
                    for cls, prob in result['probabilities'].items():
                        st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                    
        except Exception as e:
            st.error(f"Ошибка: {e}")

st.markdown("---")
st.caption("🔬 Инструмент предназначен для исследовательских целей. Окончательный диагноз ставит врач.")
