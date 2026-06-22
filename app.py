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

# КОНСТАНТЫ
ALL_FEATURES_25 = [
    'HeartRate', 'Mean_RR', 'Std_RR', 'Min_RR', 'Max_RR',
    'RR_range', 'RR_cv', 'pNN50', 'pNN20',
    'SD1', 'SD2', 'SD1_SD2_ratio',
    'LF_power', 'HF_power', 'LF_HF_ratio', 'Total_power',
    'LF_norm', 'HF_norm',
    'Signal_mean', 'Signal_std', 'Signal_rms',
    'zero_crossing_rate', 'signal_entropy',
    'QRS_width', 'T_wave_asymmetry'
]

EXPECTED_CLASSES = {
    '100': 'Норма', '101': 'Норма', '103': 'Норма', '105': 'Норма',
    '109': 'Норма', '111': 'Норма', '112': 'Норма', '113': 'Норма',
    '114': 'Норма', '115': 'Норма', '116': 'Норма', '117': 'Норма',
    '118': 'Норма', '119': 'ЖЭ', '121': 'Норма', '122': 'Норма',
    '123': 'Норма', '124': 'Норма', '200': 'ЖЭ', '201': 'ЖЭ',
    '202': 'ЖЭ', '203': 'ЖЭ', '205': 'ЖЭ', '207': 'ЖЭ',
    '208': 'ЖЭ', '209': 'НЭ', '210': 'ЖЭ', '212': 'ЖЭ',
    '213': 'ЖЭ', '214': 'ЖЭ', '215': 'ЖЭ', '217': 'ЖЭ',
    '219': 'ЖЭ', '220': 'НЭ', '221': 'ЖЭ', '222': 'ЖЭ',
    '223': 'ЖЭ', '228': 'ЖЭ', '230': 'ЖЭ', '231': 'ЖЭ',
    '232': 'ЖЭ', '233': 'ЖЭ', '234': 'ЖЭ',
}

# НАСТРОЙКА СТРАНИЦЫ
st.set_page_config(
    page_title="ЭКГ Аритмия | AI Диагностика",
    page_icon="❤️",
    layout="wide"
)

# ЗАГРУЗКА МОДЕЛИ

@st.cache_resource
def load_model():
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

# ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (ИДЕНТИЧНЫЕ ОБУЧАЮЩЕМУ СКРИПТУ)

def extract_hrv_frequency_features(rr_intervals, sample_rate=4):
    """Частотные HRV-признаки (идентично обучению)"""
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

def extract_qrs_width_and_t_wave(signal, sample_rate, r_peaks):
    """QRS ширина и асимметрия T-зубца (идентично обучению)"""
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

def _default_features():
    """Запасные значения при ошибке (идентично обучению)"""
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

def extract_all_features(signal, sample_rate):
    """Извлечение признаков (идентично обучению)"""
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

# ФУНКЦИЯ ПРЕДСКАЗАНИЯ (БЕЗ ХАРДКОД-ПРАВИЛ)
def predict_from_signal(signal, sample_rate, model, scaler, label_encoder, feature_cols, confidence_threshold=0.55):
    required_length = 360 * 10
    if len(signal) < required_length:
        signal = np.pad(signal, (0, required_length - len(signal)))
    else:
        signal = signal[:required_length]
    
    all_features = extract_all_features(signal, sample_rate)
    
    full_feature_dict = {}
    for key in ALL_FEATURES_25:
        full_feature_dict[key] = all_features.get(key, 0.0)
    
    feature_df_full = pd.DataFrame([full_feature_dict])[ALL_FEATURES_25].fillna(0)
    feature_scaled_full = scaler.transform(feature_df_full)
    
    selected_indices = [ALL_FEATURES_25.index(f) for f in feature_cols]
    feature_scaled_selected = feature_scaled_full[:, selected_indices]
    
    probs = model.predict_proba(feature_scaled_selected)[0]
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

# ЗАГРУЗКА МОДЕЛИ
model, scaler, label_encoder, feature_cols = load_model()

if model is None:
    st.stop()

feature_cols_list = list(feature_cols) if not isinstance(feature_cols, list) else feature_cols

# ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
st.title("❤️ ECG Clinical AI")
st.markdown("**Классификация аритмий по ЭКГ**")

with st.sidebar:
    st.header("⚙️ Настройки")
    confidence_threshold = st.slider("Порог уверенности", min_value=0.3, max_value=0.95, value=0.65, step=0.05)
    st.markdown("---")
    st.markdown("### 📋 Классы:")
    st.markdown("- **Норма** — нормальный синусовый ритм")
    st.markdown("- **НЭ** — наджелудочковая экстрасистолия")
    st.markdown("- **ЖЭ** — желудочковая экстрасистолия (опасная!)")
    st.markdown("---")
    st.markdown("### 🔬 О модели")
    st.markdown(f"- Модель: {len(feature_cols_list)} признаков")
    st.markdown("- Алгоритм: ансамбль XGBoost + RF + GB")
    st.markdown("---")
    st.markdown("### 📌 MIT-BIH записи")
    st.markdown("- **100, 101, 103** — Норма")
    st.markdown("- **119, 200, 201** — ЖЭ")
    st.markdown("- **209, 220** — НЭ")

# Выбор способа ввода
input_method = st.radio(
    "Способ ввода:",
    ["📡 Загрузить запись MIT-BIH (wfdb)", "📁 Загрузить .dat файл", "📊 Загрузить CSV/TXT", "📷 Загрузить изображение"],
    horizontal=True
)

# РЕЖИМ 1: WFDB

if input_method == "📡 Загрузить запись MIT-BIH (wfdb)":
    st.markdown("""
    **Загрузите запись из MIT-BIH Arrhythmia Database**
    
    Приложение автоматически скачает запись через интернет.
    """)
    
    record_options = [
        "100", "101", "103", "105", "109", "111", "112", "113", "114", "115",
        "116", "117", "118", "119", "121", "122", "123", "124",
        "200", "201", "202", "203", "205", "207", "208", "209", "210",
        "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234"
    ]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        record_name = st.selectbox("Выберите запись", record_options, index=0, key="wfdb_record_select")
        expected_class = EXPECTED_CLASSES.get(record_name, "Неизвестно")
        st.caption(f"Ожидаемый класс: **{expected_class}**")
    with col2:
        sample_rate = st.number_input("Частота дискретизации (Гц)", value=360, min_value=100, max_value=1000, key="wfdb_sr_input")
    
    if st.button("📥 Загрузить запись", type="primary", key="wfdb_load_btn"):
        with st.spinner(f"Загрузка записи {record_name}..."):
            try:
                import wfdb
                record = wfdb.rdrecord(record_name, pn_dir='mitdb')
                signal = record.p_signal[:, 0][:sample_rate * 10]
                if len(signal) > 0:
                    # Сигнал НЕ нормализуется здесь, только если исходные значения слишком большие
                    if np.max(np.abs(signal)) > 100:
                        mean_val = np.mean(signal)
                        std_val = np.std(signal)
                        if std_val > 1e-8:
                            signal = (signal - mean_val) / std_val
                st.session_state['wfdb_signal_data'] = signal
                st.session_state['wfdb_record_name'] = record_name
                st.session_state['wfdb_expected_class'] = expected_class
                st.session_state['wfdb_sample_rate'] = sample_rate
                st.session_state['wfdb_loaded'] = True
                st.success(f"✅ Загружено {len(signal)} отсчётов (10 секунд)")
            except Exception as e:
                st.error(f"❌ Не удалось загрузить запись {record_name}: {e}")
    
    if st.session_state.get('wfdb_loaded', False):
        signal = st.session_state['wfdb_signal_data']
        record_name = st.session_state['wfdb_record_name']
        expected_class = st.session_state['wfdb_expected_class']
        sample_rate = st.session_state['wfdb_sample_rate']
        
        with st.expander("🔍 Диагностика сигнала", expanded=True):
            cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
            r_peaks, _ = find_peaks(cleaned, distance=sample_rate*0.3, height=np.std(cleaned)*0.4)
            if len(r_peaks) >= 3:
                rr_intervals = np.diff(r_peaks) / sample_rate
                rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) >= 3 else 0
            else:
                hr = 0
            
            st.write(f"**📊 Статистика сигнала:**")
            st.write(f"- Длина: {len(signal)} отсчётов")
            st.write(f"- Среднее: {np.mean(signal):.4f}")
            st.write(f"- Стандартное отклонение: {np.std(signal):.4f}")
            st.write(f"- Найдено R-пиков: {len(r_peaks)}")
            st.write(f"- ЧСС: {hr:.1f} уд/мин")
            
            if len(r_peaks) < 3:
                st.warning("⚠️ Найдено меньше 3 R-пиков! Сигнал не похож на ЭКГ.")
            elif hr < 30:
                st.warning(f"⚠️ ЧСС слишком низкая ({hr:.1f} уд/мин)")
            elif hr > 200:
                st.warning(f"⚠️ ЧСС слишком высокая ({hr:.1f} уд/мин)")
            else:
                st.success(f"✅ Сигнал выглядит как ЭКГ (ЧСС: {hr:.1f} уд/мин)")
        
        with st.expander("📊 Извлечённые признаки", expanded=False):
            features = extract_all_features(signal, sample_rate)
            feature_data = []
            for key in ALL_FEATURES_25:
                value = features.get(key, 0.0)
                feature_data.append({"Признак": key, "Значение": f"{value:.4f}"})
            st.table(pd.DataFrame(feature_data))
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(signal, color='#1a1a2e', linewidth=0.8)
        ax.set_xlabel("Отсчёты")
        ax.set_ylabel("Амплитуда (нормализованная)")
        ax.set_title(f"ЭКГ-сигнал: запись {record_name}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        if st.button("🔍 Анализировать ЭКГ", type="primary", key="wfdb_analyze_btn"):
            with st.spinner("🧠 Анализ..."):
                result = predict_from_signal(
                    signal, sample_rate, model, scaler,
                    label_encoder, feature_cols_list, confidence_threshold
                )
            
            st.markdown("---")
            st.subheader("📈 Результат классификации")
            
            if result['rejected']:
                st.warning(f"⚠️ Низкая уверенность: {result['confidence']:.1f}%")
            else:
                if expected_class != "Неизвестно":
                    if result['class'] == expected_class:
                        st.success(f"✅ Совпадает с ожидаемым классом: **{expected_class}**")
                    else:
                        st.warning(f"⚠️ Отличается от ожидаемого класса: **{expected_class}** → предсказано **{result['class']}**")
                
                if result['class'] == 'ЖЭ':
                    st.error(f"### 🚨 {result['class']}")
                    st.warning("⚠️ **Требуется консультация кардиолога!**")
                elif result['class'] == 'НЭ':
                    st.warning(f"### ⚠️ {result['class']}")
                else:
                    st.success(f"### ✅ {result['class']}")
                
                st.write(f"**Уверенность:** {result['confidence']:.2f}%")
                
                for cls, prob in result['probabilities'].items():
                    st.progress(prob/100, text=f"{cls}: {prob:.1f}%")


# РЕЖИМ 2: .DAT
elif input_method == "📁 Загрузить .dat файл":
    st.markdown("""
    **Загрузите .dat файл из MIT-BIH** (формат 212)
    """)
    
    uploaded_file = st.file_uploader("Выберите .dat файл", type=["dat"])
    
    col1, col2 = st.columns([2, 1])
    with col2:
        sample_rate = st.number_input("Частота дискретизации (Гц)", value=360, min_value=100, max_value=1000, key="dat_sr")
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        try:
            data = np.frombuffer(file_content, dtype=np.uint8)
            signal = []
            i = 0
            samples_collected = 0
            num_samples = 3600
            while i + 2 < len(data) and samples_collected < num_samples:
                sample1 = data[i] | ((data[i+1] & 0x0F) << 8)
                sample2 = ((data[i+1] >> 4) & 0x0F) | (data[i+2] << 4)
                if sample1 >= 2048:
                    sample1 = sample1 - 4096
                if sample2 >= 2048:
                    sample2 = sample2 - 4096
                signal.append(sample1 / 200.0)
                samples_collected += 1
                i += 3
            signal = np.array(signal[:num_samples])
            if len(signal) > 0:
                if np.max(np.abs(signal)) > 100:
                    mean_val = np.mean(signal)
                    std_val = np.std(signal)
                    if std_val > 1e-8:
                        signal = (signal - mean_val) / std_val
        except Exception as e:
            signal = None
        
        if signal is None or len(signal) < 100:
            st.error("❌ Не удалось прочитать .dat файл")
        else:
            st.success(f"✅ Загружено {len(signal)} отсчётов (10 секунд)")
            
            with st.expander("🔍 Диагностика сигнала", expanded=True):
                cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
                r_peaks, _ = find_peaks(cleaned, distance=sample_rate*0.3, height=np.std(cleaned)*0.4)
                if len(r_peaks) >= 3:
                    rr_intervals = np.diff(r_peaks) / sample_rate
                    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                    hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) >= 3 else 0
                else:
                    hr = 0
                
                st.write(f"**📊 Статистика сигнала:**")
                st.write(f"- Длина: {len(signal)} отсчётов")
                st.write(f"- Среднее: {np.mean(signal):.4f}")
                st.write(f"- Стандартное отклонение: {np.std(signal):.4f}")
                st.write(f"- Найдено R-пиков: {len(r_peaks)}")
                st.write(f"- ЧСС: {hr:.1f} уд/мин")
                
                if len(r_peaks) < 3:
                    st.warning("⚠️ Найдено меньше 3 R-пиков! Сигнал не похож на ЭКГ.")
                elif hr < 30:
                    st.warning(f"⚠️ ЧСС слишком низкая ({hr:.1f} уд/мин)")
                elif hr > 200:
                    st.warning(f"⚠️ ЧСС слишком высокая ({hr:.1f} уд/мин)")
                else:
                    st.success(f"✅ Сигнал выглядит как ЭКГ (ЧСС: {hr:.1f} уд/мин)")
            
            with st.expander("📊 Извлечённые признаки", expanded=False):
                features = extract_all_features(signal, sample_rate)
                feature_data = []
                for key in ALL_FEATURES_25:
                    value = features.get(key, 0.0)
                    feature_data.append({"Признак": key, "Значение": f"{value:.4f}"})
                st.table(pd.DataFrame(feature_data))
            
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(signal, color='#1a1a2e', linewidth=0.8)
            ax.set_xlabel("Отсчёты")
            ax.set_ylabel("Амплитуда (нормализованная)")
            ax.set_title("ЭКГ-сигнал из .dat")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            if st.button("🔍 Анализировать ЭКГ", type="primary", key="dat_analyze"):
                with st.spinner("🧠 Анализ..."):
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
                    
                    for cls, prob in result['probabilities'].items():
                        st.progress(prob/100, text=f"{cls}: {prob:.1f}%")


# РЕЖИМ 3: CSV/TXT
elif input_method == "📊 Загрузить CSV/TXT":
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "txt"])
    
    col1, col2 = st.columns([2, 1])
    with col2:
        sample_rate = st.number_input("Частота дискретизации (Гц)", value=360, min_value=100, max_value=1000, key="csv_sr")
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode()
            numbers = []
            for line in content.replace(',', '\n').split('\n'):
                line = line.strip()
                if line:
                    try:
                        numbers.append(float(line))
                    except ValueError:
                        pass
            signal = np.array(numbers) if numbers else None
            
            if signal is None or len(signal) == 0:
                st.error("❌ Не удалось прочитать файл")
            else:
                st.success(f"✅ Загружено {len(signal)} отсчётов (~{len(signal)/sample_rate:.1f} сек)")
                
                with st.expander("🔍 Диагностика сигнала", expanded=True):
                    cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
                    r_peaks, _ = find_peaks(cleaned, distance=sample_rate*0.3, height=np.std(cleaned)*0.4)
                    if len(r_peaks) >= 3:
                        rr_intervals = np.diff(r_peaks) / sample_rate
                        rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                        hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) >= 3 else 0
                    else:
                        hr = 0
                    
                    st.write(f"**📊 Статистика сигнала:**")
                    st.write(f"- Длина: {len(signal)} отсчётов")
                    st.write(f"- Среднее: {np.mean(signal):.4f}")
                    st.write(f"- Стандартное отклонение: {np.std(signal):.4f}")
                    st.write(f"- Найдено R-пиков: {len(r_peaks)}")
                    st.write(f"- ЧСС: {hr:.1f} уд/мин")
                    
                    if len(r_peaks) < 3:
                        st.warning("⚠️ Найдено меньше 3 R-пиков! Сигнал не похож на ЭКГ.")
                    elif hr < 30:
                        st.warning(f"⚠️ ЧСС слишком низкая ({hr:.1f} уд/мин)")
                    elif hr > 200:
                        st.warning(f"⚠️ ЧСС слишком высокая ({hr:.1f} уд/мин)")
                    else:
                        st.success(f"✅ Сигнал выглядит как ЭКГ (ЧСС: {hr:.1f} уд/мин)")
                
                with st.expander("📊 Извлечённые признаки", expanded=False):
                    features = extract_all_features(signal, sample_rate)
                    feature_data = []
                    for key in ALL_FEATURES_25:
                        value = features.get(key, 0.0)
                        feature_data.append({"Признак": key, "Значение": f"{value:.4f}"})
                    st.table(pd.DataFrame(feature_data))
                
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(signal[:min(len(signal), 3600)], color='#1a1a2e', linewidth=0.8)
                ax.set_xlabel("Отсчёты")
                ax.set_ylabel("Амплитуда")
                ax.set_title("Загруженный ЭКГ-сигнал")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                if st.button("🔍 Анализировать ЭКГ", type="primary", key="csv_analyze"):
                    with st.spinner("🧠 Анализ..."):
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
                        elif result['class'] == 'НЭ':
                            st.warning(f"### ⚠️ {result['class']}")
                        else:
                            st.success(f"### ✅ {result['class']}")
                        
                        st.write(f"**Уверенность:** {result['confidence']:.2f}%")
                        
                        for cls, prob in result['probabilities'].items():
                            st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                        
        except Exception as e:
            st.error(f"Ошибка: {e}")

# РЕЖИМ 4: ИЗОБРАЖЕНИЕ

else:
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            sample_rate = st.number_input("Частота дискретизации (Гц)", value=360, min_value=100, max_value=1000, key="img_sr")
        
        if st.button("🔍 Анализировать ЭКГ", type="primary", key="img_analyze"):
            with st.spinner("🩻 Извлечение сигнала..."):
                try:
                    img = np.array(image.convert('RGB'))
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    gray = cv2.equalizeHist(gray)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    y_coords, x_coords = np.where(binary > 0)
                    if len(x_coords) == 0:
                        signal = np.mean(gray, axis=0)
                        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
                    else:
                        signal = []
                        for x in range(gray.shape[1]):
                            y_vals = y_coords[x_coords == x]
                            if len(y_vals) > 0:
                                y_norm = (np.mean(y_vals) - gray.shape[0]/2) / (gray.shape[0]/2)
                                signal.append(y_norm)
                            else:
                                signal.append(0)
                        signal = np.array(signal)
                    
                    if len(signal) != 3600:
                        signal = resample(signal, 3600)
                    
                    if np.max(np.abs(signal)) > 100:
                        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
                    
                    with st.expander("🔍 Диагностика сигнала", expanded=True):
                        cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
                        r_peaks, _ = find_peaks(cleaned, distance=sample_rate*0.3, height=np.std(cleaned)*0.4)
                        if len(r_peaks) >= 3:
                            rr_intervals = np.diff(r_peaks) / sample_rate
                            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                            hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) >= 3 else 0
                        else:
                            hr = 0
                        
                        st.write(f"**📊 Статистика сигнала:**")
                        st.write(f"- Длина: {len(signal)} отсчётов")
                        st.write(f"- Среднее: {np.mean(signal):.4f}")
                        st.write(f"- Стандартное отклонение: {np.std(signal):.4f}")
                        st.write(f"- Найдено R-пиков: {len(r_peaks)}")
                        st.write(f"- ЧСС: {hr:.1f} уд/мин")
                        
                        if len(r_peaks) < 3:
                            st.warning("⚠️ Найдено меньше 3 R-пиков! Сигнал не похож на ЭКГ.")
                        elif hr < 30:
                            st.warning(f"⚠️ ЧСС слишком низкая ({hr:.1f} уд/мин)")
                        elif hr > 200:
                            st.warning(f"⚠️ ЧСС слишком высокая ({hr:.1f} уд/мин)")
                        else:
                            st.success(f"✅ Сигнал выглядит как ЭКГ (ЧСС: {hr:.1f} уд/мин)")
                    
                    with st.expander("📊 Извлечённые признаки", expanded=False):
                        features = extract_all_features(signal, sample_rate)
                        feature_data = []
                        for key in ALL_FEATURES_25:
                            value = features.get(key, 0.0)
                            feature_data.append({"Признак": key, "Значение": f"{value:.4f}"})
                        st.table(pd.DataFrame(feature_data))
                    
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(signal[:1800], color='#1a1a2e', linewidth=0.8)
                    ax.set_xlabel("Отсчёты")
                    ax.set_ylabel("Амплитуда")
                    ax.set_title("Извлечённый сигнал")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    with st.spinner("🧠 Анализ..."):
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
                        elif result['class'] == 'НЭ':
                            st.warning(f"### ⚠️ {result['class']}")
                        else:
                            st.success(f"### ✅ {result['class']}")
                        
                        st.write(f"**Уверенность:** {result['confidence']:.2f}%")
                        
                        for cls, prob in result['probabilities'].items():
                            st.progress(prob/100, text=f"{cls}: {prob:.1f}%")
                    
                except Exception as e:
                    st.error(f"Ошибка: {e}")

st.markdown("---")
st.caption(" Инструмент предназначен для исследовательских целей. Диагноз ставит врач.")
