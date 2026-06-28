import streamlit as st
import mne
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

st.set_page_config(
    page_title="EEG-Analyzer",
    page_icon="brain_logo.svg"
)
def create_dynamic(raw, start=0.0, ch_start=0):
    eeg_indices = mne.pick_types(raw.info, meg=False, eeg=True)
    eegs = eeg_indices[ch_start:ch_start+20]
    fig = raw.plot(picks=eegs, 
                   start=start, duration=10.0, show=False)
    st.pyplot(fig)


'''
## Загрузка данных
'''
base = Path(__file__).parent
data_dir = base / "dataset"
#dataset_path = data_dir.join('/dataset')
if os.path.exists(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.vhdr')]
    if files:
        select_file = st.selectbox("Выберите файл из папки", files)
        path = os.path.join(data_dir, select_file)
        if select_file is not None:
            raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
            montage = mne.channels.make_standard_montage("standard_1020") #позиция электрода на голове (координаты).
           
            raw.set_montage(montage, on_missing="ignore") # raw.set_montage(...) “приклеивает” эти координаты к raw
    
            '''
            ## Исходные данные
            '''

            max_duration = raw.times[-1]

            st.sidebar.write('Параметры отображения')
            start = st.sidebar.slider("Время начала отображения", min_value=0.0, 
                              max_value=float(max_duration)-20.0,
                              value=min(10.0, float(max_duration)),
                              step=10.0)
            ch_count = len(raw.ch_names)
            ch_start = st.sidebar.slider("Каналы для отображения", min_value=0, 
                                         max_value=ch_count-20, value=0, step=1)
            st.sidebar.divider()
            st.sidebar.write('Параметры предобработки')
            diskret = st.sidebar.slider('Частота дискретизации', min_value=200, max_value=500, value=500, step=10)
            low_filt = st.sidebar.slider('Нижний фильтр', min_value=0.1, max_value=5.0, value=0.1, step=0.1)
            high_filt = st.sidebar.slider('Верхний фильтр', min_value=40, max_value=100, value=100, step=5)
            ica_cbox = st.sidebar.checkbox('Метод независимых компонент (ICA)')
            ssp_cbox = st.sidebar.checkbox('Проекция сигнального пространства (SSP)')

            with st.spinner("Отображение исходных данных...", width="stretch"):
                fig = create_dynamic(raw, start=start, ch_start=ch_start)
                
            '''
            ---
            ## Данные после фильтрации и изменения частоты
            '''
            try:
                raw_ed = raw.copy()
                raw_ed.resample(diskret)
                raw_ed.filter(l_freq=low_filt, h_freq=high_filt)
                with st.spinner("Отображение измененных данных...", width="stretch"):
                    fig = create_dynamic(raw_ed, start=start, ch_start=ch_start)
            except ValueError as e:
                if "must be less than Nyquist" in str(e):
                    st.warning("Невозможно восстановить сигнал с данной частотой дискретизации, используйте верхний фильтр")
                
            
                
            if ica_cbox:
                '''
                ---
                ## Метод независимых компонент
                '''
                try:
                    ica = ICA(n_components=20,
                              method='infomax',
                              random_state=97, 
                              max_iter="auto"
                             )
                    
                    ica.fit(raw_ed)
    
                    with st.spinner("Загрузка метода главных компонент...", width="stretch"):
                        st.pyplot(ica.plot_sources(raw_ed), use_container_width=True)
                except e:
                    st.warning(stre(e))
            if ssp_cbox:
                '''
                ---
                ## Проекция сигнального пространства
                '''  
                raw_ssp = raw_ed.copy()
                projs = mne.compute_proj_raw(
                    raw_ssp,
            	    n_grad=0,
            	    n_mag=0,
            	    n_eeg=2,
                    duration=2.0
                    )
                
                raw_ssp.add_proj(projs)

                with st.spinner("Загрузка проекции сигнального пространства...", width="stretch"):
                    # Посмотреть проекции
                    st.pyplot(raw_ssp.plot_projs_topomap())
                    
                    # Применить SSP
                    raw_ssp.apply_proj()
                    fig = create_dynamic(raw_ssp, start=start, ch_start=ch_start)
else:
    '''
    ### Демонстрационные файлы временно недоступны
    '''