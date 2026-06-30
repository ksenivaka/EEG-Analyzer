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

res = st.sidebar.radio("Выберите ритмы для отображения", 
                       ["Дельта", "Тетта", "Альфа", "Бетта"],
                      captions=["Медленноволновой сон", "Засыпание", 
                                "Бодрствование с закрытыми глазами", 
                                "Бодрствование с открытыми глазами"])

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
            ## Данные выбранного ритма
            '''            
            with st.spinner("Отображение исходных данных...", width="stretch"):
                raw_ed = raw.copy()
                low_filt = 0
                high_filt = 100
                match res:
                    case "Дельта":
                        low_filt = 0.5
                        high_filt = 3.5
                    case "Тетта":
                        low_filt = 4
                        high_filt = 7
                    case "Альфа":
                        low_filt = 8
                        hihg_filt = 13
                    case "Бетта":
                        low_filt = 13.5
                        high_filt = 30

                raw_ed.filter(l_freq=low_filt, h_freq=high_filt)
                fig = create_dynamic(raw_ed)
                
            