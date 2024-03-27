import streamlit as st
from diarize import raw_diarize, get_token
from tempfile import NamedTemporaryFile

if "counter" not in st.session_state:
    st.session_state.counter = 0
if "result" not in st.session_state:
    st.session_state.result = ""

ss = st.session_state
st.title('CallSentra 📞')
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<p class="big-font">Transforms customer support with AI-driven voice call analysis. 
            Our service evaluates support representatives, identifies patterns, sentiments, and opportunities for improvement. \
            By transforming call monitoring, CallSentra empowers companies to optimize operations, enhance efficiency, and elevate customer satisfaction.​</p>""", unsafe_allow_html=True)

audio = st.file_uploader("Upload an audio file", type=["mp3"])
if audio is not None:
    with NamedTemporaryFile(suffix="mp3", delete=False) as temp:
        temp.write(audio.getvalue())
        temp.seek(0)
        st.audio(temp.read())
        json_obj = raw_diarize(hf_token=get_token(), audio_path=fr"{temp.name}", device="cuda")
        for speaker_info in json_obj: 
            text = f"[{speaker_info['start']} : {speaker_info['end']}] {speaker_info['speaker']}: {speaker_info['text']}\n"
            st.write(text)
