from supertonic import TTS


def generar_tts_comentarios(comentarios, callback):
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")

    i = 1

    for comentario in comentarios:
        wav, duration = tts.synthesize(comentario['comentario'], voice_style=style, speed=1.25)
        comentario['duration'] = duration
        tts.save_audio(wav, f"tts/{comentario['frame']}.wav")
        callback(i)
        i += 1
