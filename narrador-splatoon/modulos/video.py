import platform
import subprocess
import os
from modulos.common import fps


def generar_video(video: str, ruta_salida: str, comentarios) -> str:
    salida_subtitulos = "comentarios.srt"
    __generar_subtitulos(comentarios, salida_subtitulos)

    cmd = ["ffmpeg", "-y", "-i", video, "-i", "comentarios.srt"]
    video_codec = "h264_videotoolbox" if platform.system() == "Darwin" and platform.machine() == "arm64" else "libx264"

    for comentario in comentarios:
        cmd.extend(["-i", f"tts/{comentario['frame']}.wav"])

    filter_parts = ["[0:a]volume=0.5[main_a]"]
    audio_inputs = ["[main_a]"]

    # Obtenemos TODOS los tiempos de inicio para poder cortar los audios dinámicamente
    all_start_seconds = [c['frame'] / fps for c in comentarios]

    tts_input_offset = 2  # 0=video, 1=subtitulos, 2..=clips TTS
    for i, comentario in enumerate(comentarios):
        start_seconds = all_start_seconds[i]
        duration = comentario.get('duration')

        # Buscamos cuándo empieza el SIGUIENTE comentario (si existe)
        next_start = all_start_seconds[i + 1] if i < len(comentarios) - 1 else None

        clip_filters = []

        # Si el siguiente comentario empieza ANTES de que este termine, lo cortamos (atrim)
        if next_start is not None and (duration is None or next_start < start_seconds + duration):
            trim_duration = next_start - start_seconds
            clip_filters.append(f"atrim=duration={trim_duration:.3f}")
            clip_filters.append("asetpts=PTS-STARTPTS")

        delay_ms = round(start_seconds * 1000)
        clip_filters.append(f"adelay={delay_ms}:all=1")

        # Volumen al 150% para que resalte sobre el juego
        clip_filters.append("volume=1.5")

        audio_label = f"a{i + 1}"
        filter_parts.append(f"[{i + tts_input_offset}:a]{','.join(clip_filters)}[{audio_label}]")
        audio_inputs.append(f"[{audio_label}]")

    filter_parts.append(
        f"{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=first:dropout_transition=0:normalize=0[outa]"
    )

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "0:v",
        "-map", "[outa]",
        "-map", "1:s",
        "-c:v", video_codec,
        "-c:a", "aac",
        "-c:s", "mov_text",
        "-shortest",
        ruta_salida
    ])

    print("Generando video final con audio ajustado y subtítulos...")
    subprocess.run(cmd, check=True)
    os.remove("comentarios.srt")


def __generar_subtitulos(comentarios, salida):
    srt_data = ""
    for srt_i, comentario in enumerate(comentarios, start=1):
        proximo_comentario = comentarios[srt_i] if srt_i < len(comentarios) else None
        inicio = comentario['frame'] / fps
        fin = (proximo_comentario['frame'] / fps) if proximo_comentario is not None else 99999999

        srt_data += f"{srt_i}\n{__segundos_a_srt(inicio)} --> {__segundos_a_srt(fin)}\n{comentario['comentario']}\n\n"

    with open(salida, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_data)


def __segundos_a_srt(segundos: float) -> str:
    total_ms = max(0, round(segundos * 1000))
    horas = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutos = total_ms // 60_000
    total_ms %= 60_000
    segundos_enteros = total_ms // 1000
    milisegundos = total_ms % 1000
    return f"{horas:02d}:{minutos:02d}:{segundos_enteros:02d},{milisegundos:03d}"
