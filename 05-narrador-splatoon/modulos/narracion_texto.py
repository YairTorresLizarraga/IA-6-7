import random
import re

from llama_cpp import Llama

from modulos.common import fps

__ventana_agrupacion_splats_frames = 5 * fps


def generar_narracion(historial_deteccion, callback: callable):
    llm_text_gen = Llama.from_pretrained(
        repo_id="unsloth/gemma-4-E4B-it-GGUF",
        filename="gemma-4-E4B-it-Q8_0.gguf",
        n_ctx=8192,
        n_gpu_layers=-1,
    )

    comentarios = []
    frames_splat_agrupados = set()

    for i in range(len(historial_deteccion)):
        resultados = historial_deteccion[i]
        if resultados['frame'] in frames_splat_agrupados:
            continue
        if not resultados['debe_comentar']:
            continue

        eventos = __resumir_evento(resultados)
        splatted_count = resultados.get('splatted_count', 0)

        if resultados.get('splatted_detected'):
            frame_splat_inicial = resultados['frame']
            splat_frames_unicos = [frame_splat_inicial]
            splatted_count = max(1, splatted_count)
            for futuro in historial_deteccion[i + 1:]:
                if futuro['frame'] - frame_splat_inicial > __ventana_agrupacion_splats_frames:
                    break
                if not futuro.get('splatted_detected'):
                    break
                frames_splat_agrupados.add(futuro['frame'])
                futuro_splatted_count = max(1, futuro.get('splatted_count', 1))
                if futuro_splatted_count > splatted_count:
                    splat_frames_unicos.append(futuro['frame'])
                    splatted_count = futuro_splatted_count
            resultados = resultados.copy()
            resultados['splatted_count'] = splatted_count
            resultados['splatted_grouped_count'] = splatted_count
            resultados['splatted_grouped_frames'] = splat_frames_unicos

        # Traducimos todo el caos de variables a un párrafo simple para el LLM
        contexto_natural = __traducir_contexto(resultados, eventos, splatted_count)

        mensajes = [
            {
                "role": "system",
                "content": (
                    "Eres un caster (comentarista) profesional de e-sports especializado en Splatoon. "
                    "Tu objetivo es narrar la acción en tiempo real con alta energía en UNA SOLA FRASE.\n\n"

                    "LO QUE ESTÁ PASANDO EN ESTE INSTANTE:\n"
                    f"{contexto_natural}\n\n"

                    "REGLAS ESTRICTAS DE JERGA Y TONO:\n"
                    "- Usa terminología de Splatoon (Splat, liquidado, entintar, zona, especial, holdear, pushear).\n"
                    "- No inventes nombres de equipos ni marcadores exactos.\n"
                    "- NO uses frases genéricas como 'el control está en sus manos'.\n"
                    "- Si el evento es respawn, habla de la caída y la reentrada, no de un splat a favor.\n"
                    "- Tu respuesta debe tener entre 40 y 85 caracteres.\n"
                    "- Devuelve SOLO el comentario final, sin comillas, sin introducciones ('Aquí está...') ni emojis."
                )
            },
            {
                "role": "user",
                "content": "Narra lo que acaba de pasar en la partida."
            }
        ]

        historial_comentarios = [c['comentario'] for c in comentarios[-10:]]
        texto_generado = None

        for intento in range(3):
            salida = llm_text_gen.create_chat_completion(
                mensajes,
                max_tokens=48,
                temperature=0.75 + (intento * 0.1),  # Bajamos un poco la temperatura base para mayor coherencia
                repeat_penalty=1.12,  # Reducido de 1.20 a 1.12 para no romper la gramática del español
                stop=["\n", "\n\n", "\"", "Aquí"],  # Añadimos comillas para que corte si intenta citar algo
            )

            candidato = salida['choices'][0]['message']['content']
            candidato = __normalizar_comentario(candidato)

            if not __es_demasiado_repetido(candidato, historial_comentarios) and not __comentario_incompleto(candidato):
                texto_generado = candidato
                break

        # Si el LLM falla 3 veces en dar un buen comentario, entra el Fallback Dinámico
        if texto_generado is None:
            texto_generado = __comentario_fallback_dinamico(eventos, historial_comentarios)

        generacion = {
            "frame": resultados['frame'],
            "comentario": texto_generado,
            "splatted_detected": resultados.get('splatted_detected', False),
            "splatted_count": resultados.get('splatted_count', 0),
            "splatted_grouped_frames": resultados.get('splatted_grouped_frames', []),
            "player_was_killed": resultados.get('player_was_killed', False),
            "health_worsened": resultados.get('health_worsened', False),
            "health_recovered": resultados.get('health_recovered', False),
            "health": resultados.get('health'),
        }

        comentarios.append(generacion)
        callback(len(comentarios))
        print(f"[Frame {resultados['frame']}] {texto_generado}")
    return comentarios


def __traducir_contexto(resultados, eventos, splatted_count):
    tiempo = resultados.get('tiempo_ocr')
    estado_juego = resultados.get('game_status', 'juego')

    traduccion = f"La partida está en la fase de {estado_juego}. "
    if tiempo is not None and tiempo < 60:
        traduccion += f"Quedan solo {tiempo} segundos en el reloj. "

    if splatted_count > 0:
        traduccion += f"¡El jugador acaba de conseguir {splatted_count} baja(s) rápida(s) (splat)! Es una jugada clave. "
    elif resultados.get('player_was_killed'):
        traduccion += "Mataron al jugador. "
    else:
        # Ahora confiamos solo en los eventos confirmados, no en la lectura del frame suelto
        if 'health_worsened' in eventos:
            traduccion += "El jugador ha recibido muchísimo daño de repente, su vida está en peligro. "
        elif 'health_recovered' in eventos:
            traduccion += "El jugador logró recuperarse del daño y su salud está estable de nuevo. "

        if 'ultimate_became_ready' in eventos:
            traduccion += "La habilidad especial acaba de terminar de cargar y está lista para usarse. "
        elif 'ultimate_used' in eventos:
            traduccion += "El jugador acaba de activar su habilidad especial para presionar. "

    if not any(e in eventos for e in
               ['splatted_detected', 'health_worsened', 'health_recovered', 'ultimate_became_ready', 'ultimate_used',
                'player_was_killed']):
        contextos_inactivos = [
            "El jugador está relajado, pintando el mapa con alegría y fluyendo por el escenario.",
            "Hay muchísima tensión y paranoia en el aire, el jugador pinta esperando una emboscada inminente.",
            "El jugador está metiendo presión agresiva, asfixiando al rival con pintura para arrinconarlo.",
            "Movimiento frenético puro: el jugador rota y pinta a toda velocidad sin quedarse quieto.",
            "El jugador está aplicando una estrategia paciente, buscando posiciones seguras con inteligencia."
        ]
        traduccion += random.choice(contextos_inactivos)

    return traduccion


def __comentario_fallback_dinamico(eventos, comment_history):
    aperturas = ["¡Atención!", "¡Cuidado!", "¡Ojo aquí!", "¡Qué momento!", "¡Se mueve la partida!", "¡Atentos!"]

    if 'splatted_detected' in eventos:
        bases = ["Bajas encadenadas, toca presionar la zona.", "Splats a favor, se abre el mapa.",
                 "¡Varios menos en el equipo rival!"]
    elif 'player_was_killed' in eventos:
        bases = ["Cayó el jugador, toca pensar la reentrada.", "Reaparición en marcha, hay que reagruparse.",
                 "Lo liquidan y ahora toca volver con calma."]
    elif 'health_worsened' in eventos:
        bases = ["Va muy tocado, tiene que esconderse.", "La tinta enemiga aprieta, necesita cobertura.",
                 "Está al borde de caer, ¡cuidado!"]
    elif 'health_recovered' in eventos:
        bases = ["Recupera vida y puede volver a pelear.", "Ya estabilizó la salud, toca retomar espacio.",
                 "Sale del peligro y vuelve a pintar con calma."]
    elif 'ultimate_became_ready' in eventos:
        bases = ["Especial cargado y listo para soltarlo.", "Ojo con el especial que puede cambiar todo.",
                 "Tiene la ulti lista en la bolsa."]
    elif 'ultimate_used' in eventos:
        bases = ["Especial en camino, toca ganar terreno.", "Suelta el especial para romper la línea.",
                 "¡Aprovechando esa ulti para presionar!"]
    else:
        bases = [
            # Alegría y Diversión (Disfrutando el juego)
            "¡Qué bonito es ver el mapa llenarse de color a este ritmo!",
            "Pintando por aquí, pintura por allá, ¡dejando todo impecable!",
            "¡Navegando por la tinta con todo el estilo del mundo!",
            "Pintando de lo lindo y pasándola genial en el mapa.",
            "¡Haciendo verdadero arte en el escenario a base de tinta!",
            "Disfrutando de la partida y cubriendo cada rincón con alegría.",

            # Tranquilidad y Relax (Tomando un respiro)
            "Un momento de paz para nadar tranquilo y recargar el tanque.",
            "Respirando profundo y pintando el mapa sin nada de estrés.",
            "Todo fluye con calma, asegurando el terreno como en un paseo.",
            "Relajando el ritmo un segundito para disfrutar del entintado.",
            "Pintando a sus anchas, sin nadie que moleste por ahora.",
            "Aprovechando la calma para dejar la casa limpia y pintada.",

            # Diversión táctica (Juego fresco)
            "Un poco de limpieza en el mapa nunca viene mal, ¡a pintar se ha dicho!",
            "Moviéndose como pez en el agua, preparando la zona con mucha frescura.",
            "¡Saltando y pintando a gusto mientras preparan la siguiente jugada!",
            "Cubriendo el mapa con mucha buena vibra y excelente control.",
            "¡Fluyendo por la tinta como si estuvieran surfeando en la zona!",

            # Tensión y Paranoia (Esperando el ataque)
            "¡El ambiente se corta con tijera! Todos buscando el mínimo error.",
            "¡Tensión al máximo! Nadie cede un solo milímetro de tinta.",
            "¡Mucho cuidado con los flancos, en cualquier momento salta la chispa!",
            "Ese silencio en el mapa es peligroso, alguien está preparando una emboscada.",
            "¡Pintando con nervios de acero, sabiendo que el rival acecha en la tinta!",

            # Agresividad contenida y Presión (Dominando el mapa)
            "¡Pintando con furia para arrinconar y asfixiar al equipo contrario!",
            "¡Metiendo presión pura a base de pintura, ahogando las salidas del rival!",
            "¡Ese entintado es una amenaza directa, se viene el empuje fuerte!",
            "¡Imponiendo respeto en el centro, obligando al rival a retroceder!",
            "¡Devorando el mapa! Quieren dejar al enemigo sin opciones de movilidad.",

            # Velocidad y Movimiento frenético
            "¡Movimiento frenético en la zona, no se quedan quietos ni un segundo!",
            "¡Qué agilidad para rotar y seguir pintando el mapa sin parar!",
            "¡Velocidad a tope! Cubriendo cada rincón antes de que el rival reaccione.",
            "¡No hay tiempo que perder, entintando a una velocidad de locura!",
            "¡Rotaciones rapidísimas para mantener al enemigo desorientado!",

            # Estrategia agresiva
            "Cerrando todas las vías de escape, preparan la trampa perfecta.",
            "¡Asegurando la altura de forma agresiva para dominar la visión!",
            "Pintando de forma estratégica, ¡quieren encerrarlos en su propia base!",
            "¡Forzando al rival a jugar incómodo robándole todo el territorio!",
            "Preparando el terreno para la siguiente masacre, ¡pura estrategia!"
        ]

    random.shuffle(aperturas)
    random.shuffle(bases)

    for apertura in aperturas:
        for base in bases:
            candidato = f"{apertura} {base}"
            if not __es_demasiado_repetido(candidato, comment_history):
                return candidato

    # Si por alguna razón todo falla, devuelve uno seguro
    return "¡Hay que mantener el ritmo en la zona!"


def __normalizar_comentario(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", texto)
    texto = texto.split("\n", 1)[0].strip()
    if len(texto) > 120:
        texto = texto[:120].rsplit(" ", 1)[0]
    return texto


def __es_demasiado_repetido(texto: str, comment_history) -> bool:
    texto_normalizado = texto.lower().strip("¡!., ")
    for previo in comment_history:
        previo_normalizado = previo.lower().strip("¡!., ")
        if texto_normalizado == previo_normalizado:
            return True
        palabras = set(texto_normalizado.split())
        palabras_previas = set(previo_normalizado.split())
        if palabras and len(palabras & palabras_previas) / len(palabras) >= 0.75:
            return True
    return False


def __comentario_incompleto(texto: str) -> bool:
    texto = texto.strip()
    if not texto:
        return True
    if texto.endswith((",", "...", "…", ":", ";", " y", " pero", " con", " para")):
        return True
    return texto[-1] not in ".!?¡!"


def __resumir_evento(resultados):
    eventos = []
    if resultados.get('splatted_count', 0) > 0:
        eventos.append('splatted_detected')
    if resultados.get('player_was_killed'):
        eventos.append('player_was_killed')
    if resultados.get('tempo_comment'):
        eventos.append('tempo_comment')
    if resultados.get('ultimate_became_ready'):
        eventos.append('ultimate_became_ready')
    if resultados.get('ultimate_used'):
        eventos.append('ultimate_used')
    if resultados.get('health_worsened'):
        eventos.append('health_worsened')
    if resultados.get('health_recovered'):
        eventos.append('health_recovered')
    if not eventos:
        eventos.append('tempo_comment')
    return eventos
