import json
from dotenv import load_dotenv

load_dotenv()

parametros = json.load(open('dataset/parameters.jsonc', 'r', encoding='utf-8'))

fps = 60
procesar_cada_n_frames = 60
comentar_cada_n_frames = 10 * 60  # segundos * frames de video
