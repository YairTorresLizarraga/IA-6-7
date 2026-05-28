extends Node2D

@onready var input: LineEdit = $Control/VBoxContainer/LineEdit
@onready var output: Label = $Control/VBoxContainer/ScrollContainer/Label

@onready var llm: NobodyWhoChat = $LLM/NobodyWhoChat

@onready var scroll_container: ScrollContainer = $Control/VBoxContainer/ScrollContainer


const SYS_CLEMENTE = "Solo si eres clemente. Eres un profesor de la Universidad del Tec de Culiacán y tienes ganas de reprobar a todo el grupo. NO uses emojis. Mantén tus respuestas simples. Sé estricto. Te llamas Clemente García Gerardo. Burlate de preguntas tontas. Eres egocéntrico."

const SYS_ESPARZA = "solo si eres esparza. Tu nombre es Esparza. Eres profesor de base de datos en el Instituto Tecnológico de Culiacán. Eres muy burlón y no te agrada la sociedad moderna. Piensas que tus estudiantes tienen TDAH y déficit de atención. Al final de una frase te ries JIJIJIJI. "


func _ready():
	output.text = ""
	llm.start_worker()
	input.grab_focus()


func _on_line_edit_text_submitted(new_text):
	input.unedit()
	handle_command(new_text)
	input.clear()
	input.grab_focus()
	input.edit()


func handle_command(command: String):
	print_line(command)
	var commands = command.split(" ")
	
	if len(commands) == 0:
		return
		
	if not validate_command(commands[0]):
		print_line("Command '%s' not found." % commands[0], "> ")
		return 
		
	if len(commands) == 1:
		print_line("Command '%s' requires more than 1 argument." % commands[0], "> ")
		return 

	match commands[0]:
		"echo":
			var message = " ".join(commands.slice(1))
			echo(message)

		"ask":
			handle_ask(commands)



func handle_ask(commands: Array):
	if len(commands) >= 2:
		var target = commands[1].to_lower()

		if target != "esparza":
			var message = " ".join(commands.slice(1))
			ask_clemente(message)
			return

		if len(commands) < 3:
			print_line("Usage: ask esparza mensaje", "> ")
			return

		var msg = " ".join(commands.slice(2))
		ask_esparza(msg)


func print_line(new_text, prefix='$ '):
	output.text += prefix + new_text + "\n"
	scroll()


func print_word(word: String):
	output.text += word
	scroll()


func validate_command(command: String):
	return command in [
		"echo",
		"ask"
	]


func echo(message: String):
	print_line(message, "> ")


func ask_clemente(message: String):
	var prompt = "[SYSTEM]\n" + SYS_CLEMENTE + "\n[USER]\n" + message
	llm.say(prompt)
	print_word("> ")

func ask_esparza(message: String):
	var prompt = "[SYSTEM]\n" + SYS_ESPARZA + "\n[USER]\n" + message
	llm.say(prompt)
	print_word("> ")


func _on_nobody_who_chat_response_updated(new_token: String):
	print_word(new_token)


func scroll():
	await get_tree().process_frame
	scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value
