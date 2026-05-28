# LLM in Godot

This project demonstrates how to integrate a large language model (LLM) into a Godot Engine project directly, using the [nobodywho](https://github.com/nobodywho-ooo/nobodywho) library. With this setup, you can run an LLM-powered chatbot, narrative generator, or any other text-based AI interaction entirely within Godot, without relying on cloud APIs.

In this project tutorial, we are building a terminal based text-adventure game powered by large language model.

![Demo 1](./screenshots/demo-1.jpg)

![Demo 1](./screenshots/demo-2.jpg)

![Demo 1](./screenshots/demo-3.jpg)


## 🚀 Getting Started

### Dependencies

- Godot 4.x or later installed.
- nobodywho installed into a Godot Project (via Asset Library).
- A compatible GGUF model (e.g., [gemma-2-2b-it-Q4_K_M.gguf](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf)).

### How to Run?

1. Clone this repository and launch it in Godot 4
2. Install [nobodywho](https://github.com/nobodywho-ooo/nobodywho) Godot Extension in your project.
3. Download GGUF based model from HuggingFace and place them into ggufs folder.
4. Update path to downloaded GGUF file in the NobodyWho node in the main scene.
5. Run the game. 

## 📜 License

This project is provided under the [MIT License](./LICENSE).
