"""A basic LLM agent with interactive chat (split-pane TUI).

Uses prompt_toolkit for a stable, flicker-free split-screen interface.
"""

import asyncio
import sys
import yaml
import requests
from prompt_toolkit import Application
from prompt_toolkit.layout import HSplit, Layout, Window, FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse config file: {e}")
        sys.exit(1)

    required = ["api_base", "api_key"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        print(f"Error: Missing required config keys: {', '.join(missing)}")
        sys.exit(1)

    return config


def format_conversation(messages: list) -> list:
    """Format messages into styled (style, text) fragments for FormattedTextControl."""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            result.append(("class:dim", f"[system] {content}\n"))
        elif role == "user":
            result.append(("class:user", f"[user]\n"))
            for line in content.split("\n"):
                result.append(("", f"  {line}\n"))
            result.append(("", "\n"))
        elif role == "assistant":
            result.append(("class:assistant", f"[assistant]\n"))
            for line in content.split("\n"):
                result.append(("", f"  {line}\n"))
            result.append(("", "\n"))
    if not result:
        result.append(("class:dim", "No messages yet. Start the conversation!"))
    return result


def build_api_payload(config: dict, messages: list) -> dict:
    """Build the request payload for the LLM API."""
    return {
        "model": config.get("model", "gpt-3.5-turbo"),
        "messages": messages,
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 1024),
    }


async def chat(config: dict):
    """Start an interactive chat session with the LLM."""
    api_base = config["api_base"].rstrip("/")
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    waiting = False

    # --- Output pane (uses a callable so it re-renders automatically) ---
    def get_output_fragments():
        if waiting:
            return format_conversation(messages) + [("class:dim", "\n... Waiting for response ...")]
        return format_conversation(messages)

    output_control = FormattedTextControl(text=get_output_fragments)
    output_window = Window(
        content=output_control,
        wrap_lines=True,
        right_margins=[ScrollbarMargin(display_arrows=True)],
    )
    output_frame = Frame(output_window, title="Conversation")

    # --- Input pane ---
    input_area = TextArea(
        text="",
        height=3,
        prompt="You > ",
        multiline=False,
    )
    input_frame = Frame(
        input_area,
        title="Input  (/exit or Ctrl+D to quit,  /clear to clear history)",
    )

    # --- Layout ---
    layout = Layout(HSplit([output_frame, input_frame]))
    layout.focus(input_area)

    # --- Key bindings ---
    kb = KeyBindings()

    @kb.add("enter")
    async def on_submit(event):
        """Handle message submission on Enter."""
        nonlocal waiting

        text = input_area.text.strip()
        input_area.text = ""
        if not text:
            return

        # Handle commands
        if text in ("/exit", "/quit"):
            event.app.exit()
            return
        if text == "/clear":
            messages.clear()
            messages.append({"role": "system", "content": "You are a helpful assistant."})
            output_frame.title = "Conversation"
            return

        # Add user message
        messages.append({"role": "user", "content": text})

        # Show loading state
        waiting = True
        output_frame.title = "Conversation (waiting for response...)"

        # Make API call in background thread
        loop = asyncio.get_event_loop()
        payload = build_api_payload(config, messages)

        try:
            resp = await loop.run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, json=payload, timeout=60),
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            output_frame.title = "Conversation"
        except requests.exceptions.Timeout:
            messages.append({"role": "assistant", "content": "[Error: Request timed out]"})
            output_frame.title = "Conversation (error)"
        except requests.exceptions.RequestException as e:
            messages.append({"role": "assistant", "content": f"[Error: {e}]"})
            output_frame.title = "Conversation (error)"
        except (KeyError, IndexError) as e:
            messages.append({"role": "assistant", "content": f"[Error: Unexpected API response format: {e}]"})
            output_frame.title = "Conversation (error)"
        finally:
            waiting = False
            # Scroll to bottom
            output_window.vertical_scroll = 10**9

    @kb.add("c-c")
    def on_exit(event):
        """Exit on Ctrl+C."""
        raise KeyboardInterrupt

    @kb.add("c-d")
    def on_eof(event):
        """Exit on Ctrl+D."""
        event.app.exit()

    # --- Style ---
    style = Style.from_dict({
        "frame": "bg:#1a1a2e",
        "frame.title": "bold #ffffff",
        "text-area": "bg:#000000 #e0e0e0",
        "user": "bold #00bfff",
        "assistant": "bold #32cd32",
        "dim": "italic #888888",
    })

    # --- Application ---
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=True,
        mouse_support=True,
    )
    await app.run_async()


def main():
    config = load_config()
    asyncio.run(chat(config))


if __name__ == "__main__":
    main()