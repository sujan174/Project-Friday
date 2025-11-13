# Simple TerminalUI stub for backward compatibility
class TerminalUI:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def start_progress(self, message: str):
        return None

    def print_response(self, message: str):
        print(message)

    def print_header(self, session_id: str):
        print(f"Session: {session_id}")

    def print_prompt(self):
        print("\n> ", end="", flush=True)

    def print_goodbye(self):
        print("\nGoodbye!")

    def print_thinking(self):
        print("Thinking...")

    def print_error(self, error: str):
        print(f"Error: {error}")
