# Project Aerius - Enhanced Terminal UI

Production-grade terminal interface combining the minimal aesthetic of Claude Code with the polish of modern CLI tools like Gemini CLI.

## Features

### 🎨 Beautiful Design
- **Minimal yet polished** - Clean typography with subtle colors
- **Syntax highlighting** - Code blocks with Monokai theme
- **Rich markdown rendering** - Beautiful formatted text
- **Professional color scheme** - Carefully selected palette

### ⚡ Interactive Elements
- **Animated spinners** - Smooth loading indicators during operations
- **Progress bars** - Visual feedback for multi-step tasks
- **Live status updates** - Real-time operation tracking
- **Context managers** - Clean API for operation wrapping

### 📊 Session Analytics
- **Performance tables** - Agent statistics with success rates
- **Timing metrics** - Response times and latencies
- **Session summaries** - Duration, message count, agent calls
- **Health indicators** - Visual agent status (● healthy, ● degraded, ● down)

### 🎯 User Experience
- **Error panels** - Clear, beautiful error messages
- **Command system** - Built-in commands (help, stats, agents, exit)
- **Verbose mode** - Detailed operation logging when needed
- **Fallback support** - Gracefully degrades without Rich library

## Installation

### With Enhanced UI (Recommended)

```bash
# Install all dependencies including Rich
pip install -r requirements.txt
```

### Basic UI Only

The system works without Rich, using a minimal fallback interface.

```bash
# Install core dependencies only
pip install google-generativeai python-dotenv numpy
```

## Usage

### Running with Enhanced UI (Default)

```bash
# Standard mode
python main.py

# Verbose mode (shows detailed operation logs)
python main.py --verbose
python main.py -v
```

### Running with Simple UI

```bash
# Force simple UI (no Rich library required)
python main.py --simple
```

### Built-in Commands

During a session, you can use these commands:

| Command | Description |
|---------|-------------|
| `help` | Show available commands and agents |
| `stats` | Display session statistics |
| `agents` | List all available agents with status |
| `exit` | Exit the system (also: quit, bye, q) |

## UI Components

### Welcome Screen

```
Project Aerius
Multi-Agent Orchestration System
Session 0aceb6247889

✓ 7 agents initialized
  • Slack Agent, Jira Agent, Github Agent, Notion Agent, Browser Agent
  • ... and 2 more
```

### Agent Operations

**Non-verbose mode** (clean):
```
❯ Create a Jira issue for bug fix

[Response appears directly]
```

**Verbose mode** (detailed):
```
❯ Create a Jira issue for bug fix

→ Jira Agent...
  ✓ 245ms

I've created a Jira issue...
```

### Session Statistics

```
Session Summary
┌──────────┬──────────┐
│ Duration │ 5m 23s   │
│ Messages │ 12       │
│ Agent... │ 18       │
└──────────┴──────────┘

Agent Performance
┌────────────────┬───────┬──────────┬─────────┐
│ Agent          │ Calls │ Avg Time │ Success │
├────────────────┼───────┼──────────┼─────────┤
│ Slack Agent    │ 5     │ 234ms    │ 100%    │
│ Jira Agent     │ 8     │ 456ms    │ 100%    │
│ Github Agent   │ 3     │ 678ms    │ 100%    │
│ Notion Agent   │ 2     │ 345ms    │ 100%    │
└────────────────┴───────┴──────────┴─────────┘
```

### Error Display

Beautiful error panels with rounded borders:
```
╭──── Error ────────────────────────────────────╮
│ Failed to connect to Slack workspace         │
│ Please check your SLACK_TOKEN in .env file   │
╰───────────────────────────────────────────────╯
```

## Color Scheme

Carefully selected professional colors:

| Element | Color | Usage |
|---------|-------|-------|
| Primary | `#00A8E8` | Bright blue - main accent, prompts |
| Success | `#00C853` | Green - successful operations |
| Warning | `#FFB300` | Amber - warnings, retries |
| Error | `#FF1744` | Red - errors, failures |
| Accent | `#7C4DFF` | Deep purple - spinners, highlights |
| Muted | `#78909C` | Blue grey - secondary text |
| Dim | `#546E7A` | Dim grey - debug, timestamps |

## Architecture

### Class: `EnhancedUI`

The main UI class with two design modes:

1. **Rich Mode** - Full-featured with animations and formatting
2. **Fallback Mode** - Simple text-based interface

### Key Methods

#### Session Management
- `print_welcome(session_id)` - Display welcome screen
- `print_goodbye()` - Exit message
- `print_session_summary(stats)` - Session statistics

#### User Interaction
- `print_prompt()` - Interactive prompt (❯)
- `print_response(response)` - Formatted response with markdown
- `print_help(orchestrator)` - Help with agent table

#### Agent Operations
- `agent_operation(agent_name, instruction)` - Context manager with spinner
- `thinking_spinner(message)` - Thinking indicator
- `show_retry(agent, attempt, max)` - Retry notification

#### Messages
- `print_error(error)` - Beautiful error panel
- `print_warning(warning)` - Warning with icon
- `print_success(message)` - Success with checkmark
- `print_info(info)` - Info message

#### Advanced
- `progress_bar(description, total)` - Context manager for progress
- `print_code(code, language)` - Syntax-highlighted code
- `print_banner(text, style)` - Banner message

## Context Managers

### Agent Operation with Spinner

```python
# In your code
with ui.agent_operation("slack_agent", "Send message"):
    # Your operation here
    result = await agent.execute()
    # Spinner shows automatically, timing tracked
```

### Thinking Spinner

```python
with ui.thinking_spinner("Analyzing request"):
    # Your analysis code
    intelligence = await classify_intent()
```

### Progress Bar

```python
with ui.progress_bar("Processing files", total=100) as (progress, task):
    for i in range(100):
        # Your work
        progress.update(task, advance=1)
```

## Customization

### Changing Colors

Edit `enhanced_ui.py`:

```python
self.colors = {
    'primary': '#YOUR_COLOR',
    'success': '#YOUR_COLOR',
    # ... etc
}
```

### Adjusting Spinners

Available spinners: `dots`, `arc`, `line`, `pipe`, `dots12`, `bouncingBar`

```python
with self.console.status(
    message,
    spinner="dots",  # Change this
    spinner_style=self.colors['primary']
):
    # Your operation
```

### Custom Themes for Code

Available themes: `monokai`, `github-dark`, `dracula`, `nord`, `one-dark`

```python
syntax = Syntax(
    code,
    language,
    theme="monokai",  # Change this
    line_numbers=True
)
```

## Performance

The Enhanced UI is designed to be lightweight:

- **Spinners**: ~0.1ms overhead per frame
- **Markdown rendering**: ~2-5ms for typical responses
- **Tables**: ~1-3ms for statistics display
- **No blocking**: All animations run in background

## Backward Compatibility

The `claude_ui.py` remains available for:
- Environments without Rich library
- Systems with limited terminal capabilities
- Users who prefer minimal UI
- Testing and debugging

Switch between UIs using the `--simple` flag:

```bash
# Enhanced UI
python main.py

# Simple UI
python main.py --simple
```

## Comparison: Enhanced vs Simple UI

| Feature | Enhanced UI | Simple UI |
|---------|-------------|-----------|
| Spinners | ✓ Animated | ✗ None |
| Markdown | ✓ Rich | ✓ Basic |
| Syntax Highlighting | ✓ Yes | ✗ No |
| Tables | ✓ Beautiful | ✗ Plain text |
| Colors | ✓ Full palette | ✓ Basic ANSI |
| Progress Bars | ✓ Yes | ✗ No |
| Error Panels | ✓ Bordered | ✓ Plain |
| Dependencies | Rich required | None |
| Performance | Fast | Faster |

## Tips

### For Best Experience

1. Use a modern terminal (iTerm2, Alacritty, Windows Terminal)
2. Enable verbose mode to see operation details: `python main.py -v`
3. Check statistics during long sessions: type `stats`
4. Use `help` to see available agents and their status

### Troubleshooting

**Issue**: Colors look wrong
- **Solution**: Ensure your terminal supports 256 colors or true color

**Issue**: Spinners don't animate
- **Solution**: Install Rich library: `pip install rich`

**Issue**: Slow rendering
- **Solution**: Use `--simple` mode for faster text-only interface

**Issue**: Import errors
- **Solution**: Install all requirements: `pip install -r requirements.txt`

## Future Enhancements

Planned features:
- [ ] Interactive agent selection menu
- [ ] Command history with arrow keys
- [ ] Autocomplete for agent names
- [ ] Custom themes (light/dark mode)
- [ ] Export session to HTML/PDF
- [ ] Multi-line input support
- [ ] Inline previews for images/files

## Credits

Inspired by:
- **Claude Code** - Minimal aesthetic and clean design
- **Gemini CLI** - Command system and session management
- **Rich Library** - Beautiful terminal formatting

Built with ❤️ for the AI developer community.
