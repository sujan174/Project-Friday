# Credentials Directory

This directory stores authentication credentials for various agents.

## Setup Required

### Gmail Agent

To use the Gmail agent, you need to set up OAuth2 credentials:

1. Follow the setup guide: [../GMAIL_SETUP.md](../GMAIL_SETUP.md)
2. Download your OAuth2 credentials from Google Cloud Console
3. Save the file as: `gmail_credentials.json` (in this directory)

**Files:**
- `gmail_credentials.json` - OAuth2 client credentials (from Google Cloud)
- `gmail_token.json` - Auto-generated access token (**in .gitignore**)

## Security Notes

⚠️ **IMPORTANT:**

1. **Never commit** `*_token.json` files to version control
2. **Token files are in `.gitignore`** - they will not be committed
3. **Credentials files** (`gmail_credentials.json`) are safe to commit to private repos
4. **Rotate credentials** if you suspect they've been compromised

## File Structure

```
credentials/
├── README.md                  # This file
├── gmail_credentials.json     # OAuth2 credentials (download from Google Cloud)
└── gmail_token.json          # Access token (auto-generated, in .gitignore)
```

## Adding More Agents

When adding new agents that require credentials:

1. Store credentials in this directory
2. Use descriptive names: `{service}_credentials.json`
3. Add auto-generated tokens to `.gitignore`
4. Document setup in main README

## Quick Start

1. **Gmail**: Follow [GMAIL_SETUP.md](../GMAIL_SETUP.md)
2. Run the system: `python main.py`
3. Agent will prompt for authentication on first use
4. Token is saved and auto-refreshed

---

**Status:** This directory is ready for credentials. Follow setup guides to add them.
