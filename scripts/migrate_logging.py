#!/usr/bin/env python3
"""
Automated Logging Migration Script

Converts print() statements to structured logging calls.
Creates backups of all modified files.

Usage:
    python migrate_logging.py [--dry-run] [--file path/to/file.py]

Options:
    --dry-run: Show what would be changed without modifying files
    --file: Migrate specific file only (default: all .py files)
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class LoggingMigrator:
    """Migrates print() statements to proper logging"""

    # Patterns to detect log levels from print content
    DEBUG_PATTERNS = [
        r'\[DEBUG\]', r'verbose', r'Fetching', r'Calling', r'Arguments',
        r'Result:', r'Details:', r'Using cached'
    ]

    INFO_PATTERNS = [
        r'\[INFO\]', r'completed', r'success', r'created', r'updated',
        r'initialized', r'started', r'finished'
    ]

    WARNING_PATTERNS = [
        r'\[WARNING\]', r'Warning', r'⚠', r'Note:', r'Could not',
        r'Failed to', r'Unable to', r'Skipping'
    ]

    ERROR_PATTERNS = [
        r'\[ERROR\]', r'Error', r'❌', r'✗', r'Exception', r'Failed',
        r'error_msg', r'error:'
    ]

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'prints_replaced': 0,
            'verbose_checks_removed': 0,
            'imports_added': 0
        }

    def migrate_file(self, file_path: Path) -> bool:
        """
        Migrate a single file.

        Returns:
            True if file was modified, False otherwise
        """
        self.stats['files_scanned'] += 1

        try:
            content = file_path.read_text()
            original_content = content

            # Check if logging is already imported
            has_logger_import = 'from logging_config import get_logger' in content
            has_logger_created = re.search(r'logger\s*=\s*get_logger\s*\(', content)

            # Migrate content
            content, num_replacements = self._migrate_content(content)

            # Add logger import if needed
            if num_replacements > 0 and not has_logger_import:
                content = self._add_logger_import(content, file_path)
                self.stats['imports_added'] += 1

            # Check if anything changed
            if content != original_content:
                if not self.dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix('.py.bak')
                    shutil.copy2(file_path, backup_path)

                    # Write new content
                    file_path.write_text(content)
                    print(f"✓ Migrated: {file_path} ({num_replacements} changes)")
                else:
                    print(f"[DRY RUN] Would migrate: {file_path} ({num_replacements} changes)")

                self.stats['files_modified'] += 1
                self.stats['prints_replaced'] += num_replacements
                return True

            return False

        except Exception as e:
            print(f"✗ Error migrating {file_path}: {e}")
            return False

    def _migrate_content(self, content: str) -> Tuple[str, int]:
        """
        Migrate print statements in content.

        Returns:
            (modified_content, number_of_replacements)
        """
        lines = content.split('\n')
        modified_lines = []
        replacements = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for verbose-guarded print
            if 'if self.verbose:' in line or 'if verbose:' in line:
                # Look at next line(s) for print statement
                indent = len(line) - len(line.lstrip())
                j = i + 1

                # Collect indented block
                block_lines = [line]
                while j < len(lines):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip())

                    if next_indent > indent and next_line.strip():
                        block_lines.append(next_line)
                        j += 1
                    else:
                        break

                # Process block
                new_block, block_replacements = self._migrate_verbose_block(block_lines)
                replacements += block_replacements

                if block_replacements > 0:
                    modified_lines.extend(new_block)
                    self.stats['verbose_checks_removed'] += 1
                    i = j
                    continue

            # Check for standalone print
            if 'print(' in line and not line.strip().startswith('#'):
                new_line, was_replaced = self._migrate_print_line(line)
                modified_lines.append(new_line)
                if was_replaced:
                    replacements += 1
                i += 1
                continue

            # Keep line unchanged
            modified_lines.append(line)
            i += 1

        return '\n'.join(modified_lines), replacements

    def _migrate_verbose_block(self, block_lines: List[str]) -> Tuple[List[str], int]:
        """
        Migrate a verbose-guarded block.

        Returns:
            (modified_lines, number_of_replacements)
        """
        replacements = 0
        new_lines = []

        for line in block_lines:
            if 'if self.verbose:' in line or 'if verbose:' in line:
                # Skip the if statement (logging handles this via log level)
                continue

            if 'print(' in line:
                # Reduce indent (was inside if block)
                indent = len(line) - len(line.lstrip())
                dedented_line = ' ' * (indent - 4) + line.lstrip()

                # Migrate print to logger call
                new_line, was_replaced = self._migrate_print_line(dedented_line)
                new_lines.append(new_line)
                if was_replaced:
                    replacements += 1
            else:
                # Keep other lines (with reduced indent)
                indent = len(line) - len(line.lstrip())
                if indent >= 4:
                    dedented_line = ' ' * (indent - 4) + line.lstrip()
                    new_lines.append(dedented_line)
                else:
                    new_lines.append(line)

        return new_lines, replacements

    def _migrate_print_line(self, line: str) -> Tuple[str, bool]:
        """
        Migrate a single print statement to logger call.

        Returns:
            (modified_line, was_replaced)
        """
        # Extract print content
        match = re.search(r'print\s*\((.*)\)', line, re.DOTALL)
        if not match:
            return line, False

        print_content = match.group(1)

        # Determine log level
        log_level = self._determine_log_level(print_content)

        # Extract f-string or regular string
        message = self._extract_message(print_content)

        # Build logger call
        indent = len(line) - len(line.lstrip())
        logger_call = f"{' ' * indent}logger.{log_level}({message})"

        return logger_call, True

    def _determine_log_level(self, content: str) -> str:
        """Determine appropriate log level from content"""
        content_lower = content.lower()

        # Check error patterns first (highest priority)
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.ERROR_PATTERNS):
            return 'error'

        # Warning patterns
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.WARNING_PATTERNS):
            return 'warning'

        # Info patterns
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.INFO_PATTERNS):
            return 'info'

        # Debug patterns (most verbose)
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.DEBUG_PATTERNS):
            return 'debug'

        # Default to info
        return 'info'

    def _extract_message(self, print_content: str) -> str:
        """
        Extract and clean message from print content.

        Removes agent prefixes like [SLACK AGENT] since logger provides module name.
        """
        # Remove common prefixes
        message = print_content

        # Remove agent name prefixes in brackets
        message = re.sub(r'f?"?\[[\w\s]+\]\s*', '', message)

        # Remove color codes
        message = re.sub(r'{C\.\w+}', '', message)
        message = re.sub(r'{C\.ENDC}', '', message)

        # Clean up
        message = message.strip()

        return message

    def _add_logger_import(self, content: str, file_path: Path) -> str:
        """Add logger import at top of file"""
        lines = content.split('\n')

        # Find where to insert (after other imports)
        insert_index = 0
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_char = stripped[:3]
                elif stripped.endswith(docstring_char):
                    in_docstring = False
                    insert_index = i + 1
                continue

            if in_docstring:
                continue

            # Skip comments
            if stripped.startswith('#'):
                insert_index = i + 1
                continue

            # Found an import
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_index = i + 1
                continue

            # Found non-import, non-comment, non-docstring
            if stripped and not stripped.startswith('#'):
                break

        # Insert logger import
        logger_import = 'from logging_config import get_logger, LogContext, operation_context\n'
        logger_creation = f'logger = get_logger(__name__)\n'

        # Insert both lines
        lines.insert(insert_index, '')
        lines.insert(insert_index + 1, logger_import)
        lines.insert(insert_index + 2, logger_creation)

        return '\n'.join(lines)

    def print_stats(self):
        """Print migration statistics"""
        print("\n" + "=" * 60)
        print("Migration Statistics")
        print("=" * 60)
        print(f"Files scanned:         {self.stats['files_scanned']}")
        print(f"Files modified:        {self.stats['files_modified']}")
        print(f"Print statements:      {self.stats['prints_replaced']}")
        print(f"Verbose checks removed:{self.stats['verbose_checks_removed']}")
        print(f"Logger imports added:  {self.stats['imports_added']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Migrate print() to structured logging')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without modifying files')
    parser.add_argument('--file', type=str, help='Migrate specific file only')
    parser.add_argument('--dir', type=str, default='.', help='Directory to scan (default: current)')
    args = parser.parse_args()

    migrator = LoggingMigrator(dry_run=args.dry_run)

    print("=" * 60)
    print("Logging Migration Tool")
    print("=" * 60)
    if args.dry_run:
        print("MODE: DRY RUN (no files will be modified)")
    else:
        print("MODE: LIVE (files will be modified, backups created)")
    print("=" * 60)
    print()

    # Collect files to migrate
    if args.file:
        files = [Path(args.file)]
    else:
        # Find all Python files
        root = Path(args.dir)
        files = []
        for pattern in ['**/*.py']:
            files.extend(root.glob(pattern))

        # Exclude some paths
        exclude_patterns = [
            'venv/', 'env/', '.env/', '__pycache__/',
            'build/', 'dist/', '.git/',
            'migrate_logging.py',  # Don't migrate self
            'logging_config.py'    # Don't migrate logging config
        ]

        files = [
            f for f in files
            if not any(excl in str(f) for excl in exclude_patterns)
        ]

    # Migrate files
    print(f"Found {len(files)} Python files to scan\n")

    for file_path in sorted(files):
        migrator.migrate_file(file_path)

    # Print statistics
    migrator.print_stats()

    if not args.dry_run:
        print("\n✓ Migration complete!")
        print("  - Original files backed up as *.py.bak")
        print("  - Review changes and test thoroughly")
        print("  - Remove .bak files once satisfied: rm **/*.py.bak")
    else:
        print("\n✓ Dry run complete!")
        print("  - Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
