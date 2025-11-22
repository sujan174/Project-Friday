#!/usr/bin/env python3
"""
Orchestrator Testing System

Runs multiple test sessions in parallel, with queries processed sequentially
within each session. Results are saved to the output directory.

Usage:
    python testing/run_tests.py                    # Run all tests
    python testing/run_tests.py --parallel 3       # Limit concurrent sessions
    python testing/run_tests.py --verbose          # Show detailed output
    python testing/run_tests.py --input test1.txt  # Run specific test file

Input Format:
    Each .txt file in testing/input/ contains one query per line.
    Empty lines and lines starting with # are ignored.

Output Format:
    JSON files in testing/output/ with results for each input file.
"""

import asyncio
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import OrchestratorAgent


class TestSession:
    """Manages a single test session with an orchestrator instance."""

    def __init__(self, input_file: Path, output_dir: Path, verbose: bool = False):
        self.input_file = input_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.session_name = input_file.stem
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.results: List[Dict[str, Any]] = []
        self.start_time: float = 0
        self.end_time: float = 0

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{self.session_name}] {message}")

    def _parse_queries(self) -> List[str]:
        """Parse queries from input file."""
        queries = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                queries.append(line)
        return queries

    async def initialize(self):
        """Initialize the orchestrator for this session."""
        self._log("Initializing orchestrator...")

        self.orchestrator = OrchestratorAgent(
            connectors_dir="connectors",
            verbose=False  # Keep orchestrator quiet
        )

        # Discover and load agents
        await self.orchestrator.discover_and_load_agents()

        # Count loaded agents
        loaded_count = sum(
            1 for health in self.orchestrator.agent_health.values()
            if health['status'] == 'healthy'
        )
        self._log(f"Loaded {loaded_count} agents")

    async def run(self) -> Dict[str, Any]:
        """Run all queries in this session sequentially."""
        self.start_time = time.time()

        try:
            # Initialize orchestrator
            await self.initialize()

            # Parse queries
            queries = self._parse_queries()
            self._log(f"Processing {len(queries)} queries...")

            # Process each query sequentially
            for i, query in enumerate(queries, 1):
                self._log(f"Query {i}/{len(queries)}: {query[:50]}...")

                query_start = time.time()
                success = True
                response = ""
                error = None

                tokens = {'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                try:
                    response = await self.orchestrator.process_message(query)
                    # Get token usage for this message
                    tokens = self.orchestrator.get_last_message_tokens()
                except Exception as e:
                    success = False
                    error = str(e)
                    response = f"ERROR: {e}"
                    if self.verbose:
                        traceback.print_exc()

                query_end = time.time()

                # Record result
                result = {
                    "query_number": i,
                    "query": query,
                    "response": response,
                    "success": success,
                    "error": error,
                    "duration_seconds": round(query_end - query_start, 3),
                    "timestamp": datetime.now().isoformat(),
                    "tokens": tokens
                }
                self.results.append(result)

                if success:
                    token_info = f", {tokens['total_tokens']} tokens" if tokens['total_tokens'] > 0 else ""
                    self._log(f"  -> Success ({result['duration_seconds']}s{token_info})")
                else:
                    self._log(f"  -> FAILED: {error}")

            self.end_time = time.time()

        except Exception as e:
            self.end_time = time.time()
            self._log(f"Session failed: {e}")
            if self.verbose:
                traceback.print_exc()
            raise

        finally:
            # Cleanup orchestrator
            if self.orchestrator:
                try:
                    await self.orchestrator.cleanup()
                except Exception as e:
                    self._log(f"Cleanup warning: {e}")

        return self._build_report()

    def _build_report(self) -> Dict[str, Any]:
        """Build the final test report."""
        total_queries = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total_queries - successful
        total_duration = self.end_time - self.start_time

        # Calculate total tokens
        total_tokens = {
            'prompt_tokens': sum(r.get('tokens', {}).get('prompt_tokens', 0) for r in self.results),
            'cached_tokens': sum(r.get('tokens', {}).get('cached_tokens', 0) for r in self.results),
            'completion_tokens': sum(r.get('tokens', {}).get('completion_tokens', 0) for r in self.results),
            'total_tokens': sum(r.get('tokens', {}).get('total_tokens', 0) for r in self.results)
        }

        return {
            "test_name": self.session_name,
            "input_file": str(self.input_file),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "total_duration_seconds": round(total_duration, 3),
            "summary": {
                "total_queries": total_queries,
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful/total_queries*100):.1f}%" if total_queries > 0 else "N/A",
                "total_tokens": total_tokens
            },
            "results": self.results
        }

    def save_results(self, report: Dict[str, Any]):
        """Save results to output file."""
        output_file = self.output_dir / f"{self.session_name}_output.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._log(f"Results saved to {output_file}")
        return output_file


class TestRunner:
    """Coordinates running multiple test sessions in parallel."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        max_parallel: int = 5,
        verbose: bool = False
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.verbose = verbose
        self.semaphore = asyncio.Semaphore(max_parallel)

    def discover_tests(self, specific_file: Optional[str] = None) -> List[Path]:
        """Discover test input files."""
        if specific_file:
            # Run specific test file
            test_file = self.input_dir / specific_file
            if not test_file.exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
            return [test_file]

        # Discover all .txt files in input directory
        test_files = sorted(self.input_dir.glob("*.txt"))
        return test_files

    async def run_session_with_semaphore(self, session: TestSession) -> Dict[str, Any]:
        """Run a test session with concurrency control."""
        async with self.semaphore:
            report = await session.run()
            session.save_results(report)
            return report

    async def run_all(self, specific_file: Optional[str] = None) -> Dict[str, Any]:
        """Run all test sessions in parallel."""
        start_time = time.time()

        # Discover test files
        test_files = self.discover_tests(specific_file)

        if not test_files:
            print("No test files found in input directory.")
            return {"error": "No test files found"}

        print(f"\n{'='*60}")
        print(f"  ORCHESTRATOR TEST RUNNER")
        print(f"{'='*60}")
        print(f"  Test files: {len(test_files)}")
        print(f"  Max parallel: {self.max_parallel}")
        print(f"  Output dir: {self.output_dir}")
        print(f"{'='*60}\n")

        # Create sessions
        sessions = [
            TestSession(test_file, self.output_dir, self.verbose)
            for test_file in test_files
        ]

        # Run all sessions in parallel
        print(f"Starting {len(sessions)} test session(s)...\n")

        tasks = [
            self.run_session_with_semaphore(session)
            for session in sessions
        ]

        reports = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # Build summary
        successful_reports = []
        failed_sessions = []

        for i, report in enumerate(reports):
            if isinstance(report, Exception):
                failed_sessions.append({
                    "test_file": str(test_files[i]),
                    "error": str(report)
                })
            else:
                successful_reports.append(report)

        # Calculate totals
        total_queries = sum(r['summary']['total_queries'] for r in successful_reports)
        total_successful = sum(r['summary']['successful'] for r in successful_reports)
        total_failed = sum(r['summary']['failed'] for r in successful_reports)

        # Calculate total tokens across all sessions
        total_tokens = {
            'prompt_tokens': sum(r['summary']['total_tokens']['prompt_tokens'] for r in successful_reports),
            'cached_tokens': sum(r['summary']['total_tokens'].get('cached_tokens', 0) for r in successful_reports),
            'completion_tokens': sum(r['summary']['total_tokens']['completion_tokens'] for r in successful_reports),
            'total_tokens': sum(r['summary']['total_tokens']['total_tokens'] for r in successful_reports)
        }

        # Calculate estimated cost (Gemini 2.5 Flash pricing)
        # Cached input: $0.01875/1M, Standard input: $0.075/1M, Output: $0.30/1M
        uncached_input = total_tokens['prompt_tokens'] - total_tokens['cached_tokens']
        cost = {
            'cached_input_cost': (total_tokens['cached_tokens'] / 1_000_000) * 0.01875,
            'standard_input_cost': (uncached_input / 1_000_000) * 0.075,
            'output_cost': (total_tokens['completion_tokens'] / 1_000_000) * 0.30,
        }
        cost['total_cost'] = cost['cached_input_cost'] + cost['standard_input_cost'] + cost['output_cost']

        summary = {
            "run_time": datetime.now().isoformat(),
            "total_duration_seconds": round(end_time - start_time, 3),
            "sessions": {
                "total": len(sessions),
                "completed": len(successful_reports),
                "failed": len(failed_sessions)
            },
            "queries": {
                "total": total_queries,
                "successful": total_successful,
                "failed": total_failed,
                "success_rate": f"{(total_successful/total_queries*100):.1f}%" if total_queries > 0 else "N/A"
            },
            "tokens": total_tokens,
            "cost": {k: round(v, 6) for k, v in cost.items()},
            "failed_sessions": failed_sessions
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"  TEST RUN COMPLETE")
        print(f"{'='*60}")
        print(f"  Duration: {summary['total_duration_seconds']}s")
        print(f"  Sessions: {summary['sessions']['completed']}/{summary['sessions']['total']} completed")
        print(f"  Queries: {summary['queries']['successful']}/{summary['queries']['total']} successful")
        print(f"  Success Rate: {summary['queries']['success_rate']}")
        print(f"  Tokens: {total_tokens['total_tokens']} total")
        print(f"    - Prompt: {total_tokens['prompt_tokens']} (cached: {total_tokens['cached_tokens']})")
        print(f"    - Completion: {total_tokens['completion_tokens']}")
        print(f"  Estimated Cost: ${cost['total_cost']:.4f}")

        if failed_sessions:
            print(f"\n  Failed Sessions:")
            for fs in failed_sessions:
                print(f"    - {Path(fs['test_file']).name}: {fs['error']}")

        print(f"{'='*60}\n")

        # Save summary
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run orchestrator tests with parallel sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python testing/run_tests.py                    # Run all tests
    python testing/run_tests.py --parallel 3       # Limit to 3 concurrent sessions
    python testing/run_tests.py --verbose          # Show detailed output
    python testing/run_tests.py --input test1.txt  # Run specific test
        """
    )

    parser.add_argument(
        '--input', '-i',
        help='Specific input file to test (filename only, not path)',
        default=None
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=5,
        help='Maximum number of parallel sessions (default: 5)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress output'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=PROJECT_ROOT / 'testing' / 'input',
        help='Input directory containing test files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'testing' / 'output',
        help='Output directory for test results'
    )

    args = parser.parse_args()

    # Ensure directories exist
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create runner
    runner = TestRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_parallel=args.parallel,
        verbose=args.verbose
    )

    # Run tests
    try:
        summary = asyncio.run(runner.run_all(args.input))

        # Exit with error code if there were failures
        if summary.get('sessions', {}).get('failed', 0) > 0:
            sys.exit(1)
        if summary.get('queries', {}).get('failed', 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest runner failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
