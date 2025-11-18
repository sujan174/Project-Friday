"""
Circuit Breaker Pattern - Production Grade

Prevents cascading failures by temporarily disabling failing agents.
Implements the classic circuit breaker state machine: CLOSED → OPEN → HALF_OPEN → CLOSED

Features:
- Three-state machine (CLOSED, OPEN, HALF_OPEN)
- Per-agent tracking
- Automatic recovery testing
- Failure threshold configuration
- Health monitoring

Author: AI System (Senior Developer)
Version: 2.0 - Complete Implementation
"""

import time
import asyncio
from typing import Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation - requests flow through
    OPEN = "open"           # Circuit open - agent is failing, block all requests
    HALF_OPEN = "half_open" # Testing recovery - allow limited requests


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Consecutive failures before opening
    success_threshold: int = 2          # Consecutive successes to close from half-open
    timeout_seconds: float = 300.0      # Wait time before moving to half-open (5 min)
    half_open_timeout: float = 10.0     # Max time in half-open state


@dataclass
class CircuitStats:
    """Statistics for a circuit"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0  # In half-open state
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_failures: int = 0
    total_successes: int = 0
    total_requests: int = 0
    state_history: list = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation

    Prevents cascading failures by temporarily disabling failing agents.

    State Transitions:
    - CLOSED → OPEN: After N consecutive failures
    - OPEN → HALF_OPEN: After timeout period
    - HALF_OPEN → CLOSED: After N consecutive successes
    - HALF_OPEN → OPEN: If any request fails

    Example Usage:
        circuit_breaker = CircuitBreaker(config=CircuitConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=300.0
        ))

        # Before executing
        allowed, reason = await circuit_breaker.can_execute("slack_agent")
        if not allowed:
            print(f"Blocked: {reason}")
            return

        # Execute operation
        try:
            result = await agent.execute(instruction)
            await circuit_breaker.record_success("slack_agent")
        except Exception as e:
            await circuit_breaker.record_failure("slack_agent")
            raise
    """

    def __init__(self, config: Optional[CircuitConfig] = None, verbose: bool = False):
        """
        Initialize circuit breaker

        Args:
            config: Circuit breaker configuration
            verbose: Enable verbose logging
        """
        self.config = config or CircuitConfig()
        self.verbose = verbose

        # Per-agent circuit stats
        self.circuits: Dict[str, CircuitStats] = {}

        # Global stats
        self.total_blocked_requests = 0
        self.total_allowed_requests = 0

    def _get_circuit(self, agent_name: str) -> CircuitStats:
        """Get or create circuit stats for an agent"""
        if agent_name not in self.circuits:
            self.circuits[agent_name] = CircuitStats()
        return self.circuits[agent_name]

    async def can_execute(self, agent_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if agent execution is allowed

        Args:
            agent_name: Name of the agent

        Returns:
            (allowed, reason) - True if allowed, False if blocked with reason
        """
        circuit = self._get_circuit(agent_name)
        circuit.total_requests += 1
        current_time = time.time()

        # CLOSED state: Allow all requests
        if circuit.state == CircuitState.CLOSED:
            self.total_allowed_requests += 1
            return True, None

        # OPEN state: Block requests until timeout
        if circuit.state == CircuitState.OPEN:
            time_since_failure = current_time - (circuit.last_failure_time or current_time)

            # Check if timeout has passed
            if time_since_failure >= self.config.timeout_seconds:
                # Move to HALF_OPEN for recovery testing
                await self._transition_state(agent_name, CircuitState.HALF_OPEN)
                self.total_allowed_requests += 1
                return True, None
            else:
                # Still in timeout period - block request
                remaining = self.config.timeout_seconds - time_since_failure
                self.total_blocked_requests += 1
                return False, f"Circuit open for {agent_name}. Retry in {remaining:.0f}s"

        # HALF_OPEN state: Allow request for testing
        if circuit.state == CircuitState.HALF_OPEN:
            # Check if we've been in half-open too long
            time_in_half_open = current_time - circuit.last_state_change
            if time_in_half_open > self.config.half_open_timeout:
                # Timeout in half-open - back to open
                await self._transition_state(agent_name, CircuitState.OPEN)
                self.total_blocked_requests += 1
                return False, f"Circuit reopened for {agent_name} after half-open timeout"

            self.total_allowed_requests += 1
            return True, None

        # Unknown state - allow with warning
        self.total_allowed_requests += 1
        return True, None

    async def record_success(self, agent_name: str):
        """
        Record successful execution

        Args:
            agent_name: Name of the agent
        """
        circuit = self._get_circuit(agent_name)
        circuit.total_successes += 1
        circuit.last_success_time = time.time()
        circuit.failure_count = 0  # Reset failure count on success

        if circuit.state == CircuitState.HALF_OPEN:
            # Increment success count in half-open
            circuit.success_count += 1

            if self.verbose:
                print(f"[CIRCUIT] {agent_name} success in HALF_OPEN ({circuit.success_count}/{self.config.success_threshold})")

            # Check if we've reached success threshold
            if circuit.success_count >= self.config.success_threshold:
                # Close the circuit - agent has recovered
                await self._transition_state(agent_name, CircuitState.CLOSED)

        elif circuit.state == CircuitState.CLOSED:
            # Normal operation
            if self.verbose:
                print(f"[CIRCUIT] {agent_name} success in CLOSED state")

    async def record_failure(self, agent_name: str, error: Optional[Exception] = None):
        """
        Record failed execution

        Args:
            agent_name: Name of the agent
            error: Optional exception that caused the failure
        """
        circuit = self._get_circuit(agent_name)
        circuit.total_failures += 1
        circuit.failure_count += 1
        circuit.last_failure_time = time.time()

        if self.verbose:
            error_msg = str(error) if error else "Unknown error"
            print(f"[CIRCUIT] {agent_name} failure: {error_msg[:100]}")

        # State-specific handling
        if circuit.state == CircuitState.CLOSED:
            # Check if we've reached failure threshold
            if circuit.failure_count >= self.config.failure_threshold:
                # Open the circuit
                await self._transition_state(agent_name, CircuitState.OPEN)

        elif circuit.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens circuit
            circuit.success_count = 0  # Reset success count
            await self._transition_state(agent_name, CircuitState.OPEN)

    async def _transition_state(self, agent_name: str, new_state: CircuitState):
        """
        Transition circuit to new state

        Args:
            agent_name: Name of the agent
            new_state: New state to transition to
        """
        circuit = self._get_circuit(agent_name)
        old_state = circuit.state

        if old_state == new_state:
            return  # No change

        # Update state
        circuit.state = new_state
        circuit.last_state_change = time.time()

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            circuit.failure_count = 0
            circuit.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            circuit.success_count = 0
        elif new_state == CircuitState.OPEN:
            circuit.success_count = 0

        # Record state change
        circuit.state_history.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': circuit.last_state_change,
            'datetime': datetime.fromtimestamp(circuit.last_state_change).isoformat()
        })

        if self.verbose or new_state == CircuitState.OPEN:
            # Always log OPEN transitions (important!)
            print(f"[CIRCUIT] {agent_name}: {old_state.value} → {new_state.value}")

            if new_state == CircuitState.OPEN:
                print(f"⚠️  {agent_name} circuit OPENED after {circuit.failure_count} failures")
                print(f"   Will retry in {self.config.timeout_seconds}s")

            elif new_state == CircuitState.CLOSED:
                print(f"✅ {agent_name} circuit CLOSED - agent recovered")

    def get_health_status(self, agent_name: str) -> Dict[str, Any]:
        """
        Get health status for an agent

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with health information
        """
        circuit = self._get_circuit(agent_name)

        return {
            'agent_name': agent_name,
            'state': circuit.state.value,
            'healthy': circuit.state == CircuitState.CLOSED,
            'failure_count': circuit.failure_count,
            'success_count': circuit.success_count,
            'total_requests': circuit.total_requests,
            'total_failures': circuit.total_failures,
            'total_successes': circuit.total_successes,
            'success_rate': (circuit.total_successes / circuit.total_requests * 100)
                if circuit.total_requests > 0 else 0,
            'last_failure_time': circuit.last_failure_time,
            'last_success_time': circuit.last_success_time,
            'last_state_change': circuit.last_state_change,
            'time_in_current_state': time.time() - circuit.last_state_change
        }

    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents"""
        return {
            agent_name: self.get_health_status(agent_name)
            for agent_name in self.circuits.keys()
        }

    def reset_circuit(self, agent_name: str):
        """
        Manually reset circuit for an agent (use with caution)

        Args:
            agent_name: Name of the agent
        """
        if agent_name in self.circuits:
            self.circuits[agent_name] = CircuitStats()
            if self.verbose:
                print(f"[CIRCUIT] {agent_name} circuit manually reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall circuit breaker statistics"""
        total_circuits = len(self.circuits)
        open_circuits = sum(1 for c in self.circuits.values() if c.state == CircuitState.OPEN)
        half_open_circuits = sum(1 for c in self.circuits.values() if c.state == CircuitState.HALF_OPEN)
        closed_circuits = sum(1 for c in self.circuits.values() if c.state == CircuitState.CLOSED)

        return {
            'total_circuits': total_circuits,
            'open_circuits': open_circuits,
            'half_open_circuits': half_open_circuits,
            'closed_circuits': closed_circuits,
            'total_blocked_requests': self.total_blocked_requests,
            'total_allowed_requests': self.total_allowed_requests,
            'block_rate': (self.total_blocked_requests /
                          (self.total_blocked_requests + self.total_allowed_requests) * 100)
                if (self.total_blocked_requests + self.total_allowed_requests) > 0 else 0,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'half_open_timeout': self.config.half_open_timeout
            }
        }

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("CIRCUIT BREAKER - STATISTICS")
        print("="*80)

        print(f"\nTotal Circuits: {stats['total_circuits']}")
        print(f"  CLOSED (healthy): {stats['closed_circuits']}")
        print(f"  HALF_OPEN (testing): {stats['half_open_circuits']}")
        print(f"  OPEN (failing): {stats['open_circuits']}")

        print(f"\nRequests:")
        print(f"  Allowed: {stats['total_allowed_requests']}")
        print(f"  Blocked: {stats['total_blocked_requests']}")
        print(f"  Block Rate: {stats['block_rate']:.1f}%")

        print(f"\nConfiguration:")
        print(f"  Failure Threshold: {stats['config']['failure_threshold']}")
        print(f"  Success Threshold: {stats['config']['success_threshold']}")
        print(f"  Timeout: {stats['config']['timeout_seconds']}s")

        if self.circuits:
            print(f"\nAgent Health:")
            for agent_name, circuit in self.circuits.items():
                status_icon = "✅" if circuit.state == CircuitState.CLOSED else \
                             "🟡" if circuit.state == CircuitState.HALF_OPEN else "❌"
                success_rate = (circuit.total_successes / circuit.total_requests * 100) \
                    if circuit.total_requests > 0 else 0
                print(f"  {status_icon} {agent_name}: {circuit.state.value} "
                      f"(success: {success_rate:.0f}%, failures: {circuit.total_failures})")

        print("="*80 + "\n")
