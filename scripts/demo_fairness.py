#!/usr/bin/env python3
############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# demo_fairness.py: Demo script for fair-share scheduling behavior
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Demonstration script for fair-share scheduling in MindRouter2.

This script simulates two users (one heavy, one light) making requests
to show how the fair-share scheduler balances load.
"""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List

# Configuration
BASE_URL = "http://localhost:8000"
MODEL = "llama3.2"  # Change to an available model

# These should be actual API keys from the seeded users
HEAVY_USER_KEY = "your-heavy-user-api-key"  # e.g., faculty user
LIGHT_USER_KEY = "your-light-user-api-key"  # e.g., student user


@dataclass
class RequestResult:
    """Result of a single request."""
    user: str
    request_num: int
    start_time: float
    end_time: float
    success: bool
    queue_delay_ms: int = 0
    total_time_ms: int = 0


async def make_request(
    session: aiohttp.ClientSession,
    api_key: str,
    user_name: str,
    request_num: int,
) -> RequestResult:
    """Make a single chat completion request."""
    start_time = time.time()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"Count from 1 to 10. Request #{request_num}"}
        ],
        "max_tokens": 50,
    }

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            end_time = time.time()
            data = await response.json()

            return RequestResult(
                user=user_name,
                request_num=request_num,
                start_time=start_time,
                end_time=end_time,
                success=response.status == 200,
                total_time_ms=int((end_time - start_time) * 1000),
            )
    except Exception as e:
        end_time = time.time()
        return RequestResult(
            user=user_name,
            request_num=request_num,
            start_time=start_time,
            end_time=end_time,
            success=False,
        )


async def heavy_user_load(session: aiohttp.ClientSession) -> List[RequestResult]:
    """Simulate heavy user making many requests."""
    results = []
    print("Heavy user starting continuous requests...")

    for i in range(20):
        result = await make_request(session, HEAVY_USER_KEY, "heavy", i + 1)
        results.append(result)
        print(f"  Heavy #{i+1}: {result.total_time_ms}ms {'OK' if result.success else 'FAIL'}")
        await asyncio.sleep(0.5)  # Small delay between requests

    return results


async def light_user_burst(session: aiohttp.ClientSession, delay: float) -> List[RequestResult]:
    """Simulate light user arriving after delay and making a few requests."""
    results = []

    # Wait before starting
    await asyncio.sleep(delay)
    print(f"\nLight user arriving after {delay}s delay...")

    for i in range(5):
        result = await make_request(session, LIGHT_USER_KEY, "light", i + 1)
        results.append(result)
        print(f"  Light #{i+1}: {result.total_time_ms}ms {'OK' if result.success else 'FAIL'}")
        await asyncio.sleep(0.3)

    return results


async def run_demo():
    """Run the fairness demonstration."""
    print("=" * 60)
    print("MindRouter2 Fair-Share Scheduling Demo")
    print("=" * 60)
    print()
    print(f"Target: {BASE_URL}")
    print(f"Model: {MODEL}")
    print()

    async with aiohttp.ClientSession() as session:
        # Run heavy and light users concurrently
        heavy_task = asyncio.create_task(heavy_user_load(session))
        light_task = asyncio.create_task(light_user_burst(session, delay=3.0))

        heavy_results, light_results = await asyncio.gather(heavy_task, light_task)

    # Analyze results
    print()
    print("=" * 60)
    print("Results Analysis")
    print("=" * 60)

    heavy_times = [r.total_time_ms for r in heavy_results if r.success]
    light_times = [r.total_time_ms for r in light_results if r.success]

    print(f"\nHeavy user:")
    print(f"  Successful requests: {len(heavy_times)}/{len(heavy_results)}")
    if heavy_times:
        print(f"  Average latency: {sum(heavy_times)/len(heavy_times):.0f}ms")
        print(f"  Min/Max: {min(heavy_times)}/{max(heavy_times)}ms")

    print(f"\nLight user:")
    print(f"  Successful requests: {len(light_times)}/{len(light_results)}")
    if light_times:
        print(f"  Average latency: {sum(light_times)/len(light_times):.0f}ms")
        print(f"  Min/Max: {min(light_times)}/{max(light_times)}ms")

    # Check fairness
    if heavy_times and light_times:
        heavy_avg = sum(heavy_times) / len(heavy_times)
        light_avg = sum(light_times) / len(light_times)

        print()
        if light_avg < heavy_avg * 1.5:
            print("FAIR: Light user got reasonable latency despite heavy user load")
        else:
            print("UNFAIR: Light user experienced significantly higher latency")


def main():
    """Main entry point."""
    print("Before running this demo:")
    print("1. Start MindRouter2: docker compose up")
    print("2. Register at least one backend with the model")
    print("3. Update HEAVY_USER_KEY and LIGHT_USER_KEY in this script")
    print()
    input("Press Enter to continue...")

    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
