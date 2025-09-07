#!/usr/bin/env python3
"""
Test script to demonstrate time-based batching with early retry handling.

This shows how the new system:
1. Sends batches every 60s regardless of API response time
2. Handles early failures and adds them to next batch immediately  
3. Optimizes throughput when APIs are slow
"""

import asyncio
import time
import random
from utils.smart_queue import SmartQueue

async def mock_slow_executor(experiment_pair):
    """
    Mock executor that simulates variable API response times and failures.
    
    This helps demonstrate the optimization:
    - Some calls return quickly (30s) 
    - Some calls are slow (90s)
    - Some calls fail and need retry
    """
    category, variable = experiment_pair
    task_id = f"{category}:{variable}"
    
    # Simulate variable response times
    if "fast" in variable:
        response_time = random.uniform(10, 30)  # Fast responses
    elif "slow" in variable:
        response_time = random.uniform(70, 120)  # Slow responses  
    else:
        response_time = random.uniform(30, 90)  # Mixed speeds
    
    print(f"🕐 {task_id} - Starting (will take {response_time:.1f}s)")
    await asyncio.sleep(response_time)
    
    # Simulate failure rate
    if random.random() < 0.2:  # 20% failure rate
        print(f"❌ {task_id} - Failed after {response_time:.1f}s")
        raise Exception(f"Mock failure for {task_id}")
    
    print(f"✅ {task_id} - Completed after {response_time:.1f}s")
    return f"Success: {task_id}"

async def test_time_based_batching():
    """Demonstrate the time-based batching optimization"""
    
    print("🧪 TESTING TIME-BASED BATCHING")
    print("=" * 60)
    print("This test simulates:")
    print("📤 Batches sent every 60s (regardless of completion time)")
    print("⚡ Fast failures added to next batch immediately")
    print("🐌 Slow APIs don't block subsequent batches")
    print("=" * 60)
    
    # Create smart queue with small rate limit for testing
    queue = SmartQueue(rate_limit=5, logger=None)
    
    # Add tasks with different expected speeds
    test_experiments = [
        # Fast tasks (should complete in ~30s)
        ("model_a", [("speed", "fast_1"), ("speed", "fast_2"), ("speed", "fast_3")]),
        
        # Mixed speed tasks  
        ("model_b", [("speed", "mixed_1"), ("speed", "mixed_2"), ("speed", "mixed_3")]),
        
        # Slow tasks (will take 70-120s)
        ("model_c", [("speed", "slow_1"), ("speed", "slow_2"), ("speed", "slow_3")]),
        
        # More mixed tasks to fill multiple batches
        ("model_d", [("speed", "mixed_4"), ("speed", "mixed_5")]),
    ]
    
    # Add all experiments to queue
    for model_name, experiments in test_experiments:
        queue.add_model_tasks(model_name, experiments, mock_slow_executor)
        print(f"➕ Added {len(experiments)} tasks for {model_name}")
    
    print(f"\n📊 Total tasks: {queue.get_queue_status()['total_pending']}")
    print("🚀 Starting time-based processing...\n")
    
    # Process with time-based batching
    start_time = time.time()
    results = await queue.process_all()
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 RESULTS ANALYSIS")
    print("=" * 60)
    
    stats = results['stats']
    
    print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📦 Batches Processed: {stats['batches_processed']}")
    print(f"✅ Successful Tasks: {stats['completed_tasks']}")
    print(f"❌ Failed Tasks: {stats['failed_tasks']}")
    print(f"🔄 Retry Tasks: {stats['retry_tasks']}")
    print(f"📡 Total API Calls: {stats['total_api_calls']}")
    print(f"⚡ Efficiency: {stats['efficiency_percent']:.1f}%")
    
    # Calculate theoretical vs actual time
    total_tasks = stats['completed_tasks'] + stats['failed_tasks']
    old_system_time = (total_tasks / 5) * 60  # Old system: wait for each batch
    time_saved = max(0, old_system_time - total_time)
    
    print(f"\n🚀 OPTIMIZATION IMPACT:")
    print(f"   Old system (wait for completion): ~{old_system_time/60:.1f} minutes")
    print(f"   New system (time-based): {total_time/60:.1f} minutes") 
    print(f"   Time saved: {time_saved/60:.1f} minutes ({time_saved/old_system_time*100:.1f}% improvement)")
    
    # Show batch timing details
    print(f"\n📈 BATCHING DETAILS:")
    print(f"   Average calls per batch: {stats['total_api_calls']/stats['batches_processed']:.1f}")
    print(f"   Wasted slots: {stats['wasted_slots']}")
    print(f"   Rate limit utilization: {(stats['total_api_calls']/(stats['batches_processed']*5))*100:.1f}%")

async def run_comparison():
    """Run a quick comparison between old and new systems"""
    print("\n" + "=" * 60)
    print("🔬 COMPARISON: Old vs New Batching")
    print("=" * 60)
    
    print("📈 Expected improvements:")
    print("   • Slow API calls (>60s): ~40-60% faster")
    print("   • Fast failures: Immediate retry in next batch")  
    print("   • Mixed workloads: Better resource utilization")
    print("   • Multi-model runs: Optimal cross-model batching")
    
    print("\n🎯 Key scenarios where this shines:")
    print("   • APIs with variable response times")
    print("   • Workloads with retry-able failures")  
    print("   • Multiple models with different queue sizes")
    print("   • Long-running experiment batches")

if __name__ == "__main__":
    print("🧪 Time-Based Batching Test Suite")
    print("=" * 60)
    
    asyncio.run(test_time_based_batching())
    asyncio.run(run_comparison())