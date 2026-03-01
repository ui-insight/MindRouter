# MindRouter2 Scheduler

## Overview

MindRouter2 implements a **Weighted Deficit Round Robin (WDRR)** scheduler with backend scoring to achieve fair resource allocation across users while maximizing cluster throughput.

## Goals

1. **Maximize token throughput** - Process as many tokens as possible
2. **Minimize token latency** - Keep response times low
3. **Allow bursting** - Single user can use full cluster when idle
4. **Fair rebalancing** - When others arrive, quickly rebalance
5. **Handle heterogeneity** - Different backends, GPUs, models

## Fair-Share Algorithm

### Share Weights

Each user role has a weight that determines their share of resources:

| Role | Weight | Relative Share |
|------|--------|----------------|
| Student | 1 | 1x |
| Staff | 2 | 2x |
| Faculty | 3 | 3x |
| Admin | 10 | 10x |

### Deficit Counters

Each user maintains a **deficit counter** that tracks how much service they're owed:

- **Positive deficit**: User has waited, deserves more service
- **Negative deficit**: User has been served recently

When a job completes, the user's deficit is reduced by the token cost:

```python
deficit -= tokens_used
```

### Priority Calculation

Priority for each pending job is computed as:

```python
priority = (deficit + burst_credits) / weight * deprioritization_factor + wait_bonus
```

Where:
- `deficit`: Current deficit counter
- `burst_credits`: Accumulated credits from idle time
- `weight`: Role-based share weight
- `deprioritization_factor`: Penalty for heavy users (0.1 to 1.0)
- `wait_bonus`: Small bonus for time spent waiting

### Burst Credits

When the cluster is idle, burst credits accumulate for all users:

```python
burst_credits += idle_seconds * credit_rate * weight
```

This allows a single user to consume the full cluster when no one else is waiting. When contention is detected (queue becomes non-empty), burst credits decay:

```python
burst_credits *= 0.5  # Decay when contention appears
```

### Heavy User Deprioritization

Users who consume more than 50% of recent cluster capacity get temporarily deprioritized:

```python
usage_fraction = user_recent_tokens / global_recent_tokens
if usage_fraction > 0.5:
    deprioritization_factor = max(0.1, 1.0 - (usage_fraction - 0.5) * 2)
```

## Backend Scoring

### Hard Constraints

Backends must pass all hard constraints to be considered:

1. **Model Available**: Backend has the requested model
2. **Modality Support**: Backend supports vision/embeddings if needed
3. **Structured Output**: Backend supports JSON mode if requested
4. **Capacity**: `current + queued < max_concurrent`
5. **Memory Fit**: Estimated VRAM ≤ available VRAM

### Soft Scores

Eligible backends are ranked by total score:

| Factor | Points | Description |
|--------|--------|-------------|
| Model Loaded | +100 | Model already in GPU memory |
| Low Utilization | +50 | GPU utilization < 50% |
| Short Queue | +30 | Few pending requests |
| High Throughput | +20 | Fast GPU type |
| Priority | +N×10 | Admin-configured preference |

### Backend Selection

The backend with the highest total score is selected:

```python
selected_backend = argmax(backends, key=lambda b: b.total_score)
```

## Example Scenarios

### Scenario 1: Single User, Idle Cluster

```
Time 0: Student submits request
- Burst credits available
- Routed immediately to best backend
- Full cluster available for burst
```

### Scenario 2: Heavy User, Light User Arrives

```
Time 0: Faculty starts heavy load (many requests)
Time 5: Student submits single request
- Student has higher relative priority (less recent usage)
- Student request gets served quickly
- Faculty requests continue at lower priority
```

### Scenario 3: Multiple Users Competing

```
Faculty (weight=3), Staff (weight=2), Student (weight=1) all active

Distribution over time approaches:
- Faculty: 50% of resources (3/6)
- Staff: 33% of resources (2/6)
- Student: 17% of resources (1/6)
```

## Configuration

### Environment Variables

```bash
# Weights
SCHEDULER_WEIGHT_STUDENT=1
SCHEDULER_WEIGHT_STAFF=2
SCHEDULER_WEIGHT_FACULTY=3
SCHEDULER_WEIGHT_ADMIN=10

# Fairness
SCHEDULER_FAIRNESS_WINDOW=300  # 5 minute rolling window
SCHEDULER_DEPRIORITIZE_THRESHOLD=0.5  # 50%

# Scoring
SCHEDULER_SCORE_MODEL_LOADED=100
SCHEDULER_SCORE_LOW_UTILIZATION=50
SCHEDULER_SCORE_SHORT_QUEUE=30
SCHEDULER_SCORE_HIGH_THROUGHPUT=20
```

## Monitoring

### Queue Statistics

```json
{
  "queue": {
    "total": 5,
    "by_user": {"1": 3, "2": 2},
    "by_model": {"llama3.2": 4, "mistral": 1},
    "average_wait_seconds": 2.5
  },
  "fair_share": {
    "total_users": 2,
    "global_recent_tokens": 50000,
    "user_stats": [
      {"user_id": 1, "weight": 3, "deficit": -1000, "recent_tokens": 30000},
      {"user_id": 2, "weight": 1, "deficit": 500, "recent_tokens": 20000}
    ]
  }
}
```

### Scheduler Decision Records

Each routing decision is logged:

```json
{
  "request_id": "abc123",
  "selected_backend_id": 1,
  "candidate_backends": [1, 2, 3],
  "scores": {"1": 150, "2": 80, "3": 60},
  "user_deficit": -1000,
  "user_weight": 3,
  "user_recent_usage": 30000
}
```
