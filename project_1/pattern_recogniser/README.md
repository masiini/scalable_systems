# Pattern recogniser

## Implementation Details and Parameters

- **Pattern Definition:**  
  Hot path detection uses (configurable) a Kleene closure (min 2, max 5) and a 1-hour window, with a consumption policy to avoid overlapping matches. Patterns are defined separately for classic and electric bikes.

- **Shedding Logic:**  
  Shedding is dynamically controlled based on RabbitMQ queue depth and rolling p95 event processing latency. Shedding levels determine which events are forwarded, prioritizing those with higher utility scores (longer chains, proximity to the 1-hour window, target stations, and likely handoffs).

- **Shedding Parameters:**  
  - Queue thresholds: `HIGH = 10000`, `LOW = 3000`
  - Latency window: 30 seconds, max 5000 samples
  - Utility thresholds: `TAU1 = 0.7`, `TAU2 = 2.5`, `SAMPLE_MED = 0.3`
  - Prefetch: 1000

- **Consumption Policy:**  
  Configured to prevent overlapping matches and reduce state explosion.

- **Monitoring:**  
  Prometheus metrics are used to track latency, queue depth, events forwarded/dropped, and matches found.

- **Design Choices:**  
  The system prioritizes low-latency processing and robustness under bursty workloads, with modular components for easy tuning and extension.