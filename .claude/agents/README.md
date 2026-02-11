# Mamba Symmetry Experiment - Agent Team

This project uses a multi-agent system for reproducing the Mamba symmetry bias experiments.

## Available Agents

### 1. data-engineer
**Purpose**: Implements synthetic datasets for experiments
**Use when**: Creating CompositeFunctionDataset or InverseSequenceDataset
**Tools**: Read, Write, Edit, Bash, Grep, Glob

### 2. model-engineer
**Purpose**: Builds Mamba model wrappers and architectural variants
**Use when**: Implementing or modifying neural network models
**Tools**: Read, Write, Edit, Bash, Grep, Glob

### 3. experiment-runner
**Purpose**: Executes training jobs and manages hyperparameter sweeps
**Use when**: Running phase diagrams, ablations, or training experiments
**Tools**: Read, Write, Bash, Glob, Grep

### 4. visualizer
**Purpose**: Creates scientific visualizations and plots
**Use when**: Generating figures after experiments complete
**Tools**: Read, Write, Edit, Bash, Grep, Glob

### 5. analyst
**Purpose**: Analyzes results and writes scientific reports
**Use when**: Computing statistics and comparing to paper baselines
**Tools**: Read, Write, Grep, Glob, Bash

### 6. debugger
**Purpose**: Debugs code errors and training failures
**Use when**: Encountering any issues or unexpected behavior
**Tools**: Read, Edit, Bash, Grep, Glob

## How to Use

### Method 1: Explicit Request
```
Use the data-engineer agent to implement CompositeFunctionDataset
```

### Method 2: Let Claude Decide
Just describe what you need, Claude will automatically delegate to the appropriate agent based on the task.

## Workflow Example

```
1. data-engineer → Implement datasets
2. model-engineer → Build Mamba wrappers
3. experiment-runner → Run phase diagram (90 runs)
4. visualizer → Generate heatmaps and plots
5. analyst → Write comparison report
6. debugger → (as needed for any issues)
```

## Agent Coordination

Agents share work through:
- **Code**: `src/` directory (data, models, training scripts)
- **Results**: `results/` directory (organized by experiment)
- **Reports**: Generated markdown files
- **Visualizations**: PNG figures in `results/`

## Checking Agent Status

View all agents:
```
/agents
```

## Notes

- All agents use `model: sonnet` for high-quality work
- Each agent has limited tool access for safety
- Agents automatically track their work in commit messages
- Phase diagram experiment: ~90 runs × 210 epochs ≈ 1-2 days GPU time
