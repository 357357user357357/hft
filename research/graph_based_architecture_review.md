# Graph-Based Architecture for HFT Systems

## Overview

This document explores implementing a graph-based architecture for the HFT backtesting system, inspired by:
- **Obsidian.md** - Knowledge graph with bidirectional links
- **Neo4j** - Graph database with Cypher query language
- **Apache Groovy/Grails** - Convention over configuration
- **Scala/Akka** - Distributed, actor-based systems

---

## 1. Current Architecture (Scalar/Polar)

```
Price Data → Feature Extraction → Signal Generation → Order Execution
     ↓              ↓                    ↓                  ↓
  Scalar        Polar (r,θ)         Rule-based         Market
  + Polar       + Regime            + ML-ready          Orders
```

**Limitations:**
- Linear pipeline (no feedback loops)
- No explicit relationship tracking between signals
- Hard to trace "why" a trade was executed
- No distributed execution model

---

## 2. Graph-Based Architecture Concepts

### 2.1 Core Graph Model

```python
# Nodes
- PriceNode(timestamp, symbol, price)
- FeatureNode(name, value, confidence)
- SignalNode(type, strength, regime)
- TradeNode(side, size, entry, exit)
- RegimeNode(type, start_time, end_time)

# Edges
- PriceNode --[HAS_FEATURE]--> FeatureNode
- FeatureNode --[TRIGGERS]--> SignalNode
- SignalNode --[EXECUTED_AS]--> TradeNode
- TradeNode --[OCCURRED_IN]--> RegimeNode
- SignalNode --[CORRELATED_WITH]--> SignalNode
```

### 2.2 Python Graph Libraries

| Library | Use Case | Pros | Cons |
|---------|----------|------|------|
| **NetworkX** | In-memory graphs | Simple, pure Python | Not distributed, slow for large graphs |
| **igraph** | Fast graph algorithms | C backend, very fast | Less Pythonic API |
| **neo4j** | Persistent graph DB | ACID, Cypher queries | External dependency, network latency |
| **DGL/PyG** | Graph neural networks | ML-ready, GPU support | Overkill for simple relationships |
| **redisgraph** | In-memory graph DB | Fast, Redis ecosystem | Limited query complexity |

---

## 3. Proposed Architecture

### 3.1 Hybrid Approach (Python + Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Stream (Kafka/RabbitMQ)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Feature Extractors (Polar, Fel, Technical)                 │
│       ↓              ↓              ↓                        │
│  FeatureNode ←→ FeatureNode ←→ FeatureNode                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Signal Generator (Graph Pattern Matching)                  │
│       "Find subgraphs that match historical patterns"       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Execution Engine (Actor Model - Ray/Akka)                  │
│       Distributed order management                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Graph Schema Design

```cypher
// Neo4j-style schema
CREATE CONSTRAINT ON (p:PriceNode) ASSERT p.id IS UNIQUE;
CREATE CONSTRAINT ON (s:SignalNode) ASSERT s.id IS UNIQUE;
CREATE CONSTRAINT ON (t:TradeNode) ASSERT t.id IS UNIQUE;

// Index for fast temporal queries
CREATE INDEX ON :PriceNode(timestamp);
CREATE INDEX ON :SignalNode(signal_type);
```

### 3.3 Example: Pattern Matching for Mean Reversion

```cypher
// Find mean-reversion patterns in recent data
MATCH (p1:PriceNode)-[:HAS_FEATURE]->(f1:PolarFeature {theta: r})
MATCH (p2:PriceNode)-[:HAS_FEATURE]->(f2:PolarFeature {theta: s})
WHERE p1.timestamp < p2.timestamp
  AND p2.timestamp - p1.timestamp < 1000000000  // 1 second
  AND abs(f1.theta - f2.theta) > 2.5  // Large theta change
  AND f1.dr_dt < 0 AND f2.dr_dt > 0   // R contraction → expansion
RETURN p1, p2, f1, f2
ORDER BY p2.timestamp DESC
LIMIT 1
```

---

## 4. Distributed Execution (Scala/Akka Model)

### 4.1 Actor Architecture

```scala
// Scala-style pseudocode
class FeatureExtractorActor extends Actor {
  def receive = {
    case PriceUpdate(symbol, price, ts) =>
      val features = extractPolar(price)
      val graphUpdate = GraphUpdate(features)
      graphDbActor ! graphUpdate
  }
}

class SignalGeneratorActor extends Actor {
  def receive = {
    case GraphUpdated =>
      val pattern = matchPatterns(graphDb)
      if (pattern.isDefined) {
        executionActor ! TradeSignal(pattern.get)
      }
  }
}

class ExecutionActor extends Actor {
  def receive = {
    case TradeSignal(pattern) =>
      if (riskCheck(pattern)) {
        placeOrder(pattern)
      }
  }
}
```

### 4.2 Python Equivalent (Ray)

```python
import ray

@ray.remote
class FeatureExtractor:
    def __init__(self, graph_db):
        self.graph = graph_db

    def process_price(self, price_data):
        features = extract_polar(price_data.price)
        self.graph.add_features(price_data.timestamp, features)
        return features

@ray.remote
class SignalGenerator:
    def __init__(self, graph_db):
        self.graph = graph_db

    def check_signals(self):
        pattern = self.graph.match_pattern("mean_reversion")
        if pattern:
            return Signal(pattern)
        return None

# Distributed execution
extractor = FeatureExtractor.remote(graph_db)
signal_gen = SignalGenerator.remote(graph_db)

ray.get(extractor.process_price.remote(price_data))
signal = ray.get(signal_gen.check_signals.remote())
```

---

## 5. Obsidian-Style Knowledge Graph

### 5.1 Bidirectional Linking

```python
# Trade relationships
class TradeGraph:
    def __init__(self):
        self.nodes = {}  # id -> Node
        self.edges = []  # (from_id, to_id, relation)

    def link(self, source_id, target_id, relation):
        """Create bidirectional link (Obsidian-style)"""
        self.edges.append((source_id, target_id, relation))
        self.edges.append((target_id, source_id, f"inverse_{relation}"))

    def query_backlinks(self, node_id):
        """Find all nodes linking to this node"""
        return [
            (src, rel) for src, dst, rel in self.edges
            if dst == node_id
        ]
```

### 5.2 Trade Analysis Example

```python
# Query: "Show me all trades that occurred after polar mean-revert signals"
trades = graph.query("""
    MATCH (s:SignalNode {type: 'MEAN_REVERT_LONG'})
          -[r:TRIGGERED]->(t:TradeNode)
    WHERE t.exit_time > $start_date
    RETURN t, t.pnl_pct, r.confidence
    ORDER BY t.pnl_pct DESC
""")

# Query: "What regime produces the best mean-reversion trades?"
regime_performance = graph.query("""
    MATCH (s:SignalNode {type: 'MEAN_REVERT_LONG'})
          -[:OCCURRED_IN]->(r:RegimeNode)
          -[:PRODUCED]->(t:TradeNode)
    RETURN r.type, avg(t.pnl_pct) as avg_pnl, count(t) as trades
    GROUP BY r.type
    ORDER BY avg_pnl DESC
""")
```

---

## 6. Implementation Roadmap

### Phase 1: In-Memory Graph (NetworkX)
**Time:** 2-3 days
**Complexity:** Low

```python
import networkx as nx

class HFTGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_price(self, ts, price, features):
        node_id = f"price_{ts}"
        self.G.add_node(node_id, type='price', price=price, **features)

    def add_signal(self, ts, signal_type, strength):
        node_id = f"signal_{ts}"
        self.G.add_node(node_id, type='signal', signal_type=signal_type)

    def find_similar_patterns(self, current_features, k=5):
        # Find k most similar historical patterns
        pass
```

### Phase 2: Persistent Graph (Neo4j)
**Time:** 1 week
**Complexity:** Medium

- Deploy Neo4j instance
- Migrate historical data
- Implement Cypher queries for pattern matching
- Add real-time graph updates

### Phase 3: Distributed Execution (Ray)
**Time:** 1-2 weeks
**Complexity:** High

- Convert feature extractors to Ray actors
- Implement distributed signal generation
- Add fault tolerance and recovery

---

## 7. Comparison: Current vs Graph-Based

| Aspect | Current (Scalar/Polar) | Graph-Based |
|--------|----------------------|-------------|
| **Data Model** | Time series | Graph (nodes + edges) |
| **Query Pattern** | Sequential iteration | Pattern matching |
| **Relationships** | Implicit (by time) | Explicit (edges) |
| **Traceability** | Log files | Graph traversal |
| **Distribution** | Single process | Actor model (Ray/Akka) |
| **ML Integration** | Feature vectors | Graph neural networks |
| **Complexity** | Low | Medium-High |

---

## 8. Recommendation

**Start with Phase 1 (NetworkX)** because:
1. **Low risk** - Can run alongside existing system
2. **Fast iteration** - No external dependencies
3. **Proves value** - Test if graph patterns add alpha
4. **Migration path** - NetworkX → Neo4j is straightforward

**Skip distributed execution initially** because:
- Current single-process system handles ~100k trades/second
- Distribution adds complexity (consistency, latency, debugging)
- Premature optimization unless latency becomes a bottleneck

---

## 9. Next Steps

1. **Prototype NetworkX integration** (2-3 days)
   - Add graph layer to existing polar features
   - Implement basic pattern matching
   - Test on historical BTC data

2. **Define graph schema** (1 day)
   - Node types and properties
   - Edge relationships
   - Indexing strategy

3. **Benchmark pattern matching** (1-2 days)
   - Query latency vs sequential scan
   - Memory usage for large graphs
   - Break-even point for graph approach

---

*Document created: 2026-03-29*
*Status: Research phase - awaiting decision on Phase 1 implementation*
