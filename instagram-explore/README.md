# Instagram Explore Recommendation System Design

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [ML Training Pipeline](#ml-training-pipeline)
4. [Two-Tower Architecture](#two-tower-architecture)
5. [Monitoring System](#monitoring-system)
6. [Interview Deep Dive](#interview-deep-dive)

## System Overview

### Problem Statement
Design a scalable recommendation system for Instagram's Explore tab that serves personalized content to billions of users while maintaining:
- Low latency (< 100ms for initial recommendations)
- High throughput (millions of QPS)
- Real-time personalization
- Content diversity and freshness

### System Requirements

#### Functional Requirements
- Serve personalized content recommendations
- Handle multiple content types (images, videos, Reels)
- Process real-time user interactions
- Support both cold-start and warm-start scenarios
- Maintain content diversity and freshness

#### Non-Functional Requirements
- Latency: < 100ms for initial page load
- Availability: 99.99% uptime
- Scalability: Handle millions of QPS
- Consistency: Maintain recommendation quality across updates

## Architecture Deep Dive

### High-Level System Architecture

[Insert high-level-arch diagram]

Key components:

1. **Client Layer**
   - User Interface
   - Client-side caching
   - Progressive loading

2. **API Gateway & Load Balancing**
   ```python
   class LoadBalancer:
       def __init__(self):
           self.strategies = {
               'ingestion': RoundRobinStrategy(),
               'processing': LeastConnectionStrategy(),
               'serving': ConsistentHashingStrategy()
           }

       async def route_request(self, request_type, request):
           strategy = self.strategies[request_type]
           server = strategy.select_server()
           return await server.process(request)
   ```

3. **Recommendation Service**
   - Candidate retrieval
   - Ranking service
   - Feature service

4. **Storage Layer**
   - Redis cache
   - Vector store
   - Feature store

## ML Training Pipeline

[Insert ml-training-revised diagram]

### Data Pipeline
```python
class DataPipeline:
    def __init__(self):
        self.validators = {
            'schema': SchemaValidator(),
            'quality': QualityChecker(),
            'statistics': StatsAnalyzer()
        }
        
    async def process_data(self, raw_data):
        # Data validation
        validation_results = await self.validate_data(raw_data)
        if not validation_results.passed:
            raise DataQualityError(validation_results.errors)
            
        # Feature processing
        processed_data = await self.process_features(raw_data)
        
        # Train/val/test split with time-based splitting
        return self.create_datasets(processed_data)
```

### Feature Engineering
1. **User Features**
   - Historical interactions
   - Demographics
   - Session features

2. **Item Features**
   - Visual features (CNN embeddings)
   - Text features (BERT embeddings)
   - Engagement features

## Two-Tower Architecture

[Insert two-tower-detailed diagram]

### Model Architecture
```python
class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)
        self.interaction = InteractionLayer(config)
        
    def forward(self, user_features, item_features):
        # User tower processing
        user_embedding = self.user_tower(user_features)
        
        # Item tower processing
        item_embedding = self.item_tower(item_features)
        
        # Compute similarity and additional features
        similarity = self.interaction(user_embedding, item_embedding)
        return similarity
```

### Negative Sampling Strategies
1. **Hard Negative Mining**
   ```python
   class HardNegativeMiner:
       def __init__(self, queue_size=1000):
           self.queue = deque(maxlen=queue_size)
           self.threshold = 0.7
           
       def find_hard_negatives(self, anchor, positive):
           similarities = cosine_similarity(anchor, self.queue)
           hard_negatives = self.queue[
               (similarities > self.threshold) & 
               (similarities < self.positive_similarity)
           ]
           return hard_negatives
   ```

2. **Batch Negatives**
   - In-batch sampling
   - Cross-batch mining
   - Dynamic queue

### Ranking System
```python
class RankingSystem:
    def __init__(self):
        self.retrieval = AnnModel()
        self.ranker = DeepRankingModel()
        self.diversity = DiversityReranker()
        
    async def rank_items(self, user_embedding, candidates):
        # Stage 1: Initial retrieval
        initial_scores = await self.retrieval.get_scores(
            user_embedding, candidates
        )
        top_k = self.get_top_k(initial_scores, k=1000)
        
        # Stage 2: Fine-grained ranking
        final_scores = await self.ranker.score(
            user_embedding,
            top_k,
            self.get_context_features()
        )
        
        # Stage 3: Diversity reranking
        return self.diversity.rerank(final_scores)
```

## Monitoring System

[Insert detailed-monitoring diagram]

### Data Monitoring
1. **Feature Monitoring**
   - Feature drift detection
   - Distribution monitoring
   - Missing value patterns

2. **Quality Checks**
   - Schema validation
   - Business rule validation
   - Statistical checks

### Model Monitoring
```python
class ModelMonitor:
    def __init__(self):
        self.metrics = {
            'online': {
                'ctr': CTRTracker(),
                'engagement': EngagementTracker(),
                'diversity': DiversityTracker()
            },
            'offline': {
                'ndcg': NDCGCalculator(),
                'map': MAPCalculator(),
                'coverage': CoverageCalculator()
            }
        }
        
    async def track_metrics(self, predictions, actuals):
        results = {}
        for name, tracker in self.metrics.items():
            results[name] = await tracker.compute(
                predictions, actuals
            )
        return results
```

## Interview Deep Dive

### System Design Considerations

1. **Scalability**
   - How to handle 10x traffic increase?
   - Strategies for reducing latency
   - Handling hot items/users

2. **Reliability**
   - Failure handling mechanisms
   - Data consistency approaches
   - Recovery procedures

3. **ML Architecture**
   - Model selection rationale
   - Training-serving skew handling
   - Feature engineering choices

### Common Interview Questions

1. **Data Pipeline**
   Q: How do you ensure data quality?
   A: Implement multi-layer validation:
   - Schema validation
   - Data quality checks
   - Distribution monitoring
   - Anomaly detection

2. **Model Architecture**
   Q: Why choose two-tower over other architectures?
   A: Benefits include:
   - Separate user/item processing
   - Efficient ANN search
   - Scalable inference
   - Easy to update independently

3. **Online Serving**
   Q: How to handle real-time updates?
   A: Use stream processing:
   - Kafka for event streaming
   - Real-time feature updates
   - Incremental model updates

### Performance Optimization

1. **Latency Optimization**
   ```python
   class CachingStrategy:
       def __init__(self):
           self.embedding_cache = LRUCache()
           self.prediction_cache = TTLCache()
           
       async def get_predictions(self, user_id):
           if cached := await self.prediction_cache.get(user_id):
               return cached
               
           predictions = await self.compute_predictions(user_id)
           await self.prediction_cache.set(
               user_id, predictions, ttl=3600
           )
           return predictions
   ```

2. **Resource Optimization**
   - Embedding compression
   - Batch prediction
   - Caching strategies

3. **Quality Assurance**
   - A/B testing framework
   - Shadow testing
   - Gradual rollout