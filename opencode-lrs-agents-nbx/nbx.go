package lrs

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// ============================================================
// Block: A trainable component (layer group) with ID, weights, gradients
// ============================================================

// BlockID uniquely identifies a block
type BlockID string

// Block represents a trainable component that can be exchanged
type Block struct {
	ID        BlockID
	LayerName string
	Weights   []float64
	Gradients []float64
	Metadata  BlockMetadata
}

// BlockMetadata contains additional information about a block
type BlockMetadata struct {
	LayerType   string
	ParamCount  int
	GradNorm    float64
	LastUpdate  time.Time
	TrustScore  float64 // Computed by receiving workers
}

// NewBlock creates a new block
func NewBlock(id BlockID, layerName string, weights []float64) *Block {
	return &Block{
		ID:        id,
		LayerName: layerName,
		Weights:   weights,
		Gradients: make([]float64, len(weights)),
		Metadata: BlockMetadata{
			LayerType:   "linear",
			ParamCount:  len(weights),
			GradNorm:    0.0,
			LastUpdate:  time.Now(),
			TrustScore:  1.0,
		},
	}
}

// ComputeGradientNorm calculates the L2 norm of gradients
func (b *Block) ComputeGradientNorm() float64 {
	var normSq float64
	for _, g := range b.Gradients {
		normSq += g * g
	}
	b.Metadata.GradNorm = math.Sqrt(normSq)
	return b.Metadata.GradNorm
}

// Clone creates a deep copy of the block
func (b *Block) Clone() *Block {
	weightsCopy := make([]float64, len(b.Weights))
	gradsCopy := make([]float64, len(b.Gradients))
	copy(weightsCopy, b.Weights)
	copy(gradsCopy, b.Gradients)

	return &Block{
		ID:        b.ID,
		LayerName: b.LayerName,
		Weights:   weightsCopy,
		Gradients: gradsCopy,
		Metadata:  b.Metadata,
	}
}

// ============================================================
// BlockExchange: P2P-style block sharing between training workers
// ============================================================

// WorkerID identifies a worker in the exchange
type WorkerID int

// ExchangeTopology defines how workers are connected
type ExchangeTopology int

const (
	Ring ExchangeTopology = iota
	AllReduce
	Star
	Mesh
)

// BlockExchange manages the block exchange protocol
type BlockExchange struct {
	WorkerID      WorkerID
	NumWorkers    int
	Topology      ExchangeTopology
	Blocks        map[BlockID]*Block
	ReceivedBlocks map[WorkerID][]*Block
	mu           sync.RWMutex
	ExchangeInterval int
	MaxBlocksPerExchange int
	TrustScores   map[WorkerID]float64
}

func NewBlockExchange(workerID WorkerID, numWorkers int, topology ExchangeTopology) *BlockExchange {
	return &BlockExchange{
		WorkerID:           workerID,
		NumWorkers:          numWorkers,
		Topology:            topology,
		Blocks:              make(map[BlockID]*Block),
		ReceivedBlocks:      make(map[WorkerID][]*Block),
		ExchangeInterval:    10,
		MaxBlocksPerExchange: 3,
		TrustScores:         make(map[WorkerID]float64),
	}
}

func (be *BlockExchange) RegisterBlock(block *Block) {
	be.mu.Lock()
	defer be.mu.Unlock()
	be.Blocks[block.ID] = block
}

func (be *BlockExchange) GetBlock(id BlockID) (*Block, bool) {
	be.mu.RLock()
	defer be.mu.RUnlock()
	b, ok := be.Blocks[id]
	return b, ok
}

func (be *BlockExchange) GetAllBlocks() []*Block {
	be.mu.RLock()
	defer be.mu.RUnlock()
	blocks := make([]*Block, 0, len(be.Blocks))
	for _, b := range be.Blocks {
		blocks = append(blocks, b)
	}
	return blocks
}

// ============================================================
// SGDStep: Each worker trains locally, then exchanges top-K gradient blocks
// ============================================================

// SGDStep performs a local training step followed by block exchange
type SGDStep struct {
	Exchange     *BlockExchange
	LocalSteps   int
	AccumGrad    bool
	BlockSelector *BlockSelector
}

func NewSGDStep(exchange *BlockExchange, localSteps int, accumGrad bool) *SGDStep {
	return &SGDStep{
		Exchange:      exchange,
		LocalSteps:    localSteps,
		AccumGrad:     accumGrad,
		BlockSelector: NewBlockSelector(),
	}
}

func (s *SGDStep) TrainLocal(forward func() float64, backward func(), optStep func()) float64 {
	var totalLoss float64
	for step := 0; step < s.LocalSteps; step++ {
		// Forward pass
		loss := forward()
		totalLoss += loss

		// Backward pass
		backward()

		// Accumulate gradients if enabled
		if s.AccumGrad {
			// Gradients are accumulated in place
		} else {
			// Perform optimization step immediately
			optStep()
		}
	}

	// If accumulating, do final optimization step
	if s.AccumGrad {
		optStep()
	}

	return totalLoss / float64(s.LocalSteps)
}

func (s *SGDStep) SelectAndShareBlocks() []*Block {
	// Select top-K blocks by gradient magnitude to share
	blocks := s.Exchange.GetAllBlocks()
	selected := s.BlockSelector.SelectTopK(blocks, s.Exchange.MaxBlocksPerExchange)
	return selected
}

// ============================================================
// BlockSelector: Selects which gradient blocks to share based on magnitude
// ============================================================

// SelectionStrategy defines how to select blocks
type SelectionStrategy int

const (
	TopKByMagnitude SelectionStrategy = iota
	TopKByUncertainty
	RandomK
	DiversityBased
)

// BlockSelector selects blocks for exchange
type BlockSelector struct {
	Strategy SelectionStrategy
}

func NewBlockSelector() *BlockSelector {
	return &BlockSelector{
		Strategy: TopKByMagnitude,
	}
}

func (bs *BlockSelector) SelectTopK(blocks []*Block, k int) []*Block {
	if len(blocks) <= k {
		return blocks
	}

	// Compute gradient norms and sort
	type scoredBlock struct {
		block *Block
		score float64
	}

	scored := make([]scoredBlock, len(blocks))
	for i, b := range blocks {
		norm := b.ComputeGradientNorm()
		scored[i] = scoredBlock{block: b, score: norm}
	}

	// Sort by score (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Return top K
	result := make([]*Block, k)
	for i := 0; i < k; i++ {
		result[i] = scored[i].block
	}

	return result
}

func (bs *BlockSelector) SelectByUncertainty(blocks []*Block, k int) []*Block {
	// Select blocks where uncertainty is highest
	// Based on gradient variance or other uncertainty measures
	if len(blocks) <= k {
		return blocks
	}

	type scoredBlock struct {
		block   *Block
		uncertainty float64
	}

	scored := make([]scoredBlock, len(blocks))
	for i, b := range blocks {
		// Compute uncertainty as variance of gradients
		mean := 0.0
		for _, g := range b.Gradients {
			mean += g
		}
		mean /= float64(len(b.Gradients))

		var variance float64
		for _, g := range b.Gradients {
			diff := g - mean
			variance += diff * diff
		}
		variance /= float64(len(b.Gradients))

		scored[i] = scoredBlock{block: b, uncertainty: variance}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].uncertainty > scored[j].uncertainty
	})

	result := make([]*Block, k)
	for i := 0; i < k; i++ {
		result[i] = scored[i].block
	}

	return result
}

func (bs *BlockSelector) SelectRandom(blocks []*Block, k int) []*Block {
	if len(blocks) <= k {
		return blocks
	}

	// Shuffle and pick k
	indices := rand.Perm(len(blocks))[:k]
	result := make([]*Block, k)
	for i, idx := range indices {
		result[i] = blocks[idx]
	}

	return result
}

// ============================================================
// ExchangeScheduler: Coordinates block exchange timing and topology
// ============================================================

// ExchangeScheduler manages exchange timing and topology
type ExchangeScheduler struct {
	Topology         ExchangeTopology
	CurrentRingIndex int
	Neighbors        []WorkerID
	mu               sync.Mutex
}

func NewExchangeScheduler(topology ExchangeTopology, numWorkers int, myID WorkerID) *ExchangeScheduler {
	es := &ExchangeScheduler{
		Topology:         topology,
		CurrentRingIndex: int(myID),
	}

	// Build neighbor list based on topology
	switch topology {
	case Ring:
		// In a ring, I'm connected to my left and right neighbors
		prev := (int(myID) - 1 + numWorkers) % numWorkers
		next := (int(myID) + 1) % numWorkers
		es.Neighbors = []WorkerID{WorkerID(prev), WorkerID(next)}

	case Star:
		// In a star, everyone connects to a central node
		// Here we assume node 0 is the center
		for i := 0; i < numWorkers; i++ {
			if i != int(myID) {
				es.Neighbors = append(es.Neighbors, WorkerID(i))
			}
		}

	case AllReduce:
		// In all-reduce, everyone is connected to everyone
		for i := 0; i < numWorkers; i++ {
			if i != int(myID) {
				es.Neighbors = append(es.Neighbors, WorkerID(i))
			}
		}

	case Mesh:
		// Mesh: connect to nearby nodes
		for i := 0; i < numWorkers; i++ {
			if i != int(myID) && abs(i-int(myID)) <= 2 {
				es.Neighbors = append(es.Neighbors, WorkerID(i))
			}
		}
	}

	return es
}

func (es *ExchangeScheduler) GetExchangePartners() []WorkerID {
	es.mu.Lock()
	defer es.mu.Unlock()

	switch es.Topology {
	case Ring:
		// Rotate through the ring
		partners := make([]WorkerID, 2)
		partners[0] = es.Neighbors[0] // Left neighbor
		partners[1] = es.Neighbors[1] // Right neighbor
		return partners

	default:
		return es.Neighbors
	}
}

func (es *ExchangeScheduler) Step() {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.CurrentRingIndex++
}

// ============================================================
// CollaborativeOptimizer: Merges exchanged gradients
// ============================================================

// GradientMerger defines how to merge gradients from multiple workers
type GradientMerger interface {
	MergedGradients(grads [][]float64, weights []float64) []float64
}

// WeightedAverageMerger merges gradients using weighted average
type WeightedAverageMerger struct{}

func (m *WeightedAverageMerger) MergedGradients(grads [][]float64, weights []float64) []float64 {
	if len(grads) == 0 {
		return nil
	}
	if len(grads) == 1 {
		return grads[0]
	}

	// Normalize weights
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight == 0 {
		totalWeight = 1.0
	}

	// Compute weighted average
	result := make([]float64, len(grads[0]))
	for i := range result {
		weightedSum := 0.0
		for j, g := range grads {
			weightedSum += g[i] * weights[j] / totalWeight
		}
		result[i] = weightedSum
	}

	return result
}

// TrustBasedMerger merges gradients based on trust scores
type TrustBasedMerger struct {
	TrustScores map[WorkerID]float64
}

func NewTrustBasedMerger() *TrustBasedMerger {
	return &TrustBasedMerger{
		TrustScores: make(map[WorkerID]float64),
	}
}

func (m *TrustBasedMerger) MergedGradients(grads [][]float64, weights []float64) []float64 {
	// Similar to weighted average but weights are trust scores
	if len(grads) == 0 {
		return nil
	}
	if len(grads) == 1 {
		return grads[0]
	}

	result := make([]float64, len(grads[0]))
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight == 0 {
		totalWeight = 1.0
	}

	for i := range result {
		weightedSum := 0.0
		for j := range grads {
			weightedSum += grads[j][i] * weights[j] / totalWeight
		}
		result[i] = weightedSum
	}

	return result
}

// CollaborativeOptimizer merges gradients from multiple workers
type CollaborativeOptimizer struct {
	Merger             GradientMerger
	TrustScores        map[WorkerID]float64
	LocalWeights       []float64
	ExchangeHistory    []ExchangeRecord
	mu                 sync.RWMutex
}

type ExchangeRecord struct {
	Timestamp    time.Time
	NumBlocks    int
	FromWorkers  []WorkerID
	TrustWeights []float64
}

func NewCollaborativeOptimizer(merger GradientMerger) *CollaborativeOptimizer {
	return &CollaborativeOptimizer{
		Merger:          merger,
		TrustScores:     make(map[WorkerID]float64),
		LocalWeights:    make([]float64, 0),
		ExchangeHistory: make([]ExchangeRecord, 0),
	}
}

func (co *CollaborativeOptimizer) MergeGradients(blocksFromWorkers map[WorkerID][]*Block) []float64 {
	// Collect gradients from each worker
	type workerGrad struct {
		workerID WorkerID
		grads    []float64
	}

	allGrads := make([][]float64, 0)
	weights := make([]float64, 0)

	for workerID, blocks := range blocksFromWorkers {
		// Average gradients within each worker's blocks
		var combinedGrad []float64
		count := 0

		for _, block := range blocks {
			if combinedGrad == nil {
				combinedGrad = make([]float64, len(block.Gradients))
			}
			for i, g := range block.Gradients {
				combinedGrad[i] += g
			}
			count++
		}

		if count > 0 && combinedGrad != nil {
			for i := range combinedGrad {
				combinedGrad[i] /= float64(count)
			}
			allGrads = append(allGrads, combinedGrad)

			// Weight by trust score
			trust := co.TrustScores[workerID]
			if trust == 0 {
				trust = 1.0 // Default trust
			}
			weights = append(weights, trust)
		}
	}

	// Also include local gradients with higher weight
	if len(co.LocalWeights) > 0 {
		allGrads = append(allGrads, co.LocalWeights)
		weights = append(weights, 1.0) // Full trust for local
	}

	// Merge
	return co.Merger.MergedGradients(allGrads, weights)
}

func (co *CollaborativeOptimizer) UpdateTrustScore(fromWorker WorkerID, trustScore float64) {
	co.mu.Lock()
	defer co.mu.Unlock()
	co.TrustScores[fromWorker] = trustScore
}

func (co *CollaborativeOptimizer) GetTrustScore(workerID WorkerID) float64 {
	co.mu.RLock()
	defer co.mu.RUnlock()
	if score, ok := co.TrustScores[workerID]; ok {
		return score
	}
	return 1.0
}

// ============================================================
// NBXAgent: Main agent that coordinates NBX protocol
// ============================================================

// NBXAgent coordinates the Neural Block Exchange protocol
type NBXAgent struct {
	ID             WorkerID
	Exchange       *BlockExchange
	Scheduler      *ExchangeScheduler
	Selector       *BlockSelector
	Optimizer      *CollaborativeOptimizer
	LocalModel     interface {
		GetParameters() [][]float64
	}
	mu             sync.RWMutex
}

func NewNBXAgent(id WorkerID, numWorkers int, topology ExchangeTopology, model interface {
	GetParameters() [][]float64
}) *NBXAgent {
	exchange := NewBlockExchange(id, numWorkers, topology)
	scheduler := NewExchangeScheduler(topology, numWorkers, id)
	selector := NewBlockSelector()
	merger := NewTrustBasedMerger()
	optimizer := NewCollaborativeOptimizer(merger)

	// Register model parameters as blocks
	params := model.GetParameters()
	for i, p := range params {
		blockID := BlockID(fmt.Sprintf("layer_%d", i))
		block := NewBlock(blockID, fmt.Sprintf("layer_%d", i), p)
		exchange.RegisterBlock(block)
	}

	return &NBXAgent{
		ID:         id,
		Exchange:   exchange,
		Scheduler:  scheduler,
		Selector:   selector,
		Optimizer:  optimizer,
		LocalModel: model,
	}
}

func (n *NBXAgent) RunTrainingStep(forward func() float64, backward func(), updateFunc func([]float64)) float64 {
	// Step 1: Local training
	loss := forward()
	backward()

	// Step 2: Compute gradient norms for all blocks
	blocks := n.Exchange.GetAllBlocks()
	for _, b := range blocks {
		b.ComputeGradientNorm()
	}

	// Step 3: Select blocks to share
	selectedBlocks := n.Selector.SelectTopK(blocks, n.Exchange.MaxBlocksPerExchange)

	// In a real implementation, would send selectedBlocks to partners
	// and receive their blocks

	// Step 4: Simulate receiving blocks from other workers
	receivedBlocks := make(map[WorkerID][]*Block)
	// This would be actual P2P communication in real implementation
	_ = receivedBlocks

	// Step 5: Merge gradients (simulated - just use local)
	mergedGrad := make([]float64, 0)
	for _, b := range selectedBlocks {
		mergedGrad = append(mergedGrad, b.Gradients...)
	}

	// Step 6: Apply merged gradients
	if len(mergedGrad) > 0 {
		updateFunc(mergedGrad)
	}

	// Step 7: Update exchange scheduler
	n.Scheduler.Step()

	return loss
}

// ============================================================
// Utility functions
// ============================================================

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// ComputeTrustScore computes trust based on gradient similarity
func ComputeTrustScore(localGrad, remoteGrad []float64) float64 {
	if len(localGrad) != len(remoteGrad) {
		return 0.5
	}

	// Compute cosine similarity
	dotProduct := 0.0
	localNormSq := 0.0
	remoteNormSq := 0.0

	for i := range localGrad {
		dotProduct += localGrad[i] * remoteGrad[i]
		localNormSq += localGrad[i] * localGrad[i]
		remoteNormSq += remoteGrad[i] * remoteGrad[i]
	}

	localNorm := math.Sqrt(localNormSq)
	remoteNorm := math.Sqrt(remoteNormSq)

	if localNorm == 0 || remoteNorm == 0 {
		return 0.5
	}

	similarity := dotProduct / (localNorm * remoteNorm)

	// Convert similarity to trust score (0 to 1)
	// Use tanh to squash to [0, 1]
	trust := (math.Tanh(similarity) + 1) / 2

	return trust
}
