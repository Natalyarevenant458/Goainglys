package main

import (
	"math"
	"math/rand"
)

// Graph represents a graph with adjacency list and node features
type Graph struct {
	Adjacency    [][]int      // Adjacency list: node -> neighbors
	NodeFeatures [][]float64 // Node features: node -> features
	NumNodes     int
	NumEdges     int
}

// NewGraph creates a new graph
func NewGraph(adjacency [][]int, nodeFeatures [][]float64) *Graph {
	numNodes := len(nodeFeatures)
	numEdges := 0
	for _, neighbors := range adjacency {
		numEdges += len(neighbors)
	}
	// Undirected graph: divide by 2
	numEdges /= 2

	return &Graph{
		Adjacency:    adjacency,
		NodeFeatures: nodeFeatures,
		NumNodes:     numNodes,
		NumEdges:     numEdges,
	}
}

// GraphConv implements GraphSAGE-style graph convolution
type GraphConv struct {
	W      [][]float64 // Weight matrix (inDim x outDim)
	B      []float64  // Bias (outDim)
	InDim  int
	OutDim int
	Act    func(float64) float64
}

// NewGraphConv creates a new graph convolution layer
func NewGraphConv(inDim, outDim int, act func(float64) float64) *GraphConv {
	w := make([][]float64, inDim)
	for i := range w {
		w[i] = make([]float64, outDim)
		for j := range w[i] {
			w[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inDim))
		}
	}

	b := make([]float64, outDim)
	for i := range b {
		b[i] = 0.0
	}

	return &GraphConv{
		W:     w,
		B:     b,
		InDim: inDim,
		OutDim: outDim,
		Act:   act,
	}
}

// Forward performs graph convolution
// graph: input graph
// Returns: node embeddings (numNodes x outDim)
func (gc *GraphConv) Forward(graph *Graph) [][]float64 {
	numNodes := graph.NumNodes
	inDim := gc.InDim
	outDim := gc.OutDim

	// Step 1: Aggregate neighbor features (mean pooling)
	// aggregate[node] = mean of neighbor features + self features
	aggregated := make([][]float64, numNodes)
	for i := range aggregated {
		aggregated[i] = make([]float64, inDim)
	}

	for node := 0; node < numNodes; node++ {
		// Add self features
		for f := 0; f < inDim; f++ {
			aggregated[node][f] = graph.NodeFeatures[node][f]
		}

		// Add neighbor features
		neighbors := graph.Adjacency[node]
		if len(neighbors) > 0 {
			for _, neighbor := range neighbors {
				for f := 0; f < inDim; f++ {
					aggregated[node][f] += graph.NodeFeatures[neighbor][f]
				}
			}
			// Divide by (1 + num_neighbors) for mean
			divisor := float64(1 + len(neighbors))
			for f := 0; f < inDim; f++ {
				aggregated[node][f] /= divisor
			}
		}
	}

	// Step 2: Apply linear transformation + activation
	output := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		output[i] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			sum := gc.B[j]
			for k := 0; k < inDim; k++ {
				sum += aggregated[i][k] * gc.W[k][j]
			}
			if gc.Act != nil {
				output[i][j] = gc.Act(sum)
			} else {
				output[i][j] = sum
			}
		}
	}

	return output
}

// MessagePassing implements multi-hop message passing
type MessagePassing struct {
	Convs     []*GraphConv
	NumLayers int
	HiddenDim int
}

// NewMessagePassing creates a multi-hop message passing layer
func NewMessagePassing(inputDim, hiddenDim, numHops int) *MessagePassing {
	convs := make([]*GraphConv, numHops)
	
	// First layer: input -> hidden
	convs[0] = NewGraphConv(inputDim, hiddenDim, relu)
	
	// Middle layers: hidden -> hidden
	for i := 1; i < numHops-1; i++ {
		convs[i] = NewGraphConv(hiddenDim, hiddenDim, relu)
	}
	
	// Last layer: hidden -> hidden (no activation for graph-level output)
	if numHops > 1 {
		convs[numHops-1] = NewGraphConv(hiddenDim, hiddenDim, nil)
	}

	return &MessagePassing{
		Convs:     convs,
		NumLayers: numHops,
		HiddenDim: hiddenDim,
	}
}

// Forward performs multi-hop message passing
// graph: input graph
// Returns: final node embeddings (numNodes x hiddenDim)
func (mp *MessagePassing) Forward(graph *Graph) [][]float64 {
	// Create a temporary graph to pass through layers
	tempGraph := &Graph{
		Adjacency:    graph.Adjacency,
		NodeFeatures: graph.NodeFeatures,
		NumNodes:     graph.NumNodes,
		NumEdges:     graph.NumEdges,
	}

	var output [][]float64

	for layer := 0; layer < mp.NumLayers; layer++ {
		output = mp.Convs[layer].Forward(tempGraph)

		// Update node features for next layer
		tempGraph.NodeFeatures = output
	}

	return output
}

// GraphAttention implements Graph Attention Network (GAT) style attention
type GraphAttention struct {
	W          [][]float64   // Feature transformation (inDim x outDim)
	Attention  [][]float64   // Attention weights (outDim x 1)
	B          []float64     // Bias (outDim)
	InDim      int
	OutDim     int
	NumHeads   int
	Act        func(float64) float64
}

// NewGraphAttention creates a new graph attention layer
func NewGraphAttention(inDim, outDim, numHeads int, act func(float64) float64) *GraphAttention {
	// Use same dimension for all heads
	headDim := outDim / numHeads
	if headDim*numHeads != outDim {
		headDim = outDim
		numHeads = 1
	}

	w := make([][]float64, inDim)
	for i := range w {
		w[i] = make([]float64, outDim)
		for j := range w[i] {
			w[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inDim))
		}
	}

	// Attention mechanism: single attention weight per head
	attention := make([][]float64, outDim)
	for i := range attention {
		attention[i] = make([]float64, 1)
		for j := range attention[i] {
			attention[i][j] = rand.NormFloat64() * 0.01
		}
	}

	b := make([]float64, outDim)
	for i := range b {
		b[i] = 0.0
	}

	return &GraphAttention{
		W:         w,
		Attention: attention,
		B:         b,
		InDim:     inDim,
		OutDim:    outDim,
		NumHeads:  numHeads,
		Act:       act,
	}
}

// Forward performs graph attention
// graph: input graph
// Returns: node embeddings with attention (numNodes x outDim)
func (gat *GraphAttention) Forward(graph *Graph) [][]float64 {
	numNodes := graph.NumNodes
	inDim := gat.InDim
	outDim := gat.OutDim

	// Step 1: Linear transformation
	h := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		h[i] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			sum := gat.B[j]
			for k := 0; k < inDim; k++ {
				sum += graph.NodeFeatures[i][k] * gat.W[k][j]
			}
			h[i][j] = sum
		}
	}

	// Step 2: Compute attention coefficients
	// e_ij = LeakyReLU(attention @ (h_i || h_j))
	attentionCoeffs := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		attentionCoeffs[i] = make([]float64, numNodes)
		
		// Self attention + neighbor attention
		for j := 0; j < numNodes; j++ {
			// Compute attention key
			var attnScore float64
			for k := 0; k < outDim; k++ {
				attnScore += (h[i][k] - h[j][k]) * gat.Attention[k][0]
			}
			// LeakyReLU
			if attnScore > 0 {
				attentionCoeffs[i][j] = attnScore
			} else {
				attentionCoeffs[i][j] = 0.01 * attnScore
			}
		}
	}

	// Step 3: Normalize attention coefficients (softmax over neighbors)
	normalizedAttn := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		normalizedAttn[i] = make([]float64, numNodes)
		
		// Get neighbors + self
		neighbors := append([]int{i}, graph.Adjacency[i]...)
		
		// Compute sum of exp(attention)
		sumExp := 0.0
		for _, n := range neighbors {
			sumExp += math.Exp(attentionCoeffs[i][n])
		}
		
		// Softmax
		for j := 0; j < numNodes; j++ {
			if sumExp > 0 {
				normalizedAttn[i][j] = math.Exp(attentionCoeffs[i][j]) / sumExp
			}
		}
	}

	// Step 4: Weighted sum of neighbor features
	output := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		output[i] = make([]float64, outDim)
		
		neighbors := append([]int{i}, graph.Adjacency[i]...)
		
		for j := 0; j < outDim; j++ {
			sum := 0.0
			for _, n := range neighbors {
				sum += normalizedAttn[i][n] * h[n][j]
			}
			output[i][j] = sum
		}
		
		// Apply activation
		if gat.Act != nil {
			for j := 0; j < outDim; j++ {
				output[i][j] = gat.Act(output[i][j])
			}
		}
	}

	return output
}

// ReadoutPooling performs graph-level readout
type ReadoutPooling struct {
	Method string // "mean", "sum", "max"
}

// NewReadoutPooling creates a new readout pooling layer
func NewReadoutPooling(method string) *ReadoutPooling {
	return &ReadoutPooling{Method: method}
}

// Forward performs graph-level pooling
// nodeEmbeddings: (numNodes x embeddingDim)
// Returns: graph-level embedding (embeddingDim)
func (rp *ReadoutPooling) Forward(nodeEmbeddings [][]float64) []float64 {
	numNodes := len(nodeEmbeddings)
	if numNodes == 0 {
		return []float64{}
	}

	embeddingDim := len(nodeEmbeddings[0])
	result := make([]float64, embeddingDim)

	switch rp.Method {
	case "mean":
		for j := 0; j < embeddingDim; j++ {
			sum := 0.0
			for i := 0; i < numNodes; i++ {
				sum += nodeEmbeddings[i][j]
			}
			result[j] = sum / float64(numNodes)
		}
	case "sum":
		for j := 0; j < embeddingDim; j++ {
			sum := 0.0
			for i := 0; i < numNodes; i++ {
				sum += nodeEmbeddings[i][j]
			}
			result[j] = sum
		}
	case "max":
		for j := 0; j < embeddingDim; j++ {
			maxVal := nodeEmbeddings[0][j]
			for i := 1; i < numNodes; i++ {
				if nodeEmbeddings[i][j] > maxVal {
					maxVal = nodeEmbeddings[i][j]
				}
			}
			result[j] = maxVal
		}
	default:
		// Default to mean
		for j := 0; j < embeddingDim; j++ {
			sum := 0.0
			for i := 0; i < numNodes; i++ {
				sum += nodeEmbeddings[i][j]
			}
			result[j] = sum / float64(numNodes)
		}
	}

	return result
}

// GraphNetwork is a full graph neural network
type GraphNetwork struct {
	MessagePassing *MessagePassing
	Readout        *ReadoutPooling
	OutputLayer    *GraphConv
	InputDim       int
	HiddenDim      int
	OutputDim      int
}

// NewGraphNetwork creates a new graph neural network
func NewGraphNetwork(inputDim, hiddenDim, outputDim, numHops int) *GraphNetwork {
	mp := NewMessagePassing(inputDim, hiddenDim, numHops)
	readout := NewReadoutPooling("mean")
	
	// Output layer: hidden -> output
	outputLayer := NewGraphConv(hiddenDim, outputDim, nil)

	return &GraphNetwork{
		MessagePassing: mp,
		Readout:        readout,
		OutputLayer:    outputLayer,
		InputDim:       inputDim,
		HiddenDim:      hiddenDim,
		OutputDim:      outputDim,
	}
}

// Forward performs the full GNN forward pass
// graph: input graph
// Returns: graph-level prediction (outputDim)
func (gn *GraphNetwork) Forward(graph *Graph) []float64 {
	// Multi-hop message passing
	nodeEmbeddings := gn.MessagePassing.Forward(graph)

	// Create temporary graph for output layer
	tempGraph := &Graph{
		Adjacency:    graph.Adjacency,
		NodeFeatures: nodeEmbeddings,
		NumNodes:     graph.NumNodes,
		NumEdges:     graph.NumEdges,
	}

	// Output layer
	nodeOutput := gn.OutputLayer.Forward(tempGraph)

	// Readout pooling
	graphEmbedding := gn.Readout.Forward(nodeOutput)

	return graphEmbedding
}

// ForwardNodeClassification performs forward pass for node classification
// graph: input graph
// Returns: node-level predictions (numNodes x outputDim)
func (gn *GraphNetwork) ForwardNodeClassification(graph *Graph) [][]float64 {
	// Multi-hop message passing
	nodeEmbeddings := gn.MessagePassing.Forward(graph)

	// Create temporary graph for output layer
	tempGraph := &Graph{
		Adjacency:    graph.Adjacency,
		NodeFeatures: nodeEmbeddings,
		NumNodes:     graph.NumNodes,
		NumEdges:     graph.NumEdges,
	}

	// Output layer
	nodeOutput := gn.OutputLayer.Forward(tempGraph)

	return nodeOutput
}

// ComputeNodeLoss computes cross-entropy loss for node classification
func (gn *GraphNetwork) ComputeNodeLoss(graph *Graph, labels []int) float64 {
	predictions := gn.ForwardNodeClassification(graph)
	
	loss := 0.0
	for i, label := range labels {
		// Softmax + cross entropy
		maxVal := predictions[i][0]
		for j := 1; j < len(predictions[i]); j++ {
			if predictions[i][j] > maxVal {
				maxVal = predictions[i][j]
			}
		}
		
		sum := 0.0
		for j := 0; j < len(predictions[i]); j++ {
			sum += math.Exp(predictions[i][j] - maxVal)
		}
		
		// Get probability of correct class
		if label < len(predictions[i]) {
			prob := math.Exp(predictions[i][label]-maxVal) / sum
			if prob > 0 {
				loss -= math.Log(prob)
			}
		}
	}
	
	return loss / float64(len(labels))
}

// ComputeGraphLoss computes loss for graph classification
func (gn *GraphNetwork) ComputeGraphLoss(graph *Graph, label int) float64 {
	prediction := gn.Forward(graph)
	
	// Simple MSE loss for single graph
	target := make([]float64, len(prediction))
	if label == 1 {
		for i := range target {
			target[i] = 1.0
		}
	}
	
	loss := 0.0
	for i := range prediction {
		diff := prediction[i] - target[i]
		loss += diff * diff
	}
	
	return loss
}

// CreateSampleGraph creates a sample graph for testing
func CreateSampleGraph() *Graph {
	// Simple chain graph: 0-1-2-3-4
	adjacency := [][]int{
		{1},       // Node 0 connects to 1
		{0, 2},    // Node 1 connects to 0, 2
		{1, 3},    // Node 2 connects to 1, 3
		{2, 4},    // Node 3 connects to 2, 4
		{3},       // Node 4 connects to 3
	}

	// Random node features
	numNodes := 5
	featureDim := 8
	nodeFeatures := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		nodeFeatures[i] = make([]float64, featureDim)
		for j := 0; j < featureDim; j++ {
			nodeFeatures[i][j] = rand.Float64()
		}
	}

	return NewGraph(adjacency, nodeFeatures)
}

// CreateSampleGraphWithCycle creates a graph with cycles for testing
func CreateSampleGraphWithCycle() *Graph {
	// Graph with cycle: 0-1-2, 1-3-4-2
	adjacency := [][]int{
		{1, 2},       // Node 0
		{0, 2, 3},   // Node 1
		{0, 1, 4},   // Node 2
		{1, 4},      // Node 3
		{2, 3},      // Node 4
	}

	numNodes := 5
	featureDim := 8
	nodeFeatures := make([][]float64, numNodes)
	for i := 0; i < numNodes; i++ {
		nodeFeatures[i] = make([]float64, featureDim)
		for j := 0; j < featureDim; j++ {
			nodeFeatures[i][j] = rand.Float64()
		}
	}

	return NewGraph(adjacency, nodeFeatures)
}
