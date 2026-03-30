package main

import (
	"math"
	"math/rand"
	"sort"
)

// MoEConfig holds the configuration for Mixture of Experts
type MoEConfig struct {
	NumExperts int     // Number of expert networks
	TopK       int     // Number of experts to route to
	ExpertDim  int     // Hidden dimension for experts
	Capacity   float64 // Expert capacity factor
}

// Expert is a simple 2-layer MLP
type Expert struct {
	W1 [][]float64 // First weight matrix (inputDim -> hiddenDim)
	B1 []float64   // First bias (hiddenDim)
	W2 [][]float64 // Second weight matrix (hiddenDim -> outputDim)
	B2 []float64   // Second bias (outputDim)
}

// NewExpert creates a new expert with random weights
func NewExpert(inputDim, hiddenDim, outputDim int) *Expert {
	w1 := make([][]float64, inputDim)
	for i := range w1 {
		w1[i] = make([]float64, hiddenDim)
		for j := range w1[i] {
			w1[i][j] = rand.NormFloat64() * 0.02
		}
	}

	b1 := make([]float64, hiddenDim)
	for i := range b1 {
		b1[i] = rand.NormFloat64() * 0.02
	}

	w2 := make([][]float64, hiddenDim)
	for i := range w2 {
		w2[i] = make([]float64, outputDim)
		for j := range w2[i] {
			w2[i][j] = rand.NormFloat64() * 0.02
		}
	}

	b2 := make([]float64, outputDim)
	for i := range b2 {
		b2[i] = rand.NormFloat64() * 0.02
	}

	return &Expert{W1: w1, B1: b1, W2: w2, B2: b2}
}

// Forward performs the forward pass through the expert
// x: input matrix (batchSize x inputDim)
// Returns: output matrix (batchSize x outputDim)
func (e *Expert) Forward(x [][]float64) [][]float64 {
	batchSize := len(x)
	inputDim := len(x[0])
	hiddenDim := len(e.B1)
	outputDim := len(e.B2)

	// First layer: linear + GELU
	// hidden = x @ W1 + b1
	hidden := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		hidden[i] = make([]float64, hiddenDim)
		for j := 0; j < hiddenDim; j++ {
			sum := e.B1[j]
			for k := 0; k < inputDim; k++ {
				sum += x[i][k] * e.W1[k][j]
			}
			// GELU activation
			hidden[i][j] = gelu(sum)
		}
	}

	// Second layer: linear
	// out = hidden @ W2 + b2
	output := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		output[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			sum := e.B2[j]
			for k := 0; k < hiddenDim; k++ {
				sum += hidden[i][k] * e.W2[k][j]
			}
			output[i][j] = sum
		}
	}

	return output
}

// GELU activation function
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// MoELayer implements the Mixture of Experts layer
type MoELayer struct {
	config     MoEConfig
	experts    []*Expert
	routerW    [][]float64 // Router weight matrix
	routerB    []float64   // Router bias
	inputDim   int
	outputDim  int
	expertCaps []int // Expert capacities per batch
}

// NewMoELayer creates a new MoE layer
func NewMoELayer(inputDim, outputDim int, config MoEConfig) *MoELayer {
	experts := make([]*Expert, config.NumExperts)
	for i := 0; i < config.NumExperts; i++ {
		experts[i] = NewExpert(inputDim, config.ExpertDim, outputDim)
	}

	// Router network: linear layer
	routerW := make([][]float64, inputDim)
	for i := range routerW {
		routerW[i] = make([]float64, config.NumExperts)
		for j := range routerW[i] {
			routerW[i][j] = rand.NormFloat64() * 0.02
		}
	}

	routerB := make([]float64, config.NumExperts)
	for i := range routerB {
		routerB[i] = rand.NormFloat64() * 0.02
	}

	return &MoELayer{
		config:    config,
		experts:   experts,
		routerW:   routerW,
		routerB:   routerB,
		inputDim:  inputDim,
		outputDim: outputDim,
		expertCaps: make([]int, config.NumExperts),
	}
}

// Route computes routing decisions for each input
// Returns: routing weights (batchSize x numExperts)
func (m *MoELayer) Route(x [][]float64) [][]float64 {
	batchSize := len(x)
	numExperts := m.config.NumExperts

	// Compute logits: (batchSize x inputDim) @ (inputDim x numExperts) -> (batchSize x numExperts)
	logits := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		logits[i] = make([]float64, numExperts)
		for j := 0; j < numExperts; j++ {
			sum := m.routerB[j]
			for k := 0; k < m.inputDim; k++ {
				sum += x[i][k] * m.routerW[k][j]
			}
			logits[i][j] = sum
		}
	}

	// Apply softmax to get routing probabilities
	routingWeights := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		routingWeights[i] = softmax(logits[i])
	}

	return routingWeights
}

// TopK selects top-k experts for each input
// Returns: indices of top-k experts (batchSize x topK), weights for top-k (batchSize x topK)
func (m *MoELayer) TopK(routingWeights [][]float64) ([][]int, [][]float64) {
	batchSize := len(routingWeights)
	numExperts := m.config.NumExperts

	expertIndices := make([][]int, batchSize)
	expertWeights := make([][]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		// Create index-value pairs
		pairs := make([]struct {
			index int
			value float64
		}, numExperts)
		for j := 0; j < numExperts; j++ {
			pairs[j] = struct {
				index int
				value float64
			}{j, routingWeights[i][j]}
		}

		// Sort by value descending
		sort.Slice(pairs, func(a, b int) bool {
			return pairs[a].value > pairs[b].value
		})

		// Take top-k
		topK := m.config.TopK
		expertIndices[i] = make([]int, topK)
		expertWeights[i] = make([]float64, topK)
		capacity := int(float64(len(routingWeights[i])) * m.config.Capacity / float64(numExperts))
		
		count := 0
		for j := 0; j < numExperts && count < topK; j++ {
			if m.expertCaps[pairs[j].index] < capacity || capacity == 0 {
				expertIndices[i][count] = pairs[j].index
				expertWeights[i][count] = pairs[j].value
				m.expertCaps[pairs[j].index]++
				count++
			}
		}
		
		// If couldn't fill topK due to capacity, normalize weights
		if count > 0 && count < topK {
			sum := 0.0
			for j := 0; j < count; j++ {
				sum += expertWeights[i][j]
			}
			if sum > 0 {
				for j := 0; j < count; j++ {
					expertWeights[i][j] /= sum
				}
			}
		}
	}

	return expertIndices, expertWeights
}

// Forward performs the MoE forward pass
// x: input (batchSize x inputDim)
// Returns: output (batchSize x outputDim)
func (m *MoELayer) Forward(x [][]float64) [][]float64 {
	batchSize := len(x)
	
	// Reset expert capacities
	for i := range m.expertCaps {
		m.expertCaps[i] = 0
	}

	// Step 1: Route inputs to experts
	routingWeights := m.Route(x)

	// Step 2: Select top-k experts
	expertIndices, expertWeights := m.TopK(routingWeights)

	// Step 3: Dispatch inputs to experts and compute outputs
	expertOutputs := make([][][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		expertOutputs[b] = make([][]float64, m.config.TopK)
	}

	// Process each sample
	for b := 0; b < batchSize; b++ {
		inputRow := make([]float64, m.inputDim)
		copy(inputRow, x[b])
		
		for k := 0; k < m.config.TopK; k++ {
			expertIdx := expertIndices[b][k]
			weight := expertWeights[b][k]
			
			// Get expert output
			expertInput := [][]float64{inputRow}
			expertOutput := m.experts[expertIdx].Forward(expertInput)
			
			// Scale by routing weight
			scaledOutput := make([]float64, m.outputDim)
			for j := 0; j < m.outputDim; j++ {
				scaledOutput[j] = expertOutput[0][j] * weight
			}
			expertOutputs[b][k] = scaledOutput
		}
	}

	// Step 4: Aggregate outputs
	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, m.outputDim)
		for k := 0; k < m.config.TopK; k++ {
			if expertOutputs[b][k] != nil {
				for j := 0; j < m.outputDim; j++ {
					output[b][j] += expertOutputs[b][k][j]
				}
			}
		}
	}

	return output
}

// Backward performs backpropagation through the MoE layer
// This is a simplified version that computes numerical gradients
func (m *MoELayer) Backward(x, gradOutput [][]float64, lr float64) {
	batchSize := len(x)
	numExperts := m.config.NumExperts
	epsilon := 1e-5

	// Compute gradients via finite differences for router
	routerGradW := make([][]float64, m.inputDim)
	for i := range routerGradW {
		routerGradW[i] = make([]float64, numExperts)
	}

	for i := 0; i < m.inputDim; i++ {
		for j := 0; j < numExperts; j++ {
			// Compute numerical gradient
			original := m.routerW[i][j]
			
			m.routerW[i][j] = original + epsilon
			outputPlus := m.Forward(x)
			lossPlus := sumSquaredDiff(outputPlus, gradOutput)
			
			m.routerW[i][j] = original - epsilon
			outputMinus := m.Forward(x)
			lossMinus := sumSquaredDiff(outputMinus, gradOutput)
			
			m.routerW[i][j] = original
			grad := (lossPlus - lossMinus) / (2 * epsilon)
			routerGradW[i][j] = grad
		}
	}

	// Update router weights
	for i := 0; i < m.inputDim; i++ {
		for j := 0; j < numExperts; j++ {
			m.routerW[i][j] -= lr * routerGradW[i][j]
		}
	}

	// Update expert weights (simplified - update each expert)
	for ex := 0; ex < numExperts; ex++ {
		m.expertBackward(x[0], gradOutput[0], m.experts[ex], lr)
	}

	_ = batchSize // batchSize was used for input dimensions
}

// expertBackward performs backpropagation for a single expert
func (m *MoELayer) expertBackward(x, gradOutput []float64, expert *Expert, lr float64) {
	epsilon := 1e-5
	inputDim := m.inputDim
	hiddenDim := len(expert.B1)
	outputDim := m.outputDim

	// Gradient for W2
	for i := 0; i < hiddenDim; i++ {
		for j := 0; j < outputDim; j++ {
			original := expert.W2[i][j]
			expert.W2[i][j] = original + epsilon
			expertOutPlus := expert.Forward([][]float64{x})
			lossPlus := sumSquaredDiffRow(expertOutPlus[0], gradOutput)
			
			expert.W2[i][j] = original - epsilon
			expertOutMinus := expert.Forward([][]float64{x})
			lossMinus := sumSquaredDiffRow(expertOutMinus[0], gradOutput)
			
			expert.W2[i][j] = original
			grad := (lossPlus - lossMinus) / (2 * epsilon)
			expert.W2[i][j] -= lr * grad
		}
	}

	// Gradient for B2
	for j := 0; j < outputDim; j++ {
		original := expert.B2[j]
		expert.B2[j] = original + epsilon
		expertOutPlus := expert.Forward([][]float64{x})
		lossPlus := sumSquaredDiffRow(expertOutPlus[0], gradOutput)
		
		expert.B2[j] = original - epsilon
		expertOutMinus := expert.Forward([][]float64{x})
		lossMinus := sumSquaredDiffRow(expertOutMinus[0], gradOutput)
		
		expert.B2[j] = original
		grad := (lossPlus - lossMinus) / (2 * epsilon)
		expert.B2[j] -= lr * grad
	}

	// Gradient for W1
	for i := 0; i < inputDim; i++ {
		for j := 0; j < hiddenDim; j++ {
			original := expert.W1[i][j]
			expert.W1[i][j] = original + epsilon
			expertOutPlus := expert.Forward([][]float64{x})
			lossPlus := sumSquaredDiffRow(expertOutPlus[0], gradOutput)
			
			expert.W1[i][j] = original - epsilon
			expertOutMinus := expert.Forward([][]float64{x})
			lossMinus := sumSquaredDiffRow(expertOutMinus[0], gradOutput)
			
			expert.W1[i][j] = original
			grad := (lossPlus - lossMinus) / (2 * epsilon)
			expert.W1[i][j] -= lr * grad
		}
	}

	// Gradient for B1
	for j := 0; j < hiddenDim; j++ {
		original := expert.B1[j]
		expert.B1[j] = original + epsilon
		expertOutPlus := expert.Forward([][]float64{x})
		lossPlus := sumSquaredDiffRow(expertOutPlus[0], gradOutput)
		
		expert.B1[j] = original - epsilon
		expertOutMinus := expert.Forward([][]float64{x})
		lossMinus := sumSquaredDiffRow(expertOutMinus[0], gradOutput)
		
		expert.B1[j] = original
		grad := (lossPlus - lossMinus) / (2 * epsilon)
		expert.B1[j] -= lr * grad
	}
}

// Helper functions
func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for _, v := range x {
		sum += math.Exp(v - maxVal)
	}

	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal) / sum
	}
	return result
}

func sumSquaredDiff(a, b [][]float64) float64 {
	sum := 0.0
	for i := range a {
		for j := range a[i] {
			diff := a[i][j] - b[i][j]
			sum += diff * diff
		}
	}
	return sum
}

func sumSquaredDiffRow(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// GetMoEStats returns routing statistics for analysis
func (m *MoELayer) GetMoEStats() map[string]float64 {
	stats := make(map[string]float64)
	
	totalLoad := 0.0
	maxLoad := 0.0
	minLoad := float64(m.expertCaps[0])
	
	for _, cap := range m.expertCaps {
		totalLoad += float64(cap)
		if float64(cap) > maxLoad {
			maxLoad = float64(cap)
		}
		if float64(cap) < minLoad {
			minLoad = float64(cap)
		}
	}
	
	stats["total_load"] = totalLoad
	stats["max_load"] = maxLoad
	stats["min_load"] = minLoad
	stats["avg_load"] = totalLoad / float64(len(m.expertCaps))
	stats["load_balance"] = maxLoad - minLoad
	
	return stats
}
