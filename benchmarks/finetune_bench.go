package benchmarks

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// LoRAConfig holds LoRA configuration
type LoRAConfig struct {
	R         int      // Rank
	Alpha     float64  // Scaling factor
	Dropout   float64  // Dropout rate
	TargetModules []string
}

// LoRALayer implements Low-Rank Adaptation
type LoRALayer struct {
	r       int
	alpha   float64
	loraA   [][]float64 // Down projection
	loraB   [][]float64 // Up projection
	scaling float64
}

// NewLoRALayer creates a new LoRA layer
func NewLoRALayer(inputDim, outputDim, r int, alpha float64) *LoRALayer {
	lora := &LoRALayer{
		r:       r,
		alpha:   alpha,
		loraA:   make([][]float64, inputDim),
		loraB:   make([][]float64, r),
		scaling: alpha / float64(r),
	}

	for i := 0; i < inputDim; i++ {
		lora.loraA[i] = make([]float64, r)
		for j := 0; j < r; j++ {
			lora.loraA[i][j] = rand.Float64() * 0.01
		}
	}

	for i := 0; i < r; i++ {
		lora.loraB[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			lora.loraB[i][j] = rand.Float64() * 0.01
		}
	}

	return lora
}

// Forward performs LoRA forward pass
func (l *LoRALayer) Forward(input [][]float64, baseOutput [][]float64) [][]float64 {
	batchSize := len(input)
	outputDim := len(baseOutput[0])

	// Compute LoRA adjustment: input @ loraA @ loraB
	// input: [batch, inputDim]
	// loraA: [inputDim, r]
	// loraB: [r, outputDim]

	// First matmul: input @ loraA -> [batch, r]
	intermediate := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		intermediate[b] = make([]float64, l.r)
		inputDim := len(input[b])
		for rIdx := 0; rIdx < l.r; rIdx++ {
			sum := 0.0
			for i := 0; i < inputDim; i++ {
				sum += input[b][i] * l.loraA[i][rIdx]
			}
			intermediate[b][rIdx] = sum
		}
	}

	// Second matmul: intermediate @ loraB -> [batch, outputDim]
	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, outputDim)
		for o := 0; o < outputDim; o++ {
			sum := 0.0
			for rIdx := 0; rIdx < l.r; rIdx++ {
				sum += intermediate[b][rIdx] * l.loraB[rIdx][o]
			}
			// Add to base output with scaling
			output[b][o] = baseOutput[b][o] + sum*l.scaling
		}
	}

	return output
}

// Backward performs LoRA backward pass (simplified)
func (l *LoRALayer) Backward(input, gradOutput [][]float64) {
	batchSize := len(input)
	inputDim := len(input[0])
	outputDim := len(gradOutput[0])

	// Simplified gradient computation
	// In real implementation, would compute gradients for loraA and loraB
	
	// Compute dL/d(loraB)
	gradB := make([][]float64, l.r)
	for rIdx := 0; rIdx < l.r; rIdx++ {
		gradB[rIdx] = make([]float64, outputDim)
	}

	// Compute intermediate (input @ loraA)
	intermediate := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		intermediate[b] = make([]float64, l.r)
		for rIdx := 0; rIdx < l.r; rIdx++ {
			sum := 0.0
			for i := 0; i < inputDim; i++ {
				sum += input[b][i] * l.loraA[i][rIdx]
			}
			intermediate[b][rIdx] = sum
		}
	}

	// Gradient accumulation
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputDim; o++ {
			gradVal := gradOutput[b][o] * l.scaling
			for rIdx := 0; rIdx < l.r; rIdx++ {
				gradB[rIdx][o] += gradVal * intermediate[b][rIdx]
			}
		}
	}

	// Update loraB (simplified - would use optimizer in real code)
	for rIdx := 0; rIdx < l.r; rIdx++ {
		for o := 0; o < outputDim; o++ {
			l.loraB[rIdx][o] -= 0.001 * gradB[rIdx][o] / float64(batchSize)
		}
	}
}

// GPT2LoRA applies LoRA to GPT-2 small
type GPT2LoRA struct {
	attention *LoRALayer
	mlp       *LoRALayer
	config    LoRAConfig
}

// NewGPT2LoRA creates a GPT-2 model with LoRA
func NewGPT2LoRA(r int, alpha float64) *GPT2LoRA {
	// GPT-2 small: 768 hidden, 12 layers
	// Attention: 768 -> 768 (Q, K, V would be combined, we simplify)
	// MLP: 768 -> 3072 -> 768

	return &GPT2LoRA{
		attention: NewLoRALayer(768, 768, r, alpha),
		mlp:       NewLoRALayer(768, 768, r, alpha),
		config:    LoRAConfig{R: r, Alpha: alpha},
	}
}

// Forward performs forward pass through GPT-2 with LoRA
func (g *GPT2LoRA) Forward(input [][]float64) [][]float64 {
	// Simplified forward - just apply LoRA layers
	batchSize := len(input)
	outputDim := 768

	// Base output (would be full model computation in real case)
	baseOutput := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		baseOutput[b] = make([]float64, outputDim)
		for i := 0; i < outputDim; i++ {
			baseOutput[b][i] = input[b][i%len(input[0])] * 0.5
		}
	}

	// Apply LoRA to attention output
	attentionOutput := g.attention.Forward(input, baseOutput)

	// Apply LoRA to MLP output
	mlpOutput := g.mlp.Forward(input, baseOutput)

	// Combine (simplified)
	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, outputDim)
		for i := 0; i < outputDim; i++ {
			output[b][i] = attentionOutput[b][i] + mlpOutput[b][i]
		}
	}

	return output
}

// AllReduceRing performs ring all-reduce
func AllReduceRing(data []float64, numNodes int, rank int) {
	// Simplified ring all-reduce
	// In real implementation, would do actual MPI-style communication
	
	n := len(data)
	steps := numNodes - 1
	
	for step := 0; step < steps; step++ {
		// Determine send/receive ranks
		sendRank := (rank + 1) % numNodes
		recvRank := (rank - 1 + numNodes) % numNodes
		_ = sendRank
		_ = recvRank
		
		// Simulate communication overhead
		// In reality, this would be actual MPI_Send/MPI_Recv
		
		// Sum reduction
		for i := 0; i < n; i++ {
			data[i] += data[i] * 0.001 // Simplified reduction
		}
		
		step++
	}
}

// DistributedAllReduce benchmarks distributed all-reduce
func DistributedAllReduce(numNodes, rank int, dataSizeMB float64) {
	dataSize := int(dataSizeMB * 1024 * 1024 / 8) // float64 count
	data := make([]float64, dataSize)
	for i := range data {
		data[i] = rand.Float64()
	}

	// Warmup
	AllReduceRing(data, numNodes, rank)

	// Benchmark
	iterations := 10
	var totalTime time.Duration
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		AllReduceRing(data, numNodes, rank)
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	_ = totalTime.Seconds() / float64(iterations)
}

// FinetuneBenchmarkResults holds fine-tuning benchmark results
type FinetuneBenchmarkResults struct {
	TestName       string
	Metric         string
	Value          float64
	LatencyMs      float64
	ComparedToFull float64
}

func RunFinetuneBenchmarks() []FinetuneBenchmarkResults {
	results := []FinetuneBenchmarkResults{}

	// LoRA update benchmark at different ranks
	ranks := []int{4, 8, 16, 32}

	for _, r := range ranks {
		alpha := float64(r) * 2
		model := NewGPT2LoRA(r, alpha)

		// Input: batch 4, seq 32, hidden 768
		input := make([][]float64, 4)
		for b := 0; b < 4; b++ {
			input[b] = make([]float64, 768)
			for i := 0; i < 768; i++ {
				input[b][i] = rand.Float64()*2 - 1
			}
		}

		fn := func() {
			output := model.Forward(input)
			
			// Create dummy gradients
			gradOutput := make([][]float64, 4)
			for b := 0; b < 4; b++ {
				gradOutput[b] = make([]float64, 768)
				for i := 0; i < 768; i++ {
					gradOutput[b][i] = rand.Float64()*2 - 1
				}
			}

			// Backward pass
			model.attention.Backward(input, gradOutput)
			model.mlp.Backward(input, gradOutput)
			_ = output
		}

		fn() // warmup

		iterations := 10
		var totalTime time.Duration
		for i := 0; i < iterations; i++ {
			t0 := time.Now()
			fn()
			t1 := time.Now()
			totalTime += t1.Sub(t0)
		}

		avgTime := totalTime.Seconds() / float64(iterations)
		results = append(results, FinetuneBenchmarkResults{
			TestName:   fmt.Sprintf("LoRA R=%d", r),
			Metric:     "Updates/sec",
			Value:      1.0 / avgTime,
			LatencyMs:  avgTime * 1000,
		})
	}

	// Compare with full fine-tuning (simulated)
	// Full fine-tuning would update ~124M params (GPT-2 small)
	// LoRA updates only ~3M params (for R=32)
	
	// Simulated full fine-tuning time
	ranks = []int{4, 8, 16, 32}
	for _, r := range ranks {
		// Full: 124M params, LoRA: (768*R + R*768) = ~5K params per layer
		// 12 layers = ~60K params for R=32, ~15K for R=8
		// Speedup is roughly proportional to parameter reduction
		paramRatio := float64(124*1000000) / float64(12*(768*r+r*768))
		
		alpha := float64(r) * 2
		model := NewGPT2LoRA(r, alpha)
		
		input := make([][]float64, 4)
		for b := 0; b < 4; b++ {
			input[b] = make([]float64, 768)
			for i := 0; i < 768; i++ {
				input[b][i] = rand.Float64()*2 - 1
			}
		}
		
		fn := func() {
			output := model.Forward(input)
			_ = output
		}
		fn()
		
		iterations := 20
		var totalTime time.Duration
		for i := 0; i < iterations; i++ {
			t0 := time.Now()
			fn()
			t1 := time.Now()
			totalTime += t1.Sub(t0)
		}
		
		avgTime := totalTime.Seconds() / float64(iterations)
		estimatedFullTime := avgTime * paramRatio
		
		results = append(results, FinetuneBenchmarkResults{
			TestName:       fmt.Sprintf("Full Fine-tune (est) R=%d", r),
			Metric:         "Speedup vs Full",
			Value:          paramRatio,
			LatencyMs:      estimatedFullTime * 1000,
			ComparedToFull: paramRatio,
		})
	}

	// Distributed all-reduce benchmark
	fn := func() { DistributedAllReduce(4, 0, 100) }
	fn() // warmup

	iterations := 5
	var totalTime time.Duration
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime := totalTime.Seconds() / float64(iterations)
	results = append(results, FinetuneBenchmarkResults{
		TestName:   "AllReduce Ring (100MB, 4 nodes)",
		Metric:     "Time (s)",
		Value:      avgTime,
		LatencyMs:  avgTime * 1000,
	})

	return results
}

// BenchmarkLoRAUpdate benchmarks LoRA update at different ranks
func BenchmarkLoRAUpdate(b *testing.B) {
	ranks := []int{4, 8, 16, 32}

	for _, r := range ranks {
		b.Run(fmt.Sprintf("rank_%d", r), func(b *testing.B) {
			alpha := float64(r) * 2
			model := NewGPT2LoRA(r, alpha)

			input := make([][]float64, 4)
			for i := range input {
				input[i] = make([]float64, 768)
				for j := range input[i] {
					input[i][j] = rand.Float64()*2 - 1
				}
			}

			gradOutput := make([][]float64, 4)
			for i := range gradOutput {
				gradOutput[i] = make([]float64, 768)
				for j := range gradOutput[i] {
					gradOutput[i][j] = rand.Float64()*2 - 1
				}
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				model.Forward(input)
				model.attention.Backward(input, gradOutput)
				model.mlp.Backward(input, gradOutput)
			}
		})
	}
}

// BenchmarkDistributedAllReduce benchmarks ring all-reduce
func BenchmarkDistributedAllReduce(b *testing.B) {
	dataSizesMB := []float64{10, 50, 100, 500}
	numNodes := 4

	for _, sizeMB := range dataSizesMB {
		b.Run(fmt.Sprintf("size_%dMB", int(sizeMB)), func(b *testing.B) {
			dataSize := int(sizeMB * 1024 * 1024 / 8)
			data := make([]float64, dataSize)
			for i := range data {
				data[i] = rand.Float64()
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				AllReduceRing(data, numNodes, 0)
			}
		})
	}
}

// BenchmarkFullVsLoRA compares full fine-tuning with LoRA
func BenchmarkFullVsLoRA(b *testing.B) {
	// Simulate full fine-tuning overhead vs LoRA
	b.Log("Note: Full fine-tuning is estimated based on parameter count ratio")
	b.Log("Actual full fine-tuning would require running full GPT-2 model")

	// Just run LoRA to have a baseline
	r := 8
	alpha := float64(r) * 2
	model := NewGPT2LoRA(r, alpha)

	input := make([][]float64, 4)
	for i := range input {
		input[i] = make([]float64, 768)
		for j := range input[i] {
			input[i][j] = rand.Float64()*2 - 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = model.Forward(input)
	}
}
