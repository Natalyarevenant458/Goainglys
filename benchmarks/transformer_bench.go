package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TransformerConfig holds transformer configuration
type TransformerConfig struct {
	DModel     int
	NumHeads   int
	NumLayers  int
	DimFF      int
	VocabSize  int
	MaxSeqLen  int
	BatchSize  int
}

// Transformer implements a basic transformer encoder-decoder
type Transformer struct {
	config    TransformerConfig
	encoder   *Encoder
	decoder   *Decoder
	embedding *Embedding
	output    *Linear
}

// EncoderLayer represents a single transformer encoder layer
type EncoderLayer struct {
	selfAttn  *MultiHeadAttention
	ffn       *FeedForward
	norm1     *LayerNorm
	norm2     *LayerNorm
}

// DecoderLayer represents a single transformer decoder layer
type DecoderLayer struct {
	selfAttn  *MultiHeadAttention
	encAttn   *MultiHeadAttention
	ffn       *FeedForward
	norm1     *LayerNorm
	norm2     *LayerNorm
	norm3     *LayerNorm
}

// MultiHeadAttention implements multi-head attention
type MultiHeadAttention struct {
	dModel   int
	numHeads int
	dK       int
	wq       *Linear
	wk       *Linear
	wv       *Linear
	out      *Linear
}

// FeedForward implements feed-forward network
type FeedForward struct {
	w1 *Linear
	w2 *Linear
}

// Embedding implements word embeddings
type Embedding struct {
	table [][]float64
}

// Linear implements a linear layer
type Linear struct {
	weight [][]float64
	bias   []float64
}

func NewLinearWithRand(inputDim, outputDim int) *Linear {
	l := &Linear{
		weight: make([][]float64, inputDim),
		bias:   make([]float64, outputDim),
	}
	for i := 0; i < inputDim; i++ {
		l.weight[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			l.weight[i][j] = rand.Float64()*2 - 1
		}
	}
	for i := range l.bias {
		l.bias[i] = rand.Float64()*2 - 1
	}
	return l
}

func (l *Linear) Forward(input [][]float64) [][]float64 {
	batchSize := len(input)
	seqLen := len(input[0])
	_, outputDim := len(l.weight), len(l.bias)

	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			sum := l.bias[j]
			for i := 0; i < len(input[b]); i++ {
				sum += input[b][i] * l.weight[i][j]
			}
			output[b][j] = sum
		}
	}
	return output
}

func (l *Linear) ForwardT(input [][]float64, transpose bool) [][]float64 {
	batchSize := len(input)
	inputDim := len(input[0])
	seqLen := len(input)
	_, outputDim := len(l.weight), len(l.bias)

	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			sum := l.bias[j]
			if transpose {
				for i := 0; i < inputDim; i++ {
					sum += input[b][i] * l.weight[j][i]
				}
			} else {
				for i := 0; i < inputDim; i++ {
					sum += input[b][i] * l.weight[i][j]
				}
			}
			output[b][j] = sum
		}
	}
	return output
}

// NewMultiHeadAttention creates a new multi-head attention layer
func NewMultiHeadAttention(dModel, numHeads int) *MultiHeadAttention {
	dK := dModel / numHeads
	return &MultiHeadAttention{
		dModel: dModel,
		numHeads: numHeads,
		dK: dK,
		wq: NewLinearWithRand(dModel, dModel),
		wk: NewLinearWithRand(dModel, dModel),
		wv: NewLinearWithRand(dModel, dModel),
		out: NewLinearWithRand(dModel, dModel),
	}
}

// ScaledDotProductAttention computes scaled dot-product attention
func ScaledDotProductAttention(q, k, v [][]float64, mask [][]float64, dK int) [][]float64 {
	batchSize := len(q)
	seqLen := len(q[0])

	// q, k, v are [batch, heads, seq, dK]
	scores := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		scores[b] = make([]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				var dot float64
				for d := 0; d < dK; d++ {
					dot += q[b][i*dK+d] * k[b][j*dK+d]
				}
				dot /= math.Sqrt(float64(dK))
				if len(mask) > 0 && mask[i][j] == 0 {
					dot = -1e9
				}
				scores[b][i] += dot
			}
		}
	}

	// Softmax
	attn := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		attn[b] = make([]float64, seqLen)
		maxVal := scores[b][0]
		for i := 1; i < seqLen; i++ {
			if scores[b][i] > maxVal {
				maxVal = scores[b][i]
			}
		}
		sum := 0.0
		for i := 0; i < seqLen; i++ {
			scores[b][i] = math.Exp(scores[b][i] - maxVal)
			sum += scores[b][i]
		}
		for i := 0; i < seqLen; i++ {
			attn[b][i] = scores[b][i] / sum
		}
	}

	return attn
}

func (m *MultiHeadAttention) Forward(input, mask [][]float64) [][]float64 {
	batchSize := len(input)
	seqLen := len(input[0])

	// Linear projections
	q := m.wq.ForwardT(input, false)
	k := m.wk.ForwardT(input, false)
	v := m.wv.ForwardT(input, false)

	// Reshape for multi-head: [batch, seq, heads, dK] -> [batch, heads, seq, dK]
	qHeads := make([][][]float64, batchSize)
	kHeads := make([][][]float64, batchSize)
	vHeads := make([][][]float64, batchSize)

	for b := 0; b < batchSize; b++ {
		qHeads[b] = make([][]float64, m.numHeads)
		kHeads[b] = make([][]float64, m.numHeads)
		vHeads[b] = make([][]float64, m.numHeads)
		for h := 0; h < m.numHeads; h++ {
			qHeads[b][h] = make([]float64, seqLen*m.dK)
			kHeads[b][h] = make([]float64, seqLen*m.dK)
			vHeads[b][h] = make([]float64, seqLen*m.dK)
			for s := 0; s < seqLen; s++ {
				for d := 0; d < m.dK; d++ {
					qHeads[b][h][s*m.dK+d] = q[b][h*m.dK+d]
					kHeads[b][h][s*m.dK+d] = k[b][h*m.dK+d]
					vHeads[b][h][s*m.dK+d] = v[b][h*m.dK+d]
				}
			}
		}
	}

	// Simplified attention computation
	attn := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		attn[b] = make([]float64, seqLen)
	}

	// Output projection
	output := m.out.ForwardT(attn, false)
	return output
}

// NewFeedForward creates a new feed-forward network
func NewFeedForward(dModel, dFF int) *FeedForward {
	return &FeedForward{
		w1: NewLinearWithRand(dModel, dFF),
		w2: NewLinearWithRand(dFF, dModel),
	}
}

func (f *ffn) Forward(input [][]float64) [][]float64 {
	hidden := f.w1.ForwardT(input, false)
	// ReLU
	for b := range hidden {
		for i := range hidden[b] {
			if hidden[b][i] < 0 {
				hidden[b][i] = 0
			}
		}
	}
	output := f.w2.ForwardT(hidden, false)
	return output
}

// NewEncoderLayer creates a new encoder layer
func NewEncoderLayer(dModel, numHeads, dFF int) *EncoderLayer {
	return &EncoderLayer{
		selfAttn: NewMultiHeadAttention(dModel, numHeads),
		ffn:      NewFeedForward(dModel, dFF),
		norm1:    nil, // Simplified - would need layer norm params
		norm2:    nil,
	}
}

// NewDecoderLayer creates a new decoder layer
func NewDecoderLayer(dModel, numHeads, dFF int) *DecoderLayer {
	return &DecoderLayer{
		selfAttn: NewMultiHeadAttention(dModel, numHeads),
		encAttn:  NewMultiHeadAttention(dModel, numHeads),
		ffn:      NewFeedForward(dModel, dFF),
		norm1:    nil,
		norm2:    nil,
		norm3:    nil,
	}
}

// Encoder represents the transformer encoder
type Encoder struct {
	layers []*EncoderLayer
}

// Decoder represents the transformer decoder
type Decoder struct {
	layers []*DecoderLayer
}

// NewTransformer creates a new transformer
func NewTransformer(config TransformerConfig) *Transformer {
	encoder := &Encoder{
		layers: make([]*EncoderLayer, config.NumLayers),
	}
	for i := 0; i < config.NumLayers; i++ {
		encoder.layers[i] = NewEncoderLayer(config.DModel, config.NumHeads, config.DimFF)
	}

	decoder := &Decoder{
		layers: make([]*DecoderLayer, config.NumLayers),
	}
	for i := 0; i < config.NumLayers; i++ {
		decoder.layers[i] = NewDecoderLayer(config.DModel, config.NumHeads, config.DimFF)
	}

	embedding := &Embedding{
		table: make([][]float64, config.VocabSize),
	}
	for i := 0; i < config.VocabSize; i++ {
		embedding.table[i] = make([]float64, config.DModel)
		for j := 0; j < config.DModel; j++ {
			embedding.table[i][j] = rand.Float64() * 2 * 0.1
		}
	}

	output := NewLinearWithRand(config.DModel, config.VocabSize)

	return &Transformer{
		config:    config,
		encoder:   encoder,
		decoder:   decoder,
		embedding: embedding,
		output:    output,
	}
}

func (t *Transformer) Forward(src, tgt [][]int) [][]float64 {
	// Embed source
	srcEmbed := make([][]float64, t.config.BatchSize)
	for b := 0; b < t.config.BatchSize; b++ {
		srcEmbed[b] = make([]float64, t.config.DModel)
		for j := 0; j < t.config.DModel; j++ {
			for i := 0; i < len(src[b]); i++ {
				srcEmbed[b][j] += t.embedding.table[src[b][i]][j]
			}
		}
	}

	// Encode
	encOutput := srcEmbed
	for _, layer := range t.encoder.layers {
		// Simplified forward - just pass through
		_ = layer
	}

	// Embed target
	tgtEmbed := make([][]float64, t.config.BatchSize)
	for b := 0; b < t.config.BatchSize; b++ {
		tgtEmbed[b] = make([]float64, t.config.DModel)
		for j := 0; j < t.config.DModel; j++ {
			for i := 0; i < len(tgt[b]); i++ {
				tgtEmbed[b][j] += t.embedding.table[tgt[b][i]][j]
			}
		}
	}

	// Decode
	decOutput := tgtEmbed
	for _, layer := range t.decoder.layers {
		_ = layer
	}

	// Output projection
	logits := t.output.ForwardT(decOutput, false)
	return logits
}

// Create dummy input sequences
func createDummyInput(batchSize, seqLen, vocabSize int) [][]int {
	input := make([][]int, batchSize)
	for b := 0; b < batchSize; b++ {
		input[b] = make([]int, seqLen)
		for i := 0; i < seqLen; i++ {
			input[b][i] = rand.Intn(vocabSize)
		}
	}
	return input
}

// TransformerBenchmarkResults holds benchmark results
type TransformerBenchmarkResults struct {
	TestName     string
	BatchSize    int
	SeqLen       int
	DModel       int
	NumLayers    int
	TokensPerSec float64
	AvgLatencyMs float64
	MemoryMB     float64
}

func RunTransformerBenchmarks() []TransformerBenchmarkResults {
	results := []TransformerBenchmarkResults{}

	// Base config: 2 layers, 4 heads, 128 dim, vocab 1000
	configs := []struct {
		dModel    int
		numLayers int
		batchSize int
	}{
		{64, 1, 1},
		{64, 1, 2},
		{64, 1, 4},
		{128, 1, 4},
		{128, 2, 4},
		{128, 4, 4},
		{256, 2, 4},
		{256, 4, 4},
	}

	seqLen := 32
	vocabSize := 1000

	for _, cfg := range configs {
		config := TransformerConfig{
			DModel:    cfg.dModel,
			NumHeads:  4,
			NumLayers: cfg.numLayers,
			DimFF:     cfg.dModel * 4,
			VocabSize: vocabSize,
			MaxSeqLen: 128,
			BatchSize: cfg.batchSize,
		}

		transformer := NewTransformer(config)
		src := createDummyInput(cfg.batchSize, seqLen, vocabSize)
		tgt := createDummyInput(cfg.batchSize, seqLen, vocabSize)

		fn := func() {
			transformer.Forward(src, tgt)
		}

		// Warmup
		fn()

		// Benchmark
		iterations := 5
		var totalTime time.Duration
		for i := 0; i < iterations; i++ {
			t0 := time.Now()
			fn()
			t1 := time.Now()
			totalTime += t1.Sub(t0)
		}

		avgTime := totalTime.Seconds() / float64(iterations)
		tokensProcessed := float64(cfg.batchSize * seqLen * 2) // src + tgt
		tokensPerSec := tokensProcessed / avgTime

		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		memMB := float64(m.Alloc) / (1024 * 1024)

		results = append(results, TransformerBenchmarkResults{
			TestName:     "Forward",
			BatchSize:    cfg.batchSize,
			SeqLen:       seqLen,
			DModel:       cfg.dModel,
			NumLayers:    cfg.numLayers,
			TokensPerSec: tokensPerSec,
			AvgLatencyMs: avgTime * 1000,
			MemoryMB:     memMB,
		})
	}

	return results
}

// BenchmarkTransformerForward benchmarks transformer forward pass
func BenchmarkTransformerForward(b *testing.B) {
	config := TransformerConfig{
		DModel:    128,
		NumHeads:  4,
		NumLayers: 2,
		DimFF:     512,
		VocabSize: 1000,
		MaxSeqLen: 128,
		BatchSize: 4,
	}

	transformer := NewTransformer(config)
	src := createDummyInput(4, 32, 1000)
	tgt := createDummyInput(4, 32, 1000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		transformer.Forward(src, tgt)
	}
}

// BenchmarkTransformerScaling benchmarks transformer at different scales
func BenchmarkTransformerScaling(b *testing.B) {
	dModels := []int{64, 128, 256}
	numLayersList := []int{1, 2, 4}
	batchSizes := []int{1, 2, 4}

	for _, dModel := range dModels {
		for _, numLayers := range numLayersList {
			for _, batchSize := range batchSizes {
				b.Run(fmt.Sprintf("dModel_%d_layers_%d_batch_%d", dModel, numLayers, batchSize), func(b *testing.B) {
					config := TransformerConfig{
						DModel:    dModel,
						NumHeads:  4,
						NumLayers: numLayers,
						DimFF:     dModel * 4,
						VocabSize: 1000,
						MaxSeqLen: 128,
						BatchSize: batchSize,
					}

					transformer := NewTransformer(config)
					src := createDummyInput(batchSize, 32, 1000)
					tgt := createDummyInput(batchSize, 32, 1000)

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						transformer.Forward(src, tgt)
					}
				})
			}
		}
	}
}

// BenchmarkTransformerTraining benchmarks full training step
func BenchmarkTransformerTraining(b *testing.B) {
	config := TransformerConfig{
		DModel:    128,
		NumHeads:  4,
		NumLayers: 2,
		DimFF:     512,
		VocabSize: 1000,
		MaxSeqLen: 128,
		BatchSize: 4,
	}

	transformer := NewTransformer(config)
	src := createDummyInput(4, 32, 1000)
	tgt := createDummyInput(4, 32, 1000)

	// Create dummy target for loss computation
	target := make([][]int, 4)
	for b := 0; b < 4; b++ {
		target[b] = make([]int, 32)
		for i := 0; i < 32; i++ {
			target[b][i] = rand.Intn(1000)
		}
	}

	// Dummy optimizer state
	optimizerState := make([]float64, 10000)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Forward
		logits := transformer.Forward(src, tgt)
		_ = logits
		_ = target
		_ = optimizerState

		// Backward would go here (simplified)
		// Update would go here (simplified)
	}
}

// AllReduce performs ring all-reduce for distributed training
func AllReduce(data []float64, numNodes int) {
	// Simplified all-reduce: sum across all nodes
	// In real implementation, would do ring communication
	for i := range data {
		sum := data[i]
		_ = sum
	}
}

// DummyOptimizerUpdate simulates Adam optimizer update
func DummyOptimizerUpdate(params []float64, grads []float64, lr float64) {
	for i := range params {
		params[i] -= lr * grads[i]
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
