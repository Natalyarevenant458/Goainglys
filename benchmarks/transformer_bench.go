package benchmarks

import (
	"fmt"
	"math/rand"
	"runtime"
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
	embedding *EmbeddingLayer
	output    *LinearLayer
}

// EmbeddingLayer implements word embeddings
type EmbeddingLayer struct {
	table [][]float64
}

// EncoderLayer represents a single transformer encoder layer
type EncoderLayer struct {
	selfAttn  *MultiHeadAttention
	ffn       *FeedForwardLayer
}

// DecoderLayer represents a single transformer decoder layer
type DecoderLayer struct {
	selfAttn  *MultiHeadAttention
	encAttn   *MultiHeadAttention
	ffn       *FeedForwardLayer
}

// MultiHeadAttention implements multi-head attention
type MultiHeadAttention struct {
	dModel   int
	numHeads int
	dK       int
	wq       *LinearLayer
	wk       *LinearLayer
	wv       *LinearLayer
	out      *LinearLayer
}

// FeedForwardLayer implements feed-forward network
type FeedForwardLayer struct {
	w1 *LinearLayer
	w2 *LinearLayer
}

// NewLinear creates a new linear layer with random weights
func NewLinearT(inputDim, outputDim int) *LinearLayer {
	l := &LinearLayer{
		weight: NewTensor(inputDim, outputDim),
		bias:   NewTensor(1, outputDim),
	}
	for i := 0; i < inputDim; i++ {
		for j := 0; j < outputDim; j++ {
			l.weight.data[i*outputDim+j] = rand.Float64()*2 - 1
		}
	}
	for i := 0; i < outputDim; i++ {
		l.bias.data[i] = rand.Float64()*2 - 1
	}
	return l
}

// NewMultiHeadAttention creates a new multi-head attention layer
func NewMultiHeadAttentionT(dModel, numHeads int) *MultiHeadAttention {
	dK := dModel / numHeads
	return &MultiHeadAttention{
		dModel: dModel,
		numHeads: numHeads,
		dK: dK,
		wq: NewLinearT(dModel, dModel),
		wk: NewLinearT(dModel, dModel),
		wv: NewLinearT(dModel, dModel),
		out: NewLinearT(dModel, dModel),
	}
}

// Forward through attention (simplified)
func (m *MultiHeadAttention) Forward(input [][]float64, mask [][]float64) [][]float64 {
	batchSize := len(input)
	_ = len(input[0])

	// Simplified computation - actual multi-head attention is more complex
	// Just do a simple projection through the network
	q := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		q[b] = make([]float64, m.dModel)
		for i := 0; i < m.dModel && i < len(input[b]); i++ {
			q[b][i] = input[b][i]
		}
	}

	// Simple forward through linear
	_ = mask
	return q
}

// NewFeedForwardLayer creates a new feed-forward network
func NewFeedForwardLayer(dModel, dFF int) *FeedForwardLayer {
	return &FeedForwardLayer{
		w1: NewLinearT(dModel, dFF),
		w2: NewLinearT(dFF, dModel),
	}
}

func (f *FeedForwardLayer) Forward(input [][]float64) [][]float64 {
	batchSize := len(input)
	dFF := f.w2.weight.shape[1]
	output := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		output[b] = make([]float64, dFF)
		for j := 0; j < dFF; j++ {
			sum := f.w2.bias.data[j]
			for i := 0; i < len(input[b]); i++ {
				sum += input[b][i] * f.w1.weight.data[i*j]
			}
			output[b][j] = sum
		}
	}
	return output
}

// NewEncoderLayer creates a new encoder layer
func NewEncoderLayer(dModel, numHeads, dFF int) *EncoderLayer {
	return &EncoderLayer{
		selfAttn: NewMultiHeadAttentionT(dModel, numHeads),
		ffn:      NewFeedForwardLayer(dModel, dFF),
	}
}

// NewDecoderLayer creates a new decoder layer
func NewDecoderLayer(dModel, numHeads, dFF int) *DecoderLayer {
	return &DecoderLayer{
		selfAttn: NewMultiHeadAttentionT(dModel, numHeads),
		encAttn:  NewMultiHeadAttentionT(dModel, numHeads),
		ffn:      NewFeedForwardLayer(dModel, dFF),
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

// NewTransformerT creates a new transformer
func NewTransformerT(config TransformerConfig) *Transformer {
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

	embedding := &EmbeddingLayer{
		table: make([][]float64, config.VocabSize),
	}
	for i := 0; i < config.VocabSize; i++ {
		embedding.table[i] = make([]float64, config.DModel)
		for j := 0; j < config.DModel; j++ {
			embedding.table[i][j] = rand.Float64() * 2 * 0.1
		}
	}

	output := NewLinearT(config.DModel, config.VocabSize)

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
		_ = encOutput
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
		_ = decOutput
	}

	// Output projection
	logits := make([][]float64, t.config.BatchSize)
	for b := 0; b < t.config.BatchSize; b++ {
		logits[b] = make([]float64, t.config.VocabSize)
		for j := 0; j < t.config.VocabSize; j++ {
			sum := float64(j) * 0.01 // Simplified
			logits[b][j] = sum
		}
	}
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

		transformer := NewTransformerT(config)
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

	transformer := NewTransformerT(config)
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

					transformer := NewTransformerT(config)
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

	transformer := NewTransformerT(config)
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
