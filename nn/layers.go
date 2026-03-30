package nn

import (
	"math"
	"math/rand"
)

// Linear implements a fully-connected layer: y = x @ W + b
type Linear struct {
	Weight      []float64
	Bias        []float64
	GradW       []float64
	GradB       []float64
	InFeatures  int
	OutFeatures int
}

// NewLinear creates a linear layer with He initialization.
func NewLinear(inFeatures, outFeatures int) *Linear {
	w := make([]float64, inFeatures*outFeatures)
	scale := math.Sqrt(2.0 / float64(inFeatures))
	for i := range w {
		w[i] = rand.NormFloat64() * scale
	}
	b := make([]float64, outFeatures)
	return &Linear{
		Weight:      w,
		Bias:        b,
		GradW:       make([]float64, inFeatures*outFeatures),
		GradB:       make([]float64, outFeatures),
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

// Forward computes y = x @ W + b for batched input.
// x shape: [batch, seq, inFeatures] or [batch*seq, inFeatures]
// returns y shape: [batch, seq, outFeatures] or [batch*seq, outFeatures]
func (l *Linear) Forward(x []float64, batch, seq, inF, outF int) []float64 {
	if inF != l.InFeatures || outF != l.OutFeatures {
		panic("Linear: dimension mismatch")
	}
	result := make([]float64, batch*seq*outF)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for o := 0; o < outF; o++ {
				sum := l.Bias[o]
				for i := 0; i < inF; i++ {
					idx := b*seq*inF + s*inF + i
					wIdx := i*outF + o
					sum += x[idx] * l.Weight[wIdx]
				}
				result[b*seq*outF+s*outF+o] = sum
			}
		}
	}
	return result
}

// Backward computes gradients w.r.t. W, b, and input.
// gradOut shape: [batch, seq, outFeatures]
// Returns gradIn shape: [batch, seq, inFeatures]
func (l *Linear) Backward(x, gradOut []float64, batch, seq, inF, outF int) []float64 {
	// gradW = x^T @ gradOut
	for i := 0; i < inF; i++ {
		for o := 0; o < outF; o++ {
			sum := 0.0
			for b := 0; b < batch; b++ {
				for s := 0; s < seq; s++ {
					idx := b*seq*inF + s*inF + i
					gIdx := b*seq*outF + s*outF + o
					sum += x[idx] * gradOut[gIdx]
				}
			}
			l.GradW[i*outF+o] = sum
		}
	}

	// gradB = sum over batch/seq of gradOut
	for o := 0; o < outF; o++ {
		sum := 0.0
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				idx := b*seq*outF + s*outF + o
				sum += gradOut[idx]
			}
		}
		l.GradB[o] = sum
	}

	// gradIn = gradOut @ W^T
	gradIn := make([]float64, batch*seq*inF)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for i := 0; i < inF; i++ {
				sum := 0.0
				for o := 0; o < outF; o++ {
					gIdx := b*seq*outF + s*outF + o
					wIdx := i*outF + o
					sum += gradOut[gIdx] * l.Weight[wIdx]
				}
				gradIn[b*seq*inF+s*inF+i] = sum
			}
		}
	}

	return gradIn
}

// Embedding is a simple lookup table.
type Embedding struct {
	Weight  []float64
	VocSize int
	Dim     int
}

// NewEmbedding creates an embedding table.
func NewEmbedding(vocSize, dim int) *Embedding {
	w := make([]float64, vocSize*dim)
	scale := math.Sqrt(1.0 / float64(dim))
	for i := range w {
		w[i] = rand.NormFloat64() * scale
	}
	return &Embedding{
		Weight:  w,
		VocSize: vocSize,
		Dim:     dim,
	}
}

// Forward looks up embedding vectors for token indices.
// tokens shape: [batch, seq], returns [batch, seq, dim]
func (e *Embedding) Forward(tokens []int, batch, seq int) []float64 {
	result := make([]float64, batch*seq*e.Dim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			idx := tokens[b*seq+s]
			if idx < 0 || idx >= e.VocSize {
				continue
			}
			for d := 0; d < e.Dim; d++ {
				result[b*seq*e.Dim+s*e.Dim+d] = e.Weight[idx*e.Dim+d]
			}
		}
	}
	return result
}

// LayerNorm applies layer normalization.
type LayerNorm struct {
	Gamma []float64
	Beta  []float64
	Dim   int
	Eps   float64
}

// NewLayerNorm creates a layer norm with gamma=1, beta=0.
func NewLayerNorm(dim int) *LayerNorm {
	gamma := make([]float64, dim)
	for i := range gamma {
		gamma[i] = 1.0
	}
	return &LayerNorm{
		Gamma: gamma,
		Beta:  make([]float64, dim),
		Dim:   dim,
		Eps:   1e-6,
	}
}

// Forward applies layer norm over the last dimension.
// x shape: [..., dim], returns same shape
func (ln *LayerNorm) Forward(x []float64, shape []int) []float64 {
	result := make([]float64, len(x))
	total := 1
	for i := 0; i < len(shape)-1; i++ {
		total *= shape[i]
	}
	feat := shape[len(shape)-1]

	for n := 0; n < total; n++ {
		offset := n * feat

		mean := 0.0
		for d := 0; d < feat; d++ {
			mean += x[offset+d]
		}
		mean /= float64(feat)

		variance := 0.0
		for d := 0; d < feat; d++ {
			diff := x[offset+d] - mean
			variance += diff * diff
		}
		variance /= float64(feat)
		std := math.Sqrt(variance + ln.Eps)

		for d := 0; d < feat; d++ {
			normalized := (x[offset+d] - mean) / std
			result[offset+d] = ln.Gamma[d]*normalized + ln.Beta[d]
		}
	}
	return result
}

// GELU applies the Gaussian Error Linear Unit activation.
func GELU(x float64) float64 {
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}

// Dropout applies dropout during training.
type Dropout struct {
	P    float64
	Mask []float64
}

func NewDropout(p float64) *Dropout {
	return &Dropout{P: p}
}

func (d *Dropout) Forward(x []float64, train bool) []float64 {
	if !train || d.P == 0 {
		return x
	}
	result := make([]float64, len(x))
	scale := 1.0 / (1.0 - d.P)
	for i := range x {
		if rand.Float64() > d.P {
			result[i] = x[i] * scale
		}
	}
	return result
}

// Sequential chains layers together.
type Sequential struct {
	Layers []interface {
		Forward(x []float64) []float64
	}
}

func (s *Sequential) Add(layer interface{ Forward(x []float64) []float64 }) {
	s.Layers = append(s.Layers, layer)
}

func (s *Sequential) Forward(x []float64) []float64 {
	for _, layer := range s.Layers {
		x = layer.Forward(x)
	}
	return x
}
