package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// Tensor is a simple float64 matrix representation
type Tensor struct {
	data   []float64
	shape  [2]int
	grad   []float64
}

// NewTensor creates a new tensor with the given shape
func NewTensor(rows, cols int) *Tensor {
	return &Tensor{
		data:  make([]float64, rows*cols),
		shape: [2]int{rows, cols},
		grad:  make([]float64, rows*cols),
	}
}

// NewTensorWithData creates a tensor with existing data
func NewTensorWithData(data []float64, rows, cols int) *Tensor {
	return &Tensor{
		data:  data,
		shape: [2]int{rows, cols},
		grad:  make([]float64, len(data)),
	}
}

// Rand fills the tensor with random values
func (t *Tensor) Rand() {
	rand.Seed(time.Now().UnixNano())
	for i := range t.data {
		t.data[i] = rand.Float64()*2 - 1
	}
}

// ZeroGrad zeros the gradients
func (t *Tensor) ZeroGrad() {
	for i := range t.grad {
		t.grad[i] = 0
	}
}

// MatMulNaive performs naive triple-loop matrix multiplication
func MatMulNaive(a, b, c *Tensor) {
	rowsA, colsA := a.shape[0], a.shape[1]
	rowsB, colsB := b.shape[0], b.shape[1]
	_ = rowsB

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := 0.0
			for k := 0; k < colsA; k++ {
				sum += a.data[i*colsA+k] * b.data[k*colsB+j]
			}
			c.data[i*colsB+j] = sum
		}
	}
}

// MatMulBlocked performs blocked matrix multiplication for cache efficiency
func MatMulBlocked(a, b, c *Tensor, blockSize int) {
	rowsA, colsA := a.shape[0], a.shape[1]
	rowsB, colsB := b.shape[0], b.shape[1]
	_ = rowsB

	for i := 0; i < rowsA; i += blockSize {
		for j := 0; j < colsB; j += blockSize {
			for k := 0; k < colsA; k += blockSize {
				iEnd := minInt(rowsA, i+blockSize)
				jEnd := minInt(colsB, j+blockSize)
				kEnd := minInt(colsA, k+blockSize)

				for ii := i; ii < iEnd; ii++ {
					for jj := j; jj < jEnd; jj++ {
						sum := c.data[ii*colsB+jj]
						for kk := k; kk < kEnd; kk++ {
							sum += a.data[ii*colsA+kk] * b.data[kk*colsB+jj]
						}
						c.data[ii*colsB+jj] = sum
					}
				}
			}
		}
	}
}

// Softmax computes softmax over the last dimension
func Softmax(input, output *Tensor) {
	rows := input.shape[0]
	cols := input.shape[1]

	for i := 0; i < rows; i++ {
		offset := i * cols
		maxVal := input.data[offset]
		for j := 1; j < cols; j++ {
			if input.data[offset+j] > maxVal {
				maxVal = input.data[offset+j]
			}
		}

		sum := 0.0
		for j := 0; j < cols; j++ {
			output.data[offset+j] = math.Exp(input.data[offset+j] - maxVal)
			sum += output.data[offset+j]
		}

		for j := 0; j < cols; j++ {
			output.data[offset+j] /= sum
		}
	}
}

// LayerNorm performs layer normalization
func LayerNorm(input, output, gamma, beta, mean, varTensor *Tensor, eps float64) {
	rows := input.shape[0]
	cols := input.shape[1]

	for i := 0; i < rows; i++ {
		offset := i * cols

		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += input.data[offset+j]
		}
		m := sum / float64(cols)
		if mean != nil && i < len(mean.data) {
			mean.data[i] = m
		}

		varSum := 0.0
		for j := 0; j < cols; j++ {
			diff := input.data[offset+j] - m
			varSum += diff * diff
		}
		v := varSum / float64(cols)
		if varTensor != nil && i < len(varTensor.data) {
			varTensor.data[i] = v
		}

		invStd := 1.0 / math.Sqrt(v+eps)
		for j := 0; j < cols; j++ {
			normalized := (input.data[offset+j] - m) * invStd
			output.data[offset+j] = normalized*gamma.data[j] + beta.data[j]
		}
	}
}

// LinearLayer represents a linear layer
type LinearLayer struct {
	weight *Tensor
	bias   *Tensor
}

func NewLinearLayer(inputDim, outputDim int) *LinearLayer {
	l := &LinearLayer{
		weight: NewTensor(inputDim, outputDim),
		bias:   NewTensor(1, outputDim),
	}
	l.weight.Rand()
	l.bias.Rand()
	return l
}

func (l *LinearLayer) Forward(input *Tensor) *Tensor {
	output := NewTensor(input.shape[0], l.weight.shape[1])
	rows := input.shape[0]
	colsIn := input.shape[1]
	colsOut := l.weight.shape[1]

	for i := 0; i < rows; i++ {
		for j := 0; j < colsOut; j++ {
			sum := l.bias.data[j]
			for k := 0; k < colsIn; k++ {
				sum += input.data[i*colsIn+k] * l.weight.data[k*colsOut+j]
			}
			output.data[i*colsOut+j] = sum
		}
	}
	return output
}

func (l *LinearLayer) Backward(input, gradOutput *Tensor) {
	rows := input.shape[0]
	colsIn := l.weight.shape[0]
	colsOut := l.weight.shape[1]

	// Weight gradient
	for i := 0; i < colsIn && i*colsOut < len(l.weight.grad); i++ {
		for j := 0; j < colsOut && i*colsOut+j < len(l.weight.grad); j++ {
			sum := 0.0
			for k := 0; k < rows; k++ {
				idx := k*colsIn+i
				outIdx := k*colsOut+j
				if idx < len(input.data) && outIdx < len(gradOutput.data) {
					sum += input.data[idx] * gradOutput.data[outIdx]
				}
			}
			gradIdx := i*colsOut+j
			if gradIdx < len(l.weight.grad) {
				l.weight.grad[gradIdx] += sum / float64(rows)
			}
		}
	}

	// Bias gradient
	for j := 0; j < colsOut; j++ {
		sum := 0.0
		for k := 0; k < rows; k++ {
			outIdx := k*colsOut+j
			if outIdx < len(gradOutput.data) {
				sum += gradOutput.data[outIdx]
			}
		}
		if j < len(l.bias.grad) {
			l.bias.grad[j] += sum / float64(rows)
		}
	}
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// MLP represents a simple multi-layer perceptron
type MLP struct {
	layers     []*LinearLayer
	activations []func(*Tensor)
}

func NewMLP(inputDim, hiddenDim, outputDim, numLayers int) *MLP {
	mlp := &MLP{
		layers:     make([]*LinearLayer, numLayers),
		activations: make([]func(*Tensor), numLayers-1),
	}

	prevDim := inputDim
	for i := 0; i < numLayers; i++ {
		if i == numLayers-1 {
			mlp.layers[i] = NewLinearLayer(prevDim, outputDim)
		} else {
			mlp.layers[i] = NewLinearLayer(prevDim, hiddenDim)
			mlp.activations[i] = func(t *Tensor) {
				for idx := range t.data {
					if t.data[idx] < 0 {
						t.data[idx] = 0
					}
				}
			}
		}
		prevDim = hiddenDim
	}

	return mlp
}

func (m *MLP) Forward(input *Tensor) *Tensor {
	output := input
	for i, layer := range m.layers {
		output = layer.Forward(output)
		if i < len(m.activations) && m.activations[i] != nil {
			m.activations[i](output)
		}
	}
	return output
}

func (m *MLP) Backward(input, gradOutput *Tensor) {
	grad := make([]float64, len(gradOutput.data))
	copy(grad, gradOutput.data)
	gradTensor := NewTensorWithData(grad, gradOutput.shape[0], gradOutput.shape[1])

	for i := len(m.layers) - 1; i >= 0; i-- {
		m.layers[i].Backward(input, gradTensor)
	}
}

func (m *MLP) ZeroGrad() {
	for _, layer := range m.layers {
		layer.weight.ZeroGrad()
		layer.bias.ZeroGrad()
	}
}

// BenchmarkMatMul512 benchmarks 512x512 matrix multiplication
func BenchmarkMatMul512(b *testing.B) {
	a := NewTensor(512, 512)
	bMat := NewTensor(512, 512)
	c := NewTensor(512, 512)
	a.Rand()
	bMat.Rand()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulNaive(a, bMat, c)
	}
	b.ReportAllocs()
}

// BenchmarkMatMulSizes benchmarks matrix multiplication at different sizes
func BenchmarkMatMulSizes(b *testing.B) {
	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			a := NewTensor(size, size)
			bMat := NewTensor(size, size)
			c := NewTensor(size, size)
			a.Rand()
			bMat.Rand()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulNaive(a, bMat, c)
			}
		})
	}
}

// BenchmarkMatMulBlocked benchmarks blocked matrix multiplication
func BenchmarkMatMulBlocked(b *testing.B) {
	sizes := []int{256, 512}
	blockSizes := []int{32, 64}

	for _, size := range sizes {
		for _, bs := range blockSizes {
			b.Run(fmt.Sprintf("size_%d_block_%d", size, bs), func(b *testing.B) {
				a := NewTensor(size, size)
				bMat := NewTensor(size, size)
				c := NewTensor(size, size)
				a.Rand()
				bMat.Rand()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatMulBlocked(a, bMat, c, bs)
				}
			})
		}
	}
}

// BenchmarkSoftmax benchmarks softmax operation
func BenchmarkSoftmax(b *testing.B) {
	input := NewTensor(1000, 1000)
	input.Rand()
	output := NewTensor(1000, 1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(input, output)
	}
	b.ReportAllocs()
}

// BenchmarkLayerNorm benchmarks layer normalization
func BenchmarkLayerNorm(b *testing.B) {
	input := NewTensor(256, 512)
	input.Rand()
	output := NewTensor(256, 512)
	gamma := NewTensor(1, 512)
	beta := NewTensor(1, 512)
	gamma.Rand()
	beta.Rand()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LayerNorm(input, output, gamma, beta, nil, nil, 1e-5)
	}
	b.ReportAllocs()
}

// BenchmarkBackward benchmarks backward pass on MLP
func BenchmarkBackward(b *testing.B) {
	mlp := NewMLP(128, 128, 10, 3)
	input := NewTensor(32, 128)
	input.Rand()

	output := mlp.Forward(input)

	gradOutput := NewTensor(output.shape[0], output.shape[1])
	gradOutput.Rand()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mlp.ZeroGrad()
		mlp.Backward(input, gradOutput)
	}
}

// BenchmarkMLPEndToEnd benchmarks full MLP forward + backward
func BenchmarkMLPEndToEnd(b *testing.B) {
	mlp := NewMLP(128, 128, 10, 3)
	input := NewTensor(32, 128)
	input.Rand()
	gradOutput := NewTensor(32, 10)
	gradOutput.Rand()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mlp.ZeroGrad()
		mlp.Forward(input)
		mlp.Backward(input, gradOutput)
	}
}

// MemoryUsage estimates current memory usage
func MemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

// runWithTiming runs fn and returns timing statistics
func runWithTiming(fn func(), iterations int) (float64, float64, float64, float64, float64) {
	fn()

	var memBefore uint64
	var memAfter uint64

	times := make([]time.Duration, iterations)
	for i := 0; i < iterations; i++ {
		runtime.GC()
		memBefore = MemoryUsage()
		t0 := time.Now()
		fn()
		t1 := time.Now()
		memAfter = MemoryUsage()
		times[i] = t1.Sub(t0)
	}

	var total time.Duration
	for _, t := range times {
		total += t
	}
	avg := total.Seconds() / float64(iterations)

	sorted := make([]time.Duration, len(times))
	copy(sorted, times)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	p50 := sorted[int(float64(len(sorted))*0.5)].Seconds()
	p95 := sorted[int(float64(len(sorted))*0.95)].Seconds()
	p99 := sorted[int(float64(len(sorted))*0.99)].Seconds()

	memUsed := float64(memAfter - memBefore) / (1024 * 1024)

	return avg, p50, p95, p99, memUsed
}

// TensorBenchmarkResults holds benchmark results for tensor operations
type TensorBenchmarkResults struct {
	Operation     string
	Size          string
	OpsPerSec     float64
	AvgLatencyMs  float64
	P50LatencyMs  float64
	P95LatencyMs  float64
	P99LatencyMs  float64
	MemoryMB      float64
}

func RunTensorBenchmarks() []TensorBenchmarkResults {
	results := []TensorBenchmarkResults{}

	sizes := []int{128, 256, 512}
	for _, size := range sizes {
		a := NewTensor(size, size)
		bMat := NewTensor(size, size)
		c := NewTensor(size, size)
		a.Rand()
		bMat.Rand()

		fn := func() { MatMulNaive(a, bMat, c) }
		avg, p50, p95, p99, mem := runWithTiming(fn, 10)

		opsPerSec := 1.0 / avg
		results = append(results, TensorBenchmarkResults{
			Operation:    "MatMul",
			Size:         fmt.Sprintf("%dx%d", size, size),
			OpsPerSec:    opsPerSec,
			AvgLatencyMs: avg * 1000,
			P50LatencyMs: p50 * 1000,
			P95LatencyMs: p95 * 1000,
			P99LatencyMs: p99 * 1000,
			MemoryMB:     mem,
		})
	}

	input := NewTensor(1000, 1000)
	input.Rand()
	output := NewTensor(1000, 1000)
	fn := func() { Softmax(input, output) }
	avg, p50, p95, p99, mem := runWithTiming(fn, 10)
	results = append(results, TensorBenchmarkResults{
		Operation:    "Softmax",
		Size:         "1000x1000",
		OpsPerSec:    1.0 / avg,
		AvgLatencyMs: avg * 1000,
		P50LatencyMs: p50 * 1000,
		P95LatencyMs: p95 * 1000,
		P99LatencyMs: p99 * 1000,
		MemoryMB:     mem,
	})

	lnInput := NewTensor(256, 512)
	lnInput.Rand()
	lnOutput := NewTensor(256, 512)
	gamma := NewTensor(1, 512)
	beta := NewTensor(1, 512)
	gamma.Rand()
	beta.Rand()
	fn = func() { LayerNorm(lnInput, lnOutput, gamma, beta, nil, nil, 1e-5) }
	avg, p50, p95, p99, mem = runWithTiming(fn, 10)
	results = append(results, TensorBenchmarkResults{
		Operation:    "LayerNorm",
		Size:         "256x512",
		OpsPerSec:    1.0 / avg,
		AvgLatencyMs: avg * 1000,
		P50LatencyMs: p50 * 1000,
		P95LatencyMs: p95 * 1000,
		P99LatencyMs: p99 * 1000,
		MemoryMB:     mem,
	})

	mlp := NewMLP(128, 128, 10, 3)
	mlpInput := NewTensor(32, 128)
	mlpInput.Rand()
	gradOutput := NewTensor(32, 10)
	gradOutput.Rand()

	fn = func() {
		mlp.ZeroGrad()
		mlp.Backward(mlpInput, gradOutput)
	}
	avg, p50, p95, p99, mem = runWithTiming(fn, 10)
	results = append(results, TensorBenchmarkResults{
		Operation:    "MLP Backward",
		Size:         "3 layers, 128 hidden",
		OpsPerSec:    1.0 / avg,
		AvgLatencyMs: avg * 1000,
		P50LatencyMs: p50 * 1000,
		P95LatencyMs: p95 * 1000,
		P99LatencyMs: p99 * 1000,
		MemoryMB:     mem,
	})

	return results
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

var resultSink float64
