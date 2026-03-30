package asr

import (
	"math"
	"math/rand"
)

// RNNLayer implements a simple RNN layer
type RNNLayer struct {
	InputSize  int
	HiddenSize int
	Weights    [][]float64
	Hidden     []float64
}

// NewRNNLayer creates a new RNN layer
func NewRNNLayer(inputSize, hiddenSize int) *RNNLayer {
	layer := &RNNLayer{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Weights:    make([][]float64, hiddenSize),
		Hidden:     make([]float64, hiddenSize),
	}

	// Initialize weights with Xavier initialization
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	for i := 0; i < hiddenSize; i++ {
		layer.Weights[i] = make([]float64, inputSize+hiddenSize+1) // +1 for bias
		for j := range layer.Weights[i] {
			layer.Weights[i][j] = rand.NormFloat64() * scale
		}
	}

	return layer
}

// Forward performs forward pass
func (l *RNNLayer) Forward(input []float64) []float64 {
	// Concatenate input and previous hidden state
	concat := make([]float64, l.InputSize+l.HiddenSize)
	copy(concat, input)
	copy(concat[l.InputSize:], l.Hidden)

	// Compute new hidden state
	newHidden := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		sum := l.Weights[i][len(l.Weights[i])-1] // bias
		for j := 0; j < l.InputSize+l.HiddenSize; j++ {
			sum += l.Weights[i][j] * concat[j]
		}
		newHidden[i] = math.Tanh(sum)
	}

	l.Hidden = newHidden
	return newHidden
}

// Reset clears hidden state
func (l *RNNLayer) Reset() {
	for i := range l.Hidden {
		l.Hidden[i] = 0
	}
}

// LSTMLayer implements a simple LSTM layer
type LSTMLayer struct {
	InputSize      int
	HiddenSize     int
	Wf, Wi, Wo, Wg [][]float64 // Weight matrices
	Hidden         []float64
	Cell           []float64
}

// NewLSTMLayer creates a new LSTM layer
func NewLSTMLayer(inputSize, hiddenSize int) *LSTMLayer {
	layer := &LSTMLayer{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Hidden:     make([]float64, hiddenSize),
		Cell:       make([]float64, hiddenSize),
	}

	// Initialize weight matrices
	// Each matrix: [input + hidden + bias]
	dim := inputSize + hiddenSize + 1
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))

	for i := 0; i < hiddenSize; i++ {
		layer.Wf = append(layer.Wf, make([]float64, dim))
		layer.Wi = append(layer.Wi, make([]float64, dim))
		layer.Wo = append(layer.Wo, make([]float64, dim))
		layer.Wg = append(layer.Wg, make([]float64, dim))

		for j := 0; j < dim; j++ {
			layer.Wf[i][j] = rand.NormFloat64() * scale
			layer.Wi[i][j] = rand.NormFloat64() * scale
			layer.Wo[i][j] = rand.NormFloat64() * scale
			layer.Wg[i][j] = rand.NormFloat64() * scale
		}
	}

	return layer
}

// Forward performs LSTM forward pass
func (l *LSTMLayer) Forward(input []float64) []float64 {
	// Concatenate input and hidden state
	concat := make([]float64, l.InputSize+l.HiddenSize)
	copy(concat, input)
	copy(concat[l.InputSize:], l.Hidden)

	// Compute gate activations
	fForget := make([]float64, l.HiddenSize)
	fInput := make([]float64, l.HiddenSize)
	fOutput := make([]float64, l.HiddenSize)
	fGate := make([]float64, l.HiddenSize)

	for i := 0; i < l.HiddenSize; i++ {
		sumF := l.Wf[i][len(l.Wf[i])-1]
		sumI := l.Wi[i][len(l.Wi[i])-1]
		sumO := l.Wo[i][len(l.Wo[i])-1]
		sumG := l.Wg[i][len(l.Wg[i])-1]

		for j := 0; j < l.InputSize+l.HiddenSize; j++ {
			val := concat[j]
			sumF += l.Wf[i][j] * val
			sumI += l.Wi[i][j] * val
			sumO += l.Wo[i][j] * val
			sumG += l.Wg[i][j] * val
		}

		fForget[i] = 1 / (1 + math.Exp(-sumF)) // sigmoid
		fInput[i] = 1 / (1 + math.Exp(-sumI))
		fOutput[i] = 1 / (1 + math.Exp(-sumO))
		fGate[i] = math.Tanh(sumG)
	}

	// Update cell state
	newCell := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newCell[i] = fForget[i]*l.Cell[i] + fInput[i]*fGate[i]
	}

	// Update hidden state
	newHidden := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newHidden[i] = fOutput[i] * math.Tanh(newCell[i])
	}

	l.Cell = newCell
	l.Hidden = newHidden

	return newHidden
}

// Reset clears LSTM state
func (l *LSTMLayer) Reset() {
	for i := range l.Hidden {
		l.Hidden[i] = 0
		l.Cell[i] = 0
	}
}

// LSTMForwardCache stores intermediate values for backward pass
type LSTMForwardCache struct {
	Concat     []float64
	ForgetGate []float64
	InputGate  []float64
	OutputGate []float64
	Gate       []float64
	CellState  []float64
	Hidden     []float64
}

// Backward performs backward pass through LSTM
func (l *LSTMLayer) Backward(input, gradOutput []float64, cache *LSTMForwardCache) (gradInput []float64) {
	hiddenSize := l.HiddenSize
	inputSize := l.InputSize

	gradInput = make([]float64, inputSize)

	dCellNext := make([]float64, hiddenSize)

	for i := 0; i < hiddenSize; i++ {
		dCt := dCellNext[i] + gradOutput[i]*math.Tanh(cache.CellState[i])

		dOt := dCt * cache.OutputGate[i] * (1 - cache.CellState[i]*cache.CellState[i])

		dGt := dCt * cache.InputGate[i] * (1 - cache.Gate[i]*cache.Gate[i])

		dIt := dCt * cache.Gate[i] * cache.InputGate[i] * (1 - cache.InputGate[i])

		dFt := dCt * cache.ForgetGate[i] * dCellNext[i] * cache.ForgetGate[i] * (1 - cache.ForgetGate[i])

		dim := inputSize + hiddenSize
		for j := 0; j < dim; j++ {
			val := cache.Concat[j]
			l.Wf[i][j] -= 0.001 * dFt * val
			l.Wi[i][j] -= 0.001 * dIt * val
			l.Wo[i][j] -= 0.001 * dOt * val
			l.Wg[i][j] -= 0.001 * dGt * val

			if j < inputSize {
				gradInput[j] += dFt*l.Wf[i][j] + dIt*l.Wi[i][j] + dOt*l.Wo[i][j] + dGt*l.Wg[i][j]
			}
		}

		l.Wf[i][dim] -= 0.001 * dFt
		l.Wi[i][dim] -= 0.001 * dIt
		l.Wo[i][dim] -= 0.001 * dOt
		l.Wg[i][dim] -= 0.001 * dGt
	}

	return gradInput
}

// ForwardWithCache performs forward pass and returns cache for backward
func (l *LSTMLayer) ForwardWithCache(input []float64) ([]float64, *LSTMForwardCache) {
	concat := make([]float64, l.InputSize+l.HiddenSize)
	copy(concat, input)
	copy(concat[l.InputSize:], l.Hidden)

	fForget := make([]float64, l.HiddenSize)
	fInput := make([]float64, l.HiddenSize)
	fOutput := make([]float64, l.HiddenSize)
	fGate := make([]float64, l.HiddenSize)

	for i := 0; i < l.HiddenSize; i++ {
		sumF := l.Wf[i][len(l.Wf[i])-1]
		sumI := l.Wi[i][len(l.Wi[i])-1]
		sumO := l.Wo[i][len(l.Wo[i])-1]
		sumG := l.Wg[i][len(l.Wg[i])-1]

		for j := 0; j < l.InputSize+l.HiddenSize; j++ {
			val := concat[j]
			sumF += l.Wf[i][j] * val
			sumI += l.Wi[i][j] * val
			sumO += l.Wo[i][j] * val
			sumG += l.Wg[i][j] * val
		}

		fForget[i] = 1 / (1 + math.Exp(-sumF))
		fInput[i] = 1 / (1 + math.Exp(-sumI))
		fOutput[i] = 1 / (1 + math.Exp(-sumO))
		fGate[i] = math.Tanh(sumG)
	}

	newCell := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newCell[i] = fForget[i]*l.Cell[i] + fInput[i]*fGate[i]
	}

	newHidden := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newHidden[i] = fOutput[i] * math.Tanh(newCell[i])
	}

	l.Cell = newCell
	l.Hidden = newHidden

	cache := &LSTMForwardCache{
		Concat:     concat,
		ForgetGate: fForget,
		InputGate:  fInput,
		OutputGate: fOutput,
		Gate:       fGate,
		CellState:  newCell,
		Hidden:     newHidden,
	}

	return newHidden, cache
}

// DenseLayer implements a fully connected layer
type DenseLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64
	Bias       []float64
}

// NewDenseLayer creates a new dense layer
func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	layer := &DenseLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    make([][]float64, outputSize),
		Bias:       make([]float64, outputSize),
	}

	scale := math.Sqrt(2.0 / float64(inputSize))
	for i := 0; i < outputSize; i++ {
		layer.Weights[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			layer.Weights[i][j] = rand.NormFloat64() * scale
		}
		layer.Bias[i] = 0
	}

	return layer
}

// Forward performs forward pass
func (l *DenseLayer) Forward(input []float64) []float64 {
	output := make([]float64, l.OutputSize)
	for i := 0; i < l.OutputSize; i++ {
		sum := l.Bias[i]
		for j := 0; j < l.InputSize; j++ {
			sum += l.Weights[i][j] * input[j]
		}
		output[i] = sum
	}
	return output
}

// Backward performs backward pass through dense layer
func (l *DenseLayer) Backward(input, gradOutput []float64, lr float64) []float64 {
	gradInput := make([]float64, l.InputSize)

	for i := 0; i < l.OutputSize; i++ {
		for j := 0; j < l.InputSize; j++ {
			l.Weights[i][j] -= lr * gradOutput[i] * input[j]
			gradInput[j] += gradOutput[i] * l.Weights[i][j]
		}
		l.Bias[i] -= lr * gradOutput[i]
	}

	return gradInput
}

// Softmax applies softmax to vector
func Softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}

// LogSoftmax applies log softmax
func LogSoftmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for _, v := range x {
		sum += math.Exp(v - maxVal)
	}

	logSum := maxVal + math.Log(sum)
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = v - logSum
	}

	return result
}

// AcousticModel represents the acoustic model
type AcousticModel struct {
	InputSize   int
	HiddenSize  int
	OutputSize  int
	RNN         *LSTMLayer
	OutputLayer *DenseLayer
}

// NewAcousticModel creates a new acoustic model
func NewAcousticModel(inputSize, hiddenSize, outputSize int) *AcousticModel {
	return &AcousticModel{
		InputSize:   inputSize,
		HiddenSize:  hiddenSize,
		OutputSize:  outputSize,
		RNN:         NewLSTMLayer(inputSize, hiddenSize),
		OutputLayer: NewDenseLayer(hiddenSize, outputSize),
	}
}

// Forward performs forward pass through the model
func (m *AcousticModel) Forward(features [][]float64) [][]float64 {
	numFrames := len(features)
	probabilities := make([][]float64, numFrames)

	m.RNN.Reset()
	for t := 0; t < numFrames; t++ {
		hidden := m.RNN.Forward(features[t])
		output := m.OutputLayer.Forward(hidden)
		probabilities[t] = Softmax(output)
	}

	return probabilities
}

// Predict performs prediction on features
func (m *AcousticModel) Predict(features [][]float64) []int {
	probs := m.Forward(features)
	predictions := make([]int, len(probs))

	for t, prob := range probs {
		maxIdx := 0
		maxVal := prob[0]
		for i, v := range prob {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		predictions[t] = maxIdx
	}

	return predictions
}

// CTCDecode performs CTC (Connectionist Temporal Classification) decoding
func CTCDecode(predictions []int, blankIdx int) []int {
	var result []int
	var prev int = -1

	for _, curr := range predictions {
		if curr != blankIdx && curr != prev {
			result = append(result, curr)
		}
		prev = curr
	}

	return result
}

// GreedyDecode performs greedy decoding with optional CTC
func GreedyDecode(probs [][]float64, blankIdx int) []int {
	predictions := make([]int, len(probs))
	for t, prob := range probs {
		maxIdx := 0
		maxVal := prob[0]
		for i, v := range prob {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		predictions[t] = maxIdx
	}
	return CTCDecode(predictions, blankIdx)
}

// BeamSearchDecode performs beam search decoding
func BeamSearchDecode(probs [][]float64, blankIdx int, beamWidth int, vocabulary []string) string {
	type Beam struct {
		Path  []int
		Score float64
	}

	beams := []Beam{{Path: []int{}, Score: 0.0}}

	for t := 0; t < len(probs); t++ {
		newBeams := []Beam{}
		for _, beam := range beams {
			for i, p := range probs[t] {
				if p < 1e-10 {
					continue
				}

				newPath := make([]int, len(beam.Path))
				copy(newPath, beam.Path)

				if i != blankIdx {
					if len(newPath) == 0 || newPath[len(newPath)-1] != i {
						newPath = append(newPath, i)
					}
				}

				newBeam := Beam{
					Path:  newPath,
					Score: beam.Score + math.Log(p),
				}
				newBeams = append(newBeams, newBeam)
			}
		}

		// Sort by score and keep top beams
		if len(newBeams) > beamWidth {
			newBeams = newBeams[:beamWidth]
		}
		beams = newBeams
	}

	// Convert best beam path to text
	if len(beams) == 0 {
		return ""
	}

	bestPath := beams[0].Path
	result := make([]rune, len(bestPath))
	for i, idx := range bestPath {
		if idx < len(vocabulary) {
			result[i] = rune(vocabulary[idx][0])
		}
	}

	return string(result)
}

// ============================================================
// CTC Loss (Connectionist Temporal Classification)
// ============================================================

// CTCLoss computes CTC loss
func CTCLoss(logits [][]float64, targets []int, blankIdx int) (float64, [][]float64) {
	T := len(logits)
	B := 1

	alpha := make([][]float64, T)
	for t := 0; t < T; t++ {
		alpha[t] = make([]float64, len(targets)*2+1)
		for i := range alpha[t] {
			alpha[t][i] = -1e9
		}
	}

	alpha[0][0] = logits[0][blankIdx]
	alpha[0][1] = logits[0][targets[0]]

	for t := 1; t < T; t++ {
		for s := 0; s < len(targets)*2+1; s++ {
			maxAlpha := -1e9

			if s%2 == 0 {
				for prevS := s; prevS >= max(0, s-2); prevS-- {
					alphaVal := alpha[t-1][prevS]
					if alphaVal > maxAlpha {
						maxAlpha = alphaVal
					}
				}
			} else {
				for prevS := max(0, s-2); prevS <= min(s, len(targets)*2); prevS++ {
					alphaVal := alpha[t-1][prevS]
					if alphaVal > maxAlpha {
						maxAlpha = alphaVal
					}
				}
			}

			idx := s / 2
			if s%2 == 1 && idx < len(targets) && (idx == 0 || targets[idx] != targets[idx-1]) {
				maxAlpha = math.Max(maxAlpha, alpha[t-1][s-1])
			}

			if maxAlpha > -1e9 {
				targetIdx := s / 2
				if s%2 == 1 && targetIdx < len(targets) {
					alpha[t][s] = maxAlpha + logits[t][targets[targetIdx]]
				} else {
					alpha[t][s] = maxAlpha + logits[t][blankIdx]
				}
			}
		}
	}

	loss := alpha[T-1][len(targets)*2] + alpha[T-1][len(targets)*2-1]

	gradLogits := make([][]float64, T)
	for t := 0; t < T; t++ {
		gradLogits[t] = make([]float64, len(logits[t]))
		for i := range gradLogits[t] {
			gradLogits[t][i] = -math.Exp(logits[t][i])
		}
	}

	return -loss / float64(B), gradLogits
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================
// Streaming CTC Decoder
// ============================================================

// StreamingDecoder performs chunked CTC decoding
type StreamingDecoder struct {
	blankIdx      int
	minDuration   int
	mergeTokens   bool
	chunkBuffer   [][]float64
	holdFrames    int
	lastDecoded   []int
	silenceFrames int
}

// NewStreamingDecoder creates a new streaming decoder
func NewStreamingDecoder(blankIdx int, minDuration int) *StreamingDecoder {
	return &StreamingDecoder{
		blankIdx:      blankIdx,
		minDuration:   minDuration,
		mergeTokens:   true,
		chunkBuffer:   make([][]float64, 0),
		holdFrames:    0,
		lastDecoded:   make([]int, 0),
		silenceFrames: 0,
	}
}

// ProcessChunk processes a chunk of frame probabilities
func (d *StreamingDecoder) ProcessChunk(chunk [][]float64) []int {
	d.chunkBuffer = append(d.chunkBuffer, chunk...)
	return d.decodeBuffer()
}

// decodeBuffer decodes buffered frames
func (d *StreamingDecoder) decodeBuffer() []int {
	if len(d.chunkBuffer) < d.minDuration {
		return nil
	}

	frame := d.chunkBuffer[0]
	d.chunkBuffer = d.chunkBuffer[1:]

	maxIdx := 0
	maxVal := frame[0]
	for i, v := range frame {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}

	result := make([]int, 0)
	if maxIdx != d.blankIdx {
		if len(d.lastDecoded) == 0 || d.lastDecoded[len(d.lastDecoded)-1] != maxIdx {
			result = append(result, maxIdx)
		}
		d.lastDecoded = append(d.lastDecoded, maxIdx)
		d.silenceFrames = 0
	} else {
		d.silenceFrames++
	}

	return result
}

// Flush flushes remaining buffer and returns final decoded
func (d *StreamingDecoder) Flush() []int {
	d.chunkBuffer = make([][]float64, 0)
	d.silenceFrames = 0
	result := d.lastDecoded
	d.lastDecoded = make([]int, 0)
	return result
}

// Reset resets the decoder state
func (d *StreamingDecoder) Reset() {
	d.chunkBuffer = make([][]float64, 0)
	d.lastDecoded = make([]int, 0)
	d.silenceFrames = 0
	d.holdFrames = 0
}
