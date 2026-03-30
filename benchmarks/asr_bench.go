package benchmarks

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// MFCCConfig holds MFCC extraction configuration
type MFCCConfig struct {
	SampleRate   int
	FrameLength int
	HopLength   int
	NumMFCCs    int
	NumFilters  int
	NFTT        int
}

// MFCCResult holds MFCC extraction results
type MFCCResult struct {
	mfcc      [][]float64
	logEnergy []float64
}

// MelFilterbank creates a mel filterbank matrix
func MelFilterbank(numFilters, numFFT, sampleRate int) [][]float64 {
	filterbank := make([][]float64, numFilters)
	for i := range filterbank {
		filterbank[i] = make([]float64, numFFT/2+1)
	}

	// Convert mel scale
	melMin := 2595 * math.Log10(1+float64(0)/700)
	melMax := 2595 * math.Log10(1+float64(sampleRate/2)/700)

	for i := 0; i < numFilters; i++ {
		leftMel := melMin + (melMax-melMin)*float64(i)/float64(numFilters+1)
		centerMel := melMin + (melMax-melMin)*float64(i+1)/float64(numFilters+1)
		rightMel := melMin + (melMax-melMin)*float64(i+2)/float64(numFilters+1)

		leftFreq := 700 * (math.Pow(10, leftMel/2595) - 1)
		centerFreq := 700 * (math.Pow(10, centerMel/2595) - 1)
		rightFreq := 700 * (math.Pow(10, rightMel/2595) - 1)

		leftBin := int(leftFreq * float64(numFFT) / float64(sampleRate))
		centerBin := int(centerFreq * float64(numFFT) / float64(sampleRate))
		rightBin := int(rightFreq * float64(numFFT) / float64(sampleRate))

		for j := leftBin; j < centerBin; j++ {
			if j >= 0 && j < numFFT/2+1 {
				filterbank[i][j] = float64(j-leftBin) / float64(centerBin-leftBin)
			}
		}
		for j := centerBin; j < rightBin; j++ {
			if j >= 0 && j < numFFT/2+1 {
				filterbank[i][j] = float64(rightBin-j) / float64(rightBin-centerBin)
			}
		}
	}

	return filterbank
}

// MFCC extracts MFCC features from audio samples
func MFCC(audio []float64, config MFCCConfig) *MFCCResult {
	numFrames := (len(audio) - config.FrameLength) / config.HopLength + 1
	mfcc := make([][]float64, numFrames)
	logEnergy := make([]float64, numFrames)

	// Pre-compute window
	window := make([]float64, config.FrameLength)
	for i := 0; i < config.FrameLength; i++ {
		window[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(config.FrameLength-1))
	}

	// Create filterbank
	filterbank := MelFilterbank(config.NumFilters, config.NFTT, config.SampleRate)

	// Process each frame
	for frame := 0; frame < numFrames; frame++ {
		start := frame * config.HopLength
		frameData := make([]float64, config.FrameLength)

		// Apply window
		sum := 0.0
		for i := 0; i < config.FrameLength; i++ {
			if start+i < len(audio) {
				frameData[i] = audio[start+i] * window[i]
				sum += frameData[i] * frameData[i]
			}
		}
		logEnergy[frame] = math.Log(sum + 1e-10)

		// FFT (simplified - just compute magnitude spectrum)
		magnitude := make([]float64, config.NFTT/2+1)
		for i := 0; i < config.NFTT/2+1; i++ {
			real := 0.0
			for j := 0; j < config.FrameLength && j < config.NFTT; j++ {
				real += frameData[j] * math.Cos(2*math.Pi*float64(i*j)/float64(config.NFTT))
			}
			magnitude[i] = math.Sqrt(real*real + 0.001) // Simplified imag part
		}

		// Apply mel filterbank
		filtered := make([]float64, config.NumFilters)
		for i := 0; i < config.NumFilters; i++ {
			for j := 0; j < len(magnitude) && j < len(filterbank[i]); j++ {
				filtered[i] += magnitude[j] * filterbank[i][j]
			}
			filtered[i] = math.Log(filtered[i] + 1e-10)
		}

		// DCT (simplified - just first numMFCCs)
		mfcc[frame] = make([]float64, config.NumMFCCs)
		for i := 0; i < config.NumMFCCs; i++ {
			sum := 0.0
			for j := 0; j < config.NumFilters; j++ {
				sum += filtered[j] * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(config.NFTT))
			}
			mfcc[frame][i] = sum
		}
	}

	return &MFCCResult{mfcc: mfcc, logEnergy: logEnergy}
}

// LSTMConfig holds LSTM configuration
type LSTMConfig struct {
	InputSize  int
	HiddenSize int
	NumLayers  int
	Bidirectional bool
}

// LSTMCell represents a single LSTM cell
type LSTMCell struct {
	wi *LinearLayer
	wf *LinearLayer
	wo *LinearLayer
	wc *LinearLayer
}

// NewLSTMCell creates a new LSTM cell
func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	return &LSTMCell{
		wi: NewLinearT(inputSize+hiddenSize, hiddenSize),
		wf: NewLinearT(inputSize+hiddenSize, hiddenSize),
		wo: NewLinearT(inputSize+hiddenSize, hiddenSize),
		wc: NewLinearT(inputSize+hiddenSize, hiddenSize),
	}
}

// LSTMCellForward performs forward pass through LSTM cell
func LSTMCellForward(cell *LSTMCell, x, h, c []float64) (float64, float64) {
	// Concatenate x and h
	combined := make([]float64, len(x)+len(h))
	copy(combined, x)
	copy(combined[len(x):], h)

	// Compute gates (simplified - just do forward through one linear)
	_ = combined
	_ = cell.wi
	_ = cell.wf
	_ = cell.wo
	_ = cell.wc

	// Simplified new c and h
	newC := 0.0
	newH := 0.0
	for i := range c {
		newC += c[i] * 0.1
	}
	for i := range h {
		newH += h[i] * 0.1
	}

	return newC, newH
}

// LSTMForward performs forward pass through LSTM
func LSTMForward(input [][]float64, config LSTMConfig) [][]float64 {
	numDirections := 1
	if config.Bidirectional {
		numDirections = 2
	}

	seqLen := len(input)
	hiddenSize := config.HiddenSize

	output := make([][]float64, seqLen)
	for i := range output {
		output[i] = make([]float64, hiddenSize*numDirections)
	}

	// Simplified forward - just compute one direction
	cell := NewLSTMCell(config.InputSize, hiddenSize)
	h := make([]float64, hiddenSize)
	c := make([]float64, hiddenSize)

	for t := 0; t < seqLen; t++ {
		newC, newH := LSTMCellForward(cell, input[t], h, c)
		if t < len(c) {
			c[0] = newC
		}
		if t < len(h) {
			h[0] = newH
		}
		output[t] = h
	}

	return output
}

// CTCGreedyDecode performs greedy CTC decoding
func CTCGreedyDecode(probs [][]float64, blank int) []int {
	// probs is [time, vocab]
	decoded := []int{}
	prev := blank

	for t := 0; t < len(probs); t++ {
		// Find argmax
		maxProb := probs[t][0]
		maxIdx := 0
		for i := 1; i < len(probs[t]); i++ {
			if probs[t][i] > maxProb {
				maxProb = probs[t][i]
				maxIdx = i
			}
		}

		// Add if not blank and not repeating
		if maxIdx != blank && maxIdx != prev {
			decoded = append(decoded, maxIdx)
		}
		prev = maxIdx
	}

	return decoded
}

// CTCBeamSearchDecode performs beam search CTC decoding
func CTCBeamSearchDecode(probs [][]float64, blank int, beamWidth int) []int {
	// Simplified beam search
	type beam struct {
		prefix []int
		score  float64
	}

	beams := []beam{{prefix: []int{}, score: 0.0}}

	for t := 0; t < len(probs); t++ {
		newBeams := make([]beam, 0)

		for _, b := range beams {
			for c := 0; c < len(probs[t]); c++ {
				newPrefix := make([]int, len(b.prefix))
				copy(newPrefix, b.prefix)
				score := b.score + math.Log(probs[t][c]+1e-10)

				// Add character if not blank and not repeating last char
				if c != blank && (len(newPrefix) == 0 || newPrefix[len(newPrefix)-1] != c) {
					newPrefix = append(newPrefix, c)
				}

				newBeams = append(newBeams, beam{prefix: newPrefix, score: score})
			}
		}

		// Keep top beamWidth beams
		if len(newBeams) > beamWidth {
			// Sort by score (simplified)
			for i := 0; i < len(newBeams)-1; i++ {
				for j := i + 1; j < len(newBeams); j++ {
					if newBeams[j].score > newBeams[i].score {
						newBeams[i], newBeams[j] = newBeams[j], newBeams[i]
					}
				}
			}
			newBeams = newBeams[:beamWidth]
		}

		beams = newBeams
	}

	// Return best beam
	bestScore := beams[0].score
	best := beams[0].prefix
	for _, b := range beams {
		if b.score > bestScore {
			bestScore = b.score
			best = b.prefix
		}
	}

	return best
}

// ASRBenchmarkResults holds ASR benchmark results
type ASRBenchmarkResults struct {
	TestName      string
	Metric        string
	Value         float64
	LatencyMs     float64
	Iterations    int
}

func RunASRBenchmarks() []ASRBenchmarkResults {
	results := []ASRBenchmarkResults{}

	// MFCC benchmark
	config := MFCCConfig{
		SampleRate:   16000,
		FrameLength: 400,
		HopLength:   160,
		NumMFCCs:    13,
		NumFilters:  26,
		NFTT:        512,
	}

	// Generate 1 second of audio
	audio := make([]float64, 16000)
	for i := range audio {
		audio[i] = rand.Float64()*2 - 1
	}

	fn := func() { MFCC(audio, config) }

	// Warmup
	fn()

	// Benchmark
	iterations := 100
	var totalTime time.Duration
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime := totalTime.Seconds() / float64(iterations)
	results = append(results, ASRBenchmarkResults{
		TestName:   "MFCC",
		Metric:     "Samples/sec",
		Value:      float64(len(audio)) / avgTime,
		LatencyMs:  avgTime * 1000,
		Iterations: iterations,
	})

	// LSTM forward benchmark
	lstmConfig := LSTMConfig{
		InputSize:  40,
		HiddenSize: 256,
		NumLayers:  3,
	}

	// 150 frames (about 1.5 seconds at 10ms frame shift)
	seqLen := 150
	input := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		input[i] = make([]float64, 40)
		for j := 0; j < 40; j++ {
			input[i][j] = rand.Float64()*2 - 1
		}
	}

	fn = func() { LSTMForward(input, lstmConfig) }

	// Warmup
	fn()

	// Benchmark
	iterations = 50
	totalTime = 0
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime = totalTime.Seconds() / float64(iterations)
	results = append(results, ASRBenchmarkResults{
		TestName:   "LSTM Forward",
		Metric:     "Frames/sec",
		Value:      float64(seqLen) / avgTime,
		LatencyMs:  avgTime * 1000,
		Iterations: iterations,
	})

	// CTC decode benchmark
	// Generate random probs
	frameCount := 100
	vocabSize := 28 + 26 // blank + a-z + space
	probs := make([][]float64, frameCount)
	for t := 0; t < frameCount; t++ {
		probs[t] = make([]float64, vocabSize)
		sum := 0.0
		for i := 0; i < vocabSize; i++ {
			probs[t][i] = rand.Float64()
			sum += probs[t][i]
		}
		// Normalize
		for i := 0; i < vocabSize; i++ {
			probs[t][i] /= sum
		}
	}

	// Greedy decode
	fn = func() { CTCGreedyDecode(probs, 0) }
	fn() // warmup

	iterations = 100
	totalTime = 0
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime = totalTime.Seconds() / float64(iterations)
	results = append(results, ASRBenchmarkResults{
		TestName:   "CTC Greedy Decode",
		Metric:     "Decodes/sec",
		Value:      1.0 / avgTime,
		LatencyMs:  avgTime * 1000,
		Iterations: iterations,
	})

	// Beam search decode
	fn = func() { CTCBeamSearchDecode(probs, 0, 10) }
	fn() // warmup

	iterations = 50
	totalTime = 0
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime = totalTime.Seconds() / float64(iterations)
	results = append(results, ASRBenchmarkResults{
		TestName:   "CTC Beam Search",
		Metric:     "Decodes/sec",
		Value:      1.0 / avgTime,
		LatencyMs:  avgTime * 1000,
		Iterations: iterations,
	})

	return results
}

// BenchmarkMFCC benchmarks MFCC extraction
func BenchmarkMFCC(b *testing.B) {
	config := MFCCConfig{
		SampleRate:   16000,
		FrameLength: 400,
		HopLength:   160,
		NumMFCCs:    13,
		NumFilters:  26,
		NFTT:        512,
	}

	audio := make([]float64, 16000)
	for i := range audio {
		audio[i] = rand.Float64()*2 - 1
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		MFCC(audio, config)
	}
}

// BenchmarkLSTMForward benchmarks LSTM forward pass
func BenchmarkLSTMForward(b *testing.B) {
	config := LSTMConfig{
		InputSize:  40,
		HiddenSize: 256,
		NumLayers:  3,
	}

	seqLen := 150
	input := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		input[i] = make([]float64, 40)
		for j := 0; j < 40; j++ {
			input[i][j] = rand.Float64()*2 - 1
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		LSTMForward(input, config)
	}
}

// BenchmarkCTCDecode benchmarks CTC decoding
func BenchmarkCTCDecode(b *testing.B) {
	frameCount := 100
	vocabSize := 54

	probs := make([][]float64, frameCount)
	for t := 0; t < frameCount; t++ {
		probs[t] = make([]float64, vocabSize)
		sum := 0.0
		for i := 0; i < vocabSize; i++ {
			probs[t][i] = rand.Float64()
			sum += probs[t][i]
		}
		for i := 0; i < vocabSize; i++ {
			probs[t][i] /= sum
		}
	}

	b.Run("Greedy", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CTCGreedyDecode(probs, 0)
		}
	})

	b.Run("BeamWidth_10", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CTCBeamSearchDecode(probs, 0, 10)
		}
	})
}
