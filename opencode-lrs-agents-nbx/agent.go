package lrs

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// MetricsTracker tracks training metrics for analysis
type MetricsTracker struct {
	stepHistory     []int
	lossHistory     []float64
	lrHistory       []float64
	gradNormHistory []float64
	gradVarHistory  []float64
	windowSize      int
}

func NewMetricsTracker(windowSize int) *MetricsTracker {
	return &MetricsTracker{
		stepHistory:     make([]int, 0),
		lossHistory:     make([]float64, 0),
		lrHistory:       make([]float64, 0),
		gradNormHistory: make([]float64, 0),
		gradVarHistory:  make([]float64, 0),
		windowSize:      windowSize,
	}
}

func (m *MetricsTracker) Record(step int, loss, lr, gradNorm, gradVar float64) {
	m.stepHistory = append(m.stepHistory, step)
	m.lossHistory = append(m.lossHistory, loss)
	m.lrHistory = append(m.lrHistory, lr)
	m.gradNormHistory = append(m.gradNormHistory, gradNorm)
	m.gradVarHistory = append(m.gradVarHistory, gradVar)

	// Keep window bounded
	if len(m.stepHistory) > m.windowSize {
		m.stepHistory = m.stepHistory[len(m.stepHistory)-m.windowSize:]
		m.lossHistory = m.lossHistory[len(m.lossHistory)-m.windowSize:]
		m.lrHistory = m.lrHistory[len(m.lrHistory)-m.windowSize:]
		m.gradNormHistory = m.gradNormHistory[len(m.gradNormHistory)-m.windowSize:]
		m.gradVarHistory = m.gradVarHistory[len(m.gradVarHistory)-m.windowSize:]
	}
}

func (m *MetricsTracker) GetRecentLoss(k int) []float64 {
	if k > len(m.lossHistory) {
		k = len(m.lossHistory)
	}
	return m.lossHistory[len(m.lossHistory)-k:]
}

func (m *MetricsTracker) GetRecentGradNorm(k int) []float64 {
	if k > len(m.gradNormHistory) {
		k = len(m.gradNormHistory)
	}
	return m.gradNormHistory[len(m.gradNormHistory)-k:]
}

func (m *MetricsTracker) GetLossVariance() float64 {
	if len(m.lossHistory) < 2 {
		return 0
	}
	mean := 0.0
	for _, l := range m.lossHistory {
		mean += l
	}
	mean /= float64(len(m.lossHistory))

	var variance float64
	for _, l := range m.lossHistory {
		diff := l - mean
		variance += diff * diff
	}
	return variance / float64(len(m.lossHistory))
}

func (m *MetricsTracker) GetGradientEfficiency() float64 {
	// Ratio of effective learning (loss decrease per LR)
	if len(m.lossHistory) < 2 || len(m.lrHistory) < 2 {
		return 0
	}

	recentLosses := m.GetRecentLoss(100)
	recentLRs := m.lrHistory[len(m.lrHistory)-len(recentLosses):]

	if len(recentLosses) < 2 {
		return 0
	}

	// Compute average loss decrease rate weighted by LR
	totalDelta := 0.0
	totalLR := 0.0
	for i := 1; i < len(recentLosses); i++ {
		delta := recentLosses[i-1] - recentLosses[i]
		if delta > 0 {
			totalDelta += delta
			totalLR += recentLRs[i]
		}
	}

	if totalLR == 0 {
		return 0
	}
	return totalDelta / totalLR
}

func (m *MetricsTracker) GetLREffectiveness() float64 {
	// Measure how much the LR is contributing to learning
	if len(m.gradNormHistory) < 10 {
		return 0
	}

	// Look at correlation between LR and gradient norm
	recentLRs := m.lrHistory[len(m.lrHistory)-10:]
	recentGrads := m.gradNormHistory[len(m.gradNormHistory)-10:]

	lrMean, gradMean := 0.0, 0.0
	for i := range recentLRs {
		lrMean += recentLRs[i]
		gradMean += recentGrads[i]
	}
	lrMean /= float64(len(recentLRs))
	gradMean /= float64(len(recentGrads))

	cov := 0.0
	lrVar := 0.0
	gradVar := 0.0
	for i := range recentLRs {
		lrDiff := recentLRs[i] - lrMean
		gradDiff := recentGrads[i] - gradMean
		cov += lrDiff * gradDiff
		lrVar += lrDiff * lrDiff
		gradVar += gradDiff * gradDiff
	}

	if lrVar == 0 || gradVar == 0 {
		return 0
	}
	return cov / math.Sqrt(lrVar*gradVar)
}

// ============================================================
// SchedulerSelector: Analyzes training curves and picks the best scheduler
// ============================================================

type SchedulerScore struct {
	Name      string
	Score     float64
	Reasons   []string
}

// SchedulerSelector analyzes training dynamics and recommends schedulers
type SchedulerSelector struct {
	trackers    map[string]*MetricsTracker
	decisionHistory []string
	windowSize  int
}

func NewSchedulerSelector() *SchedulerSelector {
	return &SchedulerSelector{
		trackers:       make(map[string]*MetricsTracker),
		decisionHistory: make([]string, 0),
		windowSize:     100,
	}
}

func (s *SchedulerSelector) TrackScheduler(name string, tracker *MetricsTracker) {
	s.trackers[name] = tracker
}

func (s *SchedulerSelector) AnalyzeAndRecommend() []SchedulerScore {
	scores := make([]SchedulerScore, 0)

	for name, tracker := range s.trackers {
		score := 0.0
		reasons := make([]string, 0)

		// Score based on gradient efficiency
		efficiency := tracker.GetGradientEfficiency()
		score += efficiency * 10

		// Score based on LR effectiveness
		lrEff := tracker.GetLREffectiveness()
		score += lrEff * 5

		// Bonus for low loss variance (stable training)
		lossVar := tracker.GetLossVariance()
		if lossVar < 0.1 {
			score += 2
			reasons = append(reasons, "stable training")
		}

		// Bonus for positive loss trend
		recentLosses := tracker.GetRecentLoss(20)
		if len(recentLosses) >= 2 {
			trend := recentLosses[0] - recentLosses[len(recentLosses)-1]
			if trend > 0 {
				score += 3
				reasons = append(reasons, "decreasing loss")
			}
		}

		scores = append(scores, SchedulerScore{
			Name:    name,
			Score:   score,
			Reasons: reasons,
		})
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	return scores
}

func (s *SchedulerSelector) RecordDecision(decision string) {
	s.decisionHistory = append(s.decisionHistory, decision)
}

// ============================================================
// LRSAwareTrainer: Main training agent wrapping model + optimizer + scheduler
// ============================================================

// ModelInterface defines what the trainer expects from a model
type ModelInterface interface {
	Forward(x []float64) []float64
	Backward(gradOut []float64)
	GetParameters() [][]float64
}

// DummyModel is a simple model for testing
type DummyModel struct {
	weights []float64
	bias    float64
}

func NewDummyModel(paramSize int) *DummyModel {
	w := make([]float64, paramSize)
	for i := range w {
		w[i] = rand.NormFloat64() * 0.1
	}
	return &DummyModel{
		weights: w,
		bias:    rand.NormFloat64() * 0.1,
	}
}

func (m *DummyModel) Forward(x []float64) []float64 {
	// Simple linear model: y = w*x + b
	result := make([]float64, len(x))
	for i := range x {
		result[i] = m.weights[i%len(m.weights)]*x[i] + m.bias
	}
	return result
}

func (m *DummyModel) Backward(gradOut []float64) {
	// Compute gradients (simplified)
	// In real implementation, would compute actual gradients
}

func (m *DummyModel) GetParameters() [][]float64 {
	return [][]float64{m.weights, {m.bias}}
}

// LRSAwareTrainer wraps model, optimizer, and scheduler
type LRSAwareTrainer struct {
	model     ModelInterface
	optimizer interface {
		Step()
		ZeroGrad()
		SetLearningRate(lr float64)
	}
	scheduler LRScheduler
	metrics   *MetricsTracker
	config    *TrainerConfig
}

type TrainerConfig struct {
	MaxSteps        int
	EvalInterval    int
	PrintInterval   int
	GradientClip    float64
	WarmupSteps     int
	TargetLoss      float64
}

func NewLRSAwareTrainer(model ModelInterface, optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
}, scheduler LRScheduler, config *TrainerConfig) *LRSAwareTrainer {
	return &LRSAwareTrainer{
		model:     model,
		optimizer: optimizer,
		scheduler: scheduler,
		metrics:   NewMetricsTracker(500),
		config:    config,
	}
}

func (t *LRSAwareTrainer) Train(steps int) {
	for step := 0; step < steps; step++ {
		// Simulate forward pass and loss
		input := make([]float64, 64)
		for i := range input {
			input[i] = rand.Float64()*2 - 1
		}

		// Forward pass
		output := t.model.Forward(input)

		// Simulate loss (MSE against target)
		target := make([]float64, len(output))
		loss := 0.0
		for i := range output {
			target[i] = 0.5 // Target value
			diff := output[i] - target[i]
			loss += diff * diff
		}
		loss /= float64(len(output))

		// Backward pass
		gradOut := make([]float64, len(output))
		for i := range output {
			gradOut[i] = 2 * (output[i] - target[i]) / float64(len(output))
		}
		t.model.Backward(gradOut)

		// Simulate gradient norm
		gradNorm := 0.0
		for _, g := range gradOut {
			gradNorm += g * g
		}
		gradNorm = math.Sqrt(gradNorm)

		// Step optimizer with scheduler
		lr := t.scheduler.Step()
		t.optimizer.SetLearningRate(lr)
		t.optimizer.Step()

		// Record metrics
		t.metrics.Record(step, loss, lr, gradNorm, gradNorm*gradNorm*0.1)

		// Print progress
		if t.config.PrintInterval > 0 && step%t.config.PrintInterval == 0 {
			fmt.Printf("Step %d: Loss=%.6f LR=%.6f GradNorm=%.4f\n",
				step, loss, lr, gradNorm)
		}

		// Check early stopping
		if t.config.TargetLoss > 0 && loss < t.config.TargetLoss {
			fmt.Printf("Reached target loss %.6f at step %d\n", loss, step)
			break
		}
	}
}

func (t *LRSAwareTrainer) GetMetrics() *MetricsTracker {
	return t.metrics
}

func (t *LRSAwareTrainer) GetScheduler() LRScheduler {
	return t.scheduler
}

// ============================================================
// AdaptiveSchedulerAgent: Agent that can SWITCH between schedulers
// ============================================================

// SchedulerCandidate represents a candidate scheduler
type SchedulerCandidate struct {
	Name      string
	Scheduler LRScheduler
	Tracker   *MetricsTracker
	Score     float64
	StepsUsed int
}

// AdaptiveSchedulerAgent can switch between schedulers based on training dynamics
type AdaptiveSchedulerAgent struct {
	model      ModelInterface
	optimizer  interface {
		Step()
		ZeroGrad()
		SetLearningRate(lr float64)
	}
	candidates   []*SchedulerCandidate
	currentIndex int
	metrics      *MetricsTracker
	config       *AdaptiveConfig
	minStepsPerScheduler int
}

type AdaptiveConfig struct {
	MaxSteps             int
	EvalInterval         int
	SwitchThreshold      float64
	ImprovementThreshold float64
	CooldownSteps        int
}

func NewAdaptiveSchedulerAgent(model ModelInterface, optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
}, candidates []*SchedulerCandidate, config *AdaptiveConfig) *AdaptiveSchedulerAgent {
	// Initialize trackers for each candidate
	for _, c := range candidates {
		c.Tracker = NewMetricsTracker(200)
	}

	return &AdaptiveSchedulerAgent{
		model:                model,
		optimizer:            optimizer,
		candidates:           candidates,
		currentIndex:         0,
		metrics:              NewMetricsTracker(500),
		config:               config,
		minStepsPerScheduler: 50,
	}
}

func (a *AdaptiveSchedulerAgent) RunEpisode(steps int) {
	for step := 0; step < steps; step++ {
		// Training step
		input := make([]float64, 64)
		for i := range input {
			input[i] = rand.Float64()*2 - 1
		}

		output := a.model.Forward(input)
		target := make([]float64, len(output))
		loss := 0.0
		for i := range output {
			target[i] = 0.5
			diff := output[i] - target[i]
			loss += diff * diff
		}
		loss /= float64(len(output))

		gradOut := make([]float64, len(output))
		for i := range output {
			gradOut[i] = 2 * (output[i] - target[i]) / float64(len(output))
		}
		a.model.Backward(gradOut)

		gradNorm := 0.0
		for _, g := range gradOut {
			gradNorm += g * g
		}
		gradNorm = math.Sqrt(gradNorm)

		// Get current scheduler and step
		currentCandidate := a.candidates[a.currentIndex]
		lr := currentCandidate.Scheduler.Step()
		a.optimizer.SetLearningRate(lr)
		a.optimizer.Step()

		// Record metrics for current scheduler
		currentCandidate.Tracker.Record(step, loss, lr, gradNorm, gradNorm*gradNorm*0.1)
		currentCandidate.StepsUsed++

		// Global metrics
		a.metrics.Record(step, loss, lr, gradNorm, gradNorm*gradNorm*0.1)

		// Periodically analyze and potentially switch
		if step > 0 && step%a.config.EvalInterval == 0 && currentCandidate.StepsUsed >= a.minStepsPerScheduler {
			a.AnalyzeAndAdapt(step)
		}

		if step%100 == 0 {
			fmt.Printf("Step %d [%s]: Loss=%.6f LR=%.6f\n",
				step, currentCandidate.Name, loss, lr)
		}
	}
}

func (a *AdaptiveSchedulerAgent) AnalyzeAndAdapt(step int) {
	// Score each candidate
	for _, c := range a.candidates {
		// Compute score based on recent performance
		eff := c.Tracker.GetGradientEfficiency()
		lrEff := c.Tracker.GetLREffectiveness()
		recentLosses := c.Tracker.GetRecentLoss(20)

		lossTrend := 0.0
		if len(recentLosses) >= 2 {
			lossTrend = recentLosses[0] - recentLosses[len(recentLosses)-1]
		}

		c.Score = eff*10 + lrEff*5 + lossTrend*3
	}

	// Sort by score
	sort.Slice(a.candidates, func(i, j int) bool {
		return a.candidates[i].Score > a.candidates[j].Score
	})

	currentCandidate := a.candidates[a.currentIndex]
	bestCandidate := a.candidates[0]

	// Check if we should switch
	if currentCandidate.Name != bestCandidate.Name {
		improvement := bestCandidate.Score - currentCandidate.Score

		if improvement > a.config.ImprovementThreshold {
			// Switch to better scheduler
			for i, c := range a.candidates {
				if c.Name == bestCandidate.Name {
					a.currentIndex = i
					fmt.Printf("\n[Step %d] Switching from %s to %s (score: %.3f -> %.3f)\n\n",
						step, currentCandidate.Name, c.Name, currentCandidate.Score, c.Score)
					break
				}
			}
		}
	}
}

func (a *AdaptiveSchedulerAgent) GetCurrentSchedulerName() string {
	return a.candidates[a.currentIndex].Name
}

func (a *AdaptiveSchedulerAgent) GetMetrics() *MetricsTracker {
	return a.metrics
}

// ============================================================
// AutoScheduler: Automatic hyperparameter tuning for LR schedule
// ============================================================

type HyperParams struct {
	BaseLR     float64
	MaxLR      float64
	WarmupSteps int
	PctStart   float64
	DivFactor  float64
}

type TrialResult struct {
	Params    HyperParams
	FinalLoss float64
	StepsToConvergence int
}

// AutoScheduler performs automatic hyperparameter search for LR schedules
type AutoScheduler struct {
	model      ModelInterface
	baseConfig *TrainerConfig
	trials     []TrialResult
	bestTrial  *TrialResult
}

func NewAutoScheduler(model ModelInterface, baseConfig *TrainerConfig) *AutoScheduler {
	return &AutoScheduler{
		model:      model,
		baseConfig: baseConfig,
		trials:     make([]TrialResult, 0),
		bestTrial:  nil,
	}
}

func (a *AutoScheduler) RunSearch(numTrials int, stepsPerTrial int) *HyperParams {
	// Random search over hyperparameters
	for i := 0; i < numTrials; i++ {
		// Sample hyperparameters
		baseLR := math.Pow(10, -4+rand.Float64()*2) // 1e-4 to 1e-2
		maxLR := baseLR * (10 + rand.Float64()*40) // 10x to 50x base
		warmupSteps := 100 + rand.Intn(2000)
		pctStart := 0.1 + rand.Float64()*0.4
		divFactor := 10 + rand.Float64()*40

		params := HyperParams{
			BaseLR:      baseLR,
			MaxLR:       maxLR,
			WarmupSteps: warmupSteps,
			PctStart:    pctStart,
			DivFactor:   divFactor,
		}

		// Create scheduler with these params
		scheduler := NewOneCycleLR(baseLR,
			WithMaxLR(maxLR),
			WithWarmupSteps(warmupSteps),
			WithTotalSteps(stepsPerTrial),
			WithDivFactor(divFactor),
			WithFinalDivFactor(1e4),
		)

		// Create optimizer
		opt := CreateOptimizer("adam", baseLR, a.model.GetParameters())

		// Create trainer
		trainer := NewLRSAwareTrainer(a.model, opt, scheduler, a.baseConfig)

		// Train
		trainer.Train(stepsPerTrial)

		// Get final loss
		metrics := trainer.GetMetrics()
		finalLoss := 0.0
		if len(metrics.lossHistory) > 0 {
			// Average of last 10 losses
			start := len(metrics.lossHistory) - 10
			if start < 0 {
				start = 0
			}
			for _, l := range metrics.lossHistory[start:] {
				finalLoss += l
			}
			finalLoss /= float64(len(metrics.lossHistory) - start)
		}

		// Count steps to convergence
		stepsToConvergence := stepsPerTrial
		for step, loss := range metrics.lossHistory {
			if loss < 0.01 {
				stepsToConvergence = step
				break
			}
		}

		result := TrialResult{
			Params:               params,
			FinalLoss:            finalLoss,
			StepsToConvergence:   stepsToConvergence,
		}
		a.trials = append(a.trials, result)

		// Track best
		if a.bestTrial == nil || finalLoss < a.bestTrial.FinalLoss {
			bestCopy := result
			a.bestTrial = &bestCopy
		}

		fmt.Printf("Trial %d: BaseLR=%.6f MaxLR=%.4f FinalLoss=%.6f\n",
			i+1, baseLR, maxLR, finalLoss)
	}

	fmt.Printf("\nBest trial: BaseLR=%.6f MaxLR=%.4f FinalLoss=%.6f\n",
		a.bestTrial.Params.BaseLR, a.bestTrial.Params.MaxLR, a.bestTrial.FinalLoss)

	return &a.bestTrial.Params
}

func (a *AutoScheduler) GetTrials() []TrialResult {
	return a.trials
}

// ============================================================
// Utility Functions
// ============================================================

// ComputeGradientNorm computes the L2 norm of gradients
func ComputeGradientNorm(params [][]float64) float64 {
	var normSq float64
	for _, p := range params {
		for _, v := range p {
			normSq += v * v
		}
	}
	return math.Sqrt(normSq)
}

// ScaleGradients scales all gradients by a factor
func ScaleGradients(params [][]float64, scale float64) {
	for _, p := range params {
		for i := range p {
			p[i] *= scale
		}
	}
}

// AddNoiseToGradients adds Gaussian noise to gradients
func AddNoiseToGradients(params [][]float64, noiseScale float64) {
	rand.Seed(time.Now().UnixNano())
	for _, p := range params {
		for i := range p {
			p[i] += rand.NormFloat64() * noiseScale
		}
	}
}

// ClipGradients clips gradients by L2 norm
func ClipGradients(params [][]float64, maxNorm float64) float64 {
	norm := ComputeGradientNorm(params)
	if norm > maxNorm {
		scale := maxNorm / norm
		ScaleGradients(params, scale)
		return maxNorm
	}
	return norm
}
