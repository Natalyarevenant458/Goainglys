package lrs

import (
	"math"
)

// LRScheduler defines the interface for learning rate schedulers.
// All schedulers operate on float64 values for consistency.
type LRScheduler interface {
	// Step advances the scheduler by one step and returns the new LR
	Step() float64
	// GetLR returns the current learning rate without advancing
	GetLR() float64
	// GetStep returns the current step counter
	GetStep() int
	// Reset resets the scheduler to initial state
	Reset()
}

// SchedulerRegistry holds available scheduler types
type SchedulerRegistry map[string]func(baseLR float64, opts ...Option) LRScheduler

var globalRegistry = make(SchedulerRegistry)

// RegisterScheduler adds a scheduler to the global registry
func RegisterScheduler(name string, factory func(baseLR float64, opts ...Option) LRScheduler) {
	globalRegistry[name] = factory
}

// Option configures a scheduler
type Option func(*schedulerConfig)

type schedulerConfig struct {
	maxLR          float64
	minLR          float64
	warmupSteps    int
	totalSteps     int
	divFactor      float64
	finalDivFactor float64
	stepSize       int
	cycleSize      int
	gamma          float64
	T0             int
	TMult          float64
	power          float64
	maxEval        float64
	patience       int
	factor         float64
	cooldown       int
	pctStart       float64
	threePhase     bool
}

// WithMaxLR sets the maximum learning rate
func WithMaxLR(lr float64) Option {
	return func(c *schedulerConfig) { c.maxLR = lr }
}

// WithMinLR sets the minimum learning rate
func WithMinLR(lr float64) Option {
	return func(c *schedulerConfig) { c.minLR = lr }
}

// WithWarmupSteps sets the number of warmup steps
func WithWarmupSteps(steps int) Option {
	return func(c *schedulerConfig) { c.warmupSteps = steps }
}

// WithTotalSteps sets the total number of training steps
func WithTotalSteps(steps int) Option {
	return func(c *schedulerConfig) { c.totalSteps = steps }
}

// WithDivFactor sets the division factor for OneCycleLR
func WithDivFactor(div float64) Option {
	return func(c *schedulerConfig) { c.divFactor = div }
}

// WithFinalDivFactor sets the final division factor for OneCycleLR
func WithFinalDivFactor(div float64) Option {
	return func(c *schedulerConfig) { c.finalDivFactor = div }
}

// WithStepSize sets the step size for CyclicLR
func WithStepSize(size int) Option {
	return func(c *schedulerConfig) { c.stepSize = size }
}

// WithCycleSize sets the cycle size (alias for step size)
func WithCycleSize(size int) Option {
	return func(c *schedulerConfig) { c.cycleSize = size }
}

// WithGamma sets the decay gamma
func WithGamma(g float64) Option {
	return func(c *schedulerConfig) { c.gamma = g }
}

// WithT0 sets T0 for CosineAnnealingWarmRestarts
func WithT0(t0 int) Option {
	return func(c *schedulerConfig) { c.T0 = t0 }
}

// WithTMult sets T multiplier for CosineAnnealingWarmRestarts
func WithTMult(t float64) Option {
	return func(c *schedulerConfig) { c.TMult = t }
}

// WithPower sets the polynomial power
func WithPower(p float64) Option {
	return func(c *schedulerConfig) { c.power = p }
}

// WithPatience sets the patience for ReduceLROnPlateau
func WithPatience(p int) Option {
	return func(c *schedulerConfig) { c.patience = p }
}

// WithFactor sets the reduction factor
func WithFactor(f float64) Option {
	return func(c *schedulerConfig) { c.factor = f }
}

// WithCooldown sets the cooldown steps for ReduceLROnPlateau
func WithCooldown(csteps int) Option {
	return func(c *schedulerConfig) { c.cooldown = csteps }
}

// WithMaxEval sets the maximum evaluation value threshold
func WithMaxEval(v float64) Option {
	return func(c *schedulerConfig) { c.maxEval = v }
}

// ============================================================
// OneCycleLR: Smith 2018 - Super-Convergence
// Anneals from max_lr/div_factor to max_lr, then to max_lr/final_div_factor
// ============================================================

// OneCycleLR implements the OneCycle learning rate schedule
type OneCycleLR struct {
	maxLR          float64
	minLR          float64
	divFactor      float64
	finalDivFactor float64
	totalSteps     int
	currentStep    int
	warmupSteps    int
	decaySteps     int
	pctStart       float64
	threePhase     bool
}

func NewOneCycleLR(baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		maxLR:          baseLR,
		minLR:          baseLR,
		divFactor:     25.0,
		finalDivFactor: 1e4,
		totalSteps:     100000,
		warmupSteps:    0,
		pctStart:       0.3,
		threePhase:     false,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	// Calculate warmup and decay steps
	warmupSteps := int(float64(cfg.totalSteps) * cfg.pctStart)
	if warmupSteps < 1 {
		warmupSteps = 1
	}
	decaySteps := cfg.totalSteps - warmupSteps

	return &OneCycleLR{
		maxLR:          cfg.maxLR,
		minLR:          cfg.minLR / cfg.finalDivFactor,
		divFactor:      cfg.divFactor,
		finalDivFactor: cfg.finalDivFactor,
		totalSteps:     cfg.totalSteps,
		currentStep:    0,
		warmupSteps:    warmupSteps,
		decaySteps:     decaySteps,
		pctStart:       cfg.pctStart,
		threePhase:     cfg.threePhase,
	}
}

func (o *OneCycleLR) Step() float64 {
	o.currentStep++
	return o.GetLR()
}

func (o *OneCycleLR) GetLR() float64 {
	step := float64(o.currentStep)
	total := float64(o.totalSteps)

	if o.threePhase {
		// Three phase: warmup -> annealing -> fine-tuning
		phase1End := float64(o.warmupSteps)
		phase2End := phase1End + (total-phase1End)*0.9

		if step <= phase1End {
			// Phase 1: Linear warmup
			highLR := o.maxLR
			lowLR := o.maxLR / o.divFactor
			return lowLR + (highLR-lowLR)*(step/phase1End)
		} else if step <= phase2End {
			// Phase 2: Annealing
			progress := (step - phase1End) / (phase2End - phase1End)
			return o.maxLR * (1 - progress)
		} else {
			// Phase 3: Fine-tuning
			progress := (step - phase2End) / (total - phase2End)
			return o.minLR * (0.5 * (1 + math.Cos(math.Pi*progress)))
		}
	} else {
		// Two phase: warmup + annealing (standard OneCycle)
		if step <= float64(o.warmupSteps) {
			// Warmup phase
			highLR := o.maxLR
			lowLR := o.maxLR / o.divFactor
			return lowLR + (highLR-lowLR)*(step/float64(o.warmupSteps))
		}
		// Annealing phase
		progress := (step - float64(o.warmupSteps)) / float64(o.decaySteps)
		return o.maxLR * math.Pow(o.finalDivFactor, -progress)
	}
}

func (o *OneCycleLR) GetStep() int {
	return o.currentStep
}

func (o *OneCycleLR) Reset() {
	o.currentStep = 0
}

// ============================================================
// CyclicLR: Triangular/Clipped Triangular CLR
// ============================================================

// CycleMode defines the CLR mode
type CycleMode int

const (
	Triangular CycleMode = iota
	Triangular2
	ExpRange
)

// CyclicLR implements cyclic learning rate schedule
type CyclicLR struct {
	baseLR      float64
	maxLR       float64
	stepSize    int
	mode        CycleMode
	gamma       float64
	scaleFn     func(int) float64
	currentStep int
	clip        float64
}

func NewCyclicLR(baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		maxLR:    baseLR * 10,
		stepSize: 2000,
		gamma:    0.99994,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	clr := &CyclicLR{
		baseLR:      baseLR,
		maxLR:      cfg.maxLR,
		stepSize:    cfg.stepSize,
		mode:        Triangular,
		gamma:       cfg.gamma,
		scaleFn:     nil,
		currentStep: 0,
		clip:        0.0, // No clip by default
	}
	return clr
}

func (c *CyclicLR) Step() float64 {
	c.currentStep++
	return c.GetLR()
}

func (c *CyclicLR) GetLR() float64 {
	cycle := float64(c.currentStep / c.stepSize)
	x := float64(c.currentStep) / float64(c.stepSize)

	var scale float64
	switch c.mode {
	case Triangular:
		// Basic triangular
		scale = 1.0 - math.Abs(2*(x-cycle)-1)

	case Triangular2:
		// Triangular2: amplitude halves each cycle
		scale = 1.0 / math.Pow(2, cycle)

	case ExpRange:
		// Exponential range
		scale = c.gamma * x

	default:
		scale = 1.0
	}

	lr := c.baseLR + (c.maxLR-c.baseLR)*scale

	// Apply clipping if set
	if c.clip > 0 && lr > c.clip {
		lr = c.clip
	}

	return lr
}

func (c *CyclicLR) GetStep() int {
	return c.currentStep
}

func (c *CyclicLR) Reset() {
	c.currentStep = 0
}

// SetClip sets the maximum LR (clipping)
func (c *CyclicLR) SetClip(maxLR float64) {
	c.clip = maxLR
}

// SetMode sets the cycle mode
func (c *CyclicLR) SetMode(mode CycleMode) {
	c.mode = mode
}

// ============================================================
// CosineAnnealingWarmRestarts: SGDR with periodic restarts
// ============================================================

// CosineAnnealingWarmRestarts implements SGDR
type CosineAnnealingWarmRestarts struct {
	baseLR      float64
	minLR       float64
	T0          int
	TMult       float64
	currentStep int
	T_i         float64
	T_cur       float64
}

func NewCosineAnnealingWarmRestarts(baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		minLR:  baseLR * 0.001,
		T0:     10000,
		TMult:  2.0,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return &CosineAnnealingWarmRestarts{
		baseLR:      baseLR,
		minLR:       cfg.minLR,
		T0:          cfg.T0,
		TMult:       cfg.TMult,
		currentStep: 0,
		T_i:         float64(cfg.T0),
		T_cur:       0,
	}
}

func (c *CosineAnnealingWarmRestarts) Step() float64 {
	c.currentStep++
	c.T_cur++
	return c.GetLR()
}

func (c *CosineAnnealingWarmRestarts) GetLR() float64 {
	// Check if we need to restart
	if c.T_cur >= c.T_i {
		c.T_cur = 0
		c.T_i = c.T_i * c.TMult
	}

	// Cosine annealing with warm restarts
	return c.minLR + (c.baseLR-c.minLR)*0.5*(1+math.Cos(math.Pi*c.T_cur/c.T_i))
}

func (c *CosineAnnealingWarmRestarts) GetStep() int {
	return c.currentStep
}

func (c *CosineAnnealingWarmRestarts) Reset() {
	c.currentStep = 0
	c.T_i = float64(c.T0)
	c.T_cur = 0
}

// ============================================================
// PolynomialLR: Polynomial decay with optional warmup
// ============================================================

// PolynomialLR implements polynomial learning rate decay
type PolynomialLR struct {
	baseLR      float64
	minLR       float64
	totalSteps  int
	warmupSteps int
	power       float64
	currentStep int
	endLR       float64 // Linear warm to this LR instead of polynomial
	lrEndDecay  bool    // Continue decaying after totalSteps
}

func NewPolynomialLR(baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		minLR:       baseLR * 0.001,
		totalSteps:  100000,
		warmupSteps: 1000,
		power:       1.0,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return &PolynomialLR{
		baseLR:      baseLR,
		minLR:       cfg.minLR,
		totalSteps:  cfg.totalSteps,
		warmupSteps: cfg.warmupSteps,
		power:       cfg.power,
		currentStep: 0,
		endLR:       cfg.minLR,
		lrEndDecay:  false,
	}
}

func (p *PolynomialLR) Step() float64 {
	p.currentStep++
	return p.GetLR()
}

func (p *PolynomialLR) GetLR() float64 {
	step := float64(p.currentStep)
	total := float64(p.totalSteps)

	if step < float64(p.warmupSteps) {
		// Linear warmup
		return p.baseLR * step / float64(p.warmupSteps)
	}

	// Polynomial decay
	decaySteps := total - float64(p.warmupSteps)
	decayProgress := (step - float64(p.warmupSteps)) / decaySteps

	if decayProgress >= 1.0 {
		// After total steps, either continue decaying or stay at minLR
		if p.lrEndDecay {
			return p.endLR * math.Pow(0.01, decayProgress-1)
		}
		return p.minLR
	}

	// Polynomial decay formula
	return (p.baseLR-p.minLR)*math.Pow(1-decayProgress, p.power) + p.minLR
}

func (p *PolynomialLR) GetStep() int {
	return p.currentStep
}

func (p *PolynomialLR) Reset() {
	p.currentStep = 0
}

// SetEndDecay enables continued decay past totalSteps
func (p *PolynomialLR) SetEndDecay(enabled bool) {
	p.lrEndDecay = enabled
}

// ============================================================
// ExponentialWarmup: Exponential LR ramp during warmup
// ============================================================

// ExponentialWarmup implements exponential warmup then exponential decay
type ExponentialWarmup struct {
	baseLR      float64
	minLR       float64
	warmupSteps int
	totalSteps  int
	currentStep int
	decayGamma  float64
}

func NewExponentialWarmup(baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		minLR:       baseLR * 0.001,
		warmupSteps: 1000,
		totalSteps:  100000,
		gamma:       0.9999,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return &ExponentialWarmup{
		baseLR:      baseLR,
		minLR:       cfg.minLR,
		warmupSteps: cfg.warmupSteps,
		totalSteps:  cfg.totalSteps,
		currentStep: 0,
		decayGamma:  cfg.gamma,
	}
}

func (e *ExponentialWarmup) Step() float64 {
	e.currentStep++
	return e.GetLR()
}

func (e *ExponentialWarmup) GetLR() float64 {
	step := float64(e.currentStep)

	if step < float64(e.warmupSteps) {
		// Exponential warmup
		return e.baseLR * math.Exp(-2.0*(1-step/float64(e.warmupSteps)))
	}

	// Exponential decay after warmup
	progress := (step - float64(e.warmupSteps)) / float64(e.totalSteps-e.warmupSteps)
	return e.minLR + (e.baseLR-e.minLR)*math.Pow(e.decayGamma, progress*float64(e.totalSteps))
}

func (e *ExponentialWarmup) GetStep() int {
	return e.currentStep
}

func (e *ExponentialWarmup) Reset() {
	e.currentStep = 0
}

// ============================================================
// ReduceLROnPlateau: Reduce on validation metric plateau
// ============================================================

// ReduceLROnPlateau reduces LR when metric plateaus
type ReduceLROnPlateau struct {
	optimizer   interface{ SetLearningRate(lr float64) }
	baseLR      float64
	minLR       float64
	factor      float64
	patience    int
	cooldown    int
	mode        string // "min" or "max"
	currentStep int
	bestMetric  float64
	badEpochs   int
	inCooldown  int
	lastLR      float64
	metricHistory []float64
}

func NewReduceLROnPlateau(optimizer interface{ SetLearningRate(lr float64) }, baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		minLR:    baseLR * 0.0001,
		factor:   0.1,
		patience: 10,
		cooldown: 5,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	r := &ReduceLROnPlateau{
		optimizer:    optimizer,
		baseLR:       baseLR,
		minLR:        cfg.minLR,
		factor:       cfg.factor,
		patience:     cfg.patience,
		cooldown:     cfg.cooldown,
		mode:         "min",
		currentStep:  0,
		bestMetric:   math.Inf(1), // For min mode
		badEpochs:    0,
		inCooldown:   0,
		lastLR:       baseLR,
		metricHistory: make([]float64, 0),
	}
	return r
}

func (r *ReduceLROnPlateau) Step() float64 {
	r.currentStep++
	r.optimizer.SetLearningRate(r.lastLR)
	return r.lastLR
}

// StepWithMetric updates scheduler with a metric value
func (r *ReduceLROnPlateau) StepWithMetric(metric float64) float64 {
	r.currentStep++
	r.metricHistory = append(r.metricHistory, metric)

	// Check if metric improved
	improved := false
	if r.mode == "min" {
		improved = metric < r.bestMetric
	} else {
		improved = metric > r.bestMetric
	}

	if improved {
		r.bestMetric = metric
		r.badEpochs = 0
	} else {
		r.badEpochs++
	}

	// Handle cooldown
	if r.inCooldown > 0 {
		r.inCooldown--
	} else if r.badEpochs >= r.patience {
		// Reduce LR
		newLR := r.lastLR * r.factor
		if newLR < r.minLR {
			newLR = r.minLR
		}
		r.lastLR = newLR
		r.badEpochs = 0
		r.inCooldown = r.cooldown
	}

	r.optimizer.SetLearningRate(r.lastLR)
	return r.lastLR
}

func (r *ReduceLROnPlateau) GetLR() float64 {
	return r.lastLR
}

func (r *ReduceLROnPlateau) GetStep() int {
	return r.currentStep
}

func (r *ReduceLROnPlateau) Reset() {
	r.currentStep = 0
	r.bestMetric = math.Inf(1)
	r.badEpochs = 0
	r.inCooldown = 0
	r.lastLR = r.baseLR
}

// GetMetricHistory returns the recorded metrics
func (r *ReduceLROnPlateau) GetMetricHistory() []float64 {
	return r.metricHistory
}

// ============================================================
// AdaptiveLR: Per-parameter LR based on gradient statistics
// ============================================================

// AdaptiveLR implements per-parameter LR adaptation based on gradient stats
type AdaptiveLR struct {
	optimizer    interface{ SetLearningRate(lr float64) }
	baseLR       float64
	minLR        float64
	maxLR        float64
	currentStep  int
	windowSize   int
	gradHistory  []float64
	gradVariance []float64
	momentum     float64
	eps          float64
}

func NewAdaptiveLR(optimizer interface{ SetLearningRate(lr float64) }, baseLR float64, opts ...Option) LRScheduler {
	cfg := &schedulerConfig{
		minLR: baseLR * 0.1,
		maxLR: baseLR * 10,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	return &AdaptiveLR{
		optimizer:    optimizer,
		baseLR:       baseLR,
		minLR:        cfg.minLR,
		maxLR:        cfg.maxLR,
		currentStep:  0,
		windowSize:   100,
		gradHistory:  make([]float64, 0),
		gradVariance: make([]float64, 0),
		momentum:     0.9,
		eps:          1e-8,
	}
}

func (a *AdaptiveLR) Step() float64 {
	a.currentStep++
	return a.GetLR()
}

func (a *AdaptiveLR) StepWithGradient(gradNorm float64) float64 {
	a.currentStep++
	a.gradHistory = append(a.gradHistory, gradNorm)

	// Calculate variance in a window
	if len(a.gradHistory) > a.windowSize {
		a.gradHistory = a.gradHistory[len(a.gradHistory)-a.windowSize:]
	}

	// Compute mean
	mean := 0.0
	for _, g := range a.gradHistory {
		mean += g
	}
	mean /= float64(len(a.gradHistory))

	// Compute variance
	var variance float64
	for _, g := range a.gradHistory {
		diff := g - mean
		variance += diff * diff
	}
	variance /= float64(len(a.gradHistory))

	// Smoothed variance update
	if len(a.gradVariance) > 0 {
		variance = a.momentum*a.gradVariance[len(a.gradVariance)-1] + (1-a.momentum)*variance
	}
	a.gradVariance = append(a.gradVariance, variance)

	// Adaptive LR based on variance
	// Low variance = stable gradients -> can increase LR
	// High variance = unstable gradients -> decrease LR
	normalizedVar := math.Sqrt(variance + a.eps)
	lr := a.baseLR / normalizedVar

	// Clamp LR
	if lr < a.minLR {
		lr = a.minLR
	}
	if lr > a.maxLR {
		lr = a.maxLR
	}

	a.optimizer.SetLearningRate(lr)
	return lr
}

func (a *AdaptiveLR) GetLR() float64 {
	return a.baseLR
}

func (a *AdaptiveLR) GetStep() int {
	return a.currentStep
}

func (a *AdaptiveLR) Reset() {
	a.currentStep = 0
	a.gradHistory = make([]float64, 0)
	a.gradVariance = make([]float64, 0)
}

// GetVarianceHistory returns gradient variance history
func (a *AdaptiveLR) GetVarianceHistory() []float64 {
	return a.gradVariance
}

// ============================================================
// MultiScheduler: Combines multiple schedulers
// ============================================================

// MultiScheduler combines multiple schedulers
type MultiScheduler struct {
	schedulers []LRScheduler
	weights    []float64
	current    int
}

func NewMultiScheduler(schedulers []LRScheduler, weights []float64) *MultiScheduler {
	if len(weights) == 0 {
		weights = make([]float64, len(schedulers))
		for i := range weights {
			weights[i] = 1.0 / float64(len(schedulers))
		}
	}
	return &MultiScheduler{
		schedulers: schedulers,
		weights:    weights,
		current:    0,
	}
}

func (m *MultiScheduler) Step() float64 {
	m.current++
	totalLR := 0.0
	for i, s := range m.schedulers {
		lr := s.Step()
		totalLR += lr * m.weights[i]
	}
	return totalLR
}

func (m *MultiScheduler) GetLR() float64 {
	totalLR := 0.0
	for i, s := range m.schedulers {
		totalLR += s.GetLR() * m.weights[i]
	}
	return totalLR
}

func (m *MultiScheduler) GetStep() int {
	if len(m.schedulers) > 0 {
		return m.schedulers[0].GetStep()
	}
	return m.current
}

func (m *MultiScheduler) Reset() {
	m.current = 0
	for _, s := range m.schedulers {
		s.Reset()
	}
}

func init() {
	RegisterScheduler("onecycle", NewOneCycleLR)
	RegisterScheduler("cyclic", NewCyclicLR)
	RegisterScheduler("cosine_annealing_warm_restarts", NewCosineAnnealingWarmRestarts)
	RegisterScheduler("polynomial", NewPolynomialLR)
	RegisterScheduler("exponential_warmup", NewExponentialWarmup)
}
