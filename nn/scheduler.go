package nn

import "math"

// SchedulerType defines the type of learning rate schedule.
type SchedulerType int

const (
	SchedulerLinear SchedulerType = iota
	SchedulerCosine
	SchedulerConstant
	SchedulerWarmupCosine
)

// Scheduler applies a learning rate schedule to an optimizer.
type Scheduler struct {
	opt           Optimizer
	schedulerType SchedulerType
	warmupSteps   int
	totalSteps    int
	currentStep   int
	baseLR        float64
	minLR         float64
}

func NewScheduler(opt Optimizer, schedulerType SchedulerType, warmupSteps, totalSteps int, baseLR float64) *Scheduler {
	return &Scheduler{
		opt:           opt,
		schedulerType: schedulerType,
		warmupSteps:   warmupSteps,
		totalSteps:    totalSteps,
		currentStep:   0,
		baseLR:        baseLR,
		minLR:         baseLR * 0.1,
	}
}

func (s *Scheduler) Step() {
	s.currentStep++
	lr := s.GetLR()
	s.opt.SetLearningRate(lr)
}

func (s *Scheduler) GetLR() float64 {
	if s.currentStep < s.warmupSteps {
		return s.baseLR * float64(s.currentStep) / float64(s.warmupSteps)
	}

	progress := float64(s.currentStep-s.warmupSteps) / float64(s.totalSteps-s.warmupSteps)
	if progress > 1 {
		progress = 1
	}

	switch s.schedulerType {
	case SchedulerLinear:
		return s.baseLR * (1.0 - progress)

	case SchedulerCosine:
		return s.minLR + (s.baseLR-s.minLR)*0.5*(1.0+math.Cos(math.Pi*progress))

	case SchedulerConstant:
		return s.baseLR

	case SchedulerWarmupCosine:
		return s.baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))

	default:
		return s.baseLR
	}
}

func (s *Scheduler) GetCurrentStep() int {
	return s.currentStep
}

func (s *Scheduler) SetTotalSteps(steps int) {
	s.totalSteps = steps
}

type LRHistory struct {
	Steps []int
	LRs   []float64
}

func NewLRHistory() *LRHistory {
	return &LRHistory{
		Steps: make([]int, 0),
		LRs:   make([]float64, 0),
	}
}

func (h *LRHistory) Record(step int, lr float64) {
	h.Steps = append(h.Steps, step)
	h.LRs = append(h.LRs, lr)
}

type WarmupScheduler struct {
	opt         Optimizer
	dModel      float64
	warmupSteps int
	currentStep int
	baseLR      float64
}

func NewWarmupScheduler(opt Optimizer, dModel float64, warmupSteps int, baseLR float64) *WarmupScheduler {
	return &WarmupScheduler{
		opt:         opt,
		dModel:      dModel,
		warmupSteps: warmupSteps,
		currentStep: 0,
		baseLR:      baseLR,
	}
}

func (w *WarmupScheduler) Step() {
	w.currentStep++
	lr := w.GetLR()
	w.opt.SetLearningRate(lr)
}

func (w *WarmupScheduler) GetLR() float64 {
	step := float64(w.currentStep)
	return math.Pow(w.dModel, -0.5) * math.Min(
		math.Pow(step, -0.5),
		step*math.Pow(float64(w.warmupSteps), -1.5),
	) * w.baseLR
}
