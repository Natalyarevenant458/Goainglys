// Package nn provides unified neural network building blocks: layers, optimizers,
// and learning rate schedulers. This package serves as the canonical implementation
// for the Goainglys ML platform, built on the tensor package.
//
// All implementations use float64 precision.
package nn

import (
	"math"
)

// ============================================================
//   OPTIMIZER INTERFACE AND IMPLEMENTATIONS
// ============================================================

type Optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
}

type ParamSlot struct {
	Data []float64
	Grad []float64
	M1   []float64 // first moment
	M2   []float64 // second moment
	Step int
}

type Adam struct {
	Slots []*ParamSlot
	lr    float64
	beta1 float64
	beta2 float64
	eps   float64
	step  int
}

func NewAdam(lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		Slots: make([]*ParamSlot, 0),
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
		eps:   eps,
	}
}

func (a *Adam) AddParams(params [][]float64) {
	for _, p := range params {
		slot := &ParamSlot{
			Data: p,
			Grad: make([]float64, len(p)),
			M1:   make([]float64, len(p)),
			M2:   make([]float64, len(p)),
		}
		a.Slots = append(a.Slots, slot)
	}
}

func (a *Adam) Step() {
	a.step++
	bc1 := 1.0 - math.Pow(a.beta1, float64(a.step))
	bc2 := 1.0 - math.Pow(a.beta2, float64(a.step))

	for _, s := range a.Slots {
		if len(s.Grad) == 0 {
			continue
		}
		for i := range s.Data {
			g := s.Grad[i]

			s.M1[i] = a.beta1*s.M1[i] + (1.0-a.beta1)*g
			s.M2[i] = a.beta2*s.M2[i] + (1.0-a.beta2)*g*g

			mHat := s.M1[i] / bc1
			vHat := s.M2[i] / bc2

			s.Data[i] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
	}
}

func (a *Adam) ZeroGrad() {
	for _, s := range a.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (a *Adam) SetLearningRate(lr float64) {
	a.lr = lr
}

type AdamW struct {
	Slots       []*ParamSlot
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int
}

func NewAdamW(lr, beta1, beta2, eps, weightDecay float64) *AdamW {
	return &AdamW{
		Slots:       make([]*ParamSlot, 0),
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
	}
}

func (a *AdamW) AddParams(params [][]float64) {
	for _, p := range params {
		slot := &ParamSlot{
			Data: p,
			Grad: make([]float64, len(p)),
			M1:   make([]float64, len(p)),
			M2:   make([]float64, len(p)),
		}
		a.Slots = append(a.Slots, slot)
	}
}

func (a *AdamW) Step() {
	a.step++
	bc1 := 1.0 - math.Pow(a.beta1, float64(a.step))
	bc2 := 1.0 - math.Pow(a.beta2, float64(a.step))

	for _, s := range a.Slots {
		if len(s.Grad) == 0 {
			continue
		}
		for i := range s.Data {
			g := s.Grad[i]
			if a.weightDecay > 0 {
				g += a.weightDecay * s.Data[i]
			}

			s.M1[i] = a.beta1*s.M1[i] + (1.0-a.beta1)*g
			s.M2[i] = a.beta2*s.M2[i] + (1.0-a.beta2)*g*g

			mHat := s.M1[i] / bc1
			vHat := s.M2[i] / bc2

			s.Data[i] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
	}
}

func (a *AdamW) ZeroGrad() {
	for _, s := range a.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (a *AdamW) SetLearningRate(lr float64) {
	a.lr = lr
}

type SGD struct {
	Slots    []*ParamSlot
	lr       float64
	momentum float64
}

func NewSGD(lr, momentum float64) *SGD {
	return &SGD{
		Slots:    make([]*ParamSlot, 0),
		lr:       lr,
		momentum: momentum,
	}
}

func (s *SGD) AddParams(params [][]float64) {
	for _, p := range params {
		slot := &ParamSlot{
			Data: p,
			Grad: make([]float64, len(p)),
			M1:   make([]float64, len(p)),
		}
		s.Slots = append(s.Slots, slot)
	}
}

func (s *SGD) Step() {
	for _, slot := range s.Slots {
		if len(slot.Grad) == 0 {
			continue
		}
		for i := range slot.Data {
			g := slot.Grad[i]
			if s.momentum > 0 {
				slot.M1[i] = s.momentum*slot.M1[i] + g
				g = slot.M1[i]
			}
			slot.Data[i] -= s.lr * g
		}
	}
}

func (s *SGD) ZeroGrad() {
	for _, slot := range s.Slots {
		for i := range slot.Grad {
			slot.Grad[i] = 0
		}
	}
}

func (s *SGD) SetLearningRate(lr float64) {
	s.lr = lr
}

type RMSprop struct {
	Slots    []*ParamSlot
	lr       float64
	alpha    float64
	eps      float64
	momentum float64
}

func NewRMSprop(lr, alpha, eps, momentum float64) *RMSprop {
	return &RMSprop{
		Slots:    make([]*ParamSlot, 0),
		lr:       lr,
		alpha:    alpha,
		eps:      eps,
		momentum: momentum,
	}
}

func (r *RMSprop) AddParams(params [][]float64) {
	for _, p := range params {
		slot := &ParamSlot{
			Data: p,
			Grad: make([]float64, len(p)),
			M1:   make([]float64, len(p)),
			M2:   make([]float64, len(p)),
		}
		r.Slots = append(r.Slots, slot)
	}
}

func (r *RMSprop) Step() {
	for _, s := range r.Slots {
		if len(s.Grad) == 0 {
			continue
		}
		for i := range s.Data {
			g := s.Grad[i]
			s.M2[i] = r.alpha*s.M2[i] + (1.0-r.alpha)*g*g
			denom := math.Sqrt(s.M2[i]) + r.eps
			update := r.lr * g / denom
			if r.momentum > 0 {
				s.M1[i] = r.momentum*s.M1[i] + update
				s.Data[i] -= s.M1[i]
			} else {
				s.Data[i] -= update
			}
		}
	}
}

func (r *RMSprop) ZeroGrad() {
	for _, slot := range r.Slots {
		for i := range slot.Grad {
			slot.Grad[i] = 0
		}
	}
}

func (r *RMSprop) SetLearningRate(lr float64) {
	r.lr = lr
}

type Lion struct {
	Slots []*ParamSlot
	lr    float64
	beta1 float64
	beta2 float64
	step  int
}

func NewLion(lr, beta1, beta2 float64) *Lion {
	return &Lion{
		Slots: make([]*ParamSlot, 0),
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
	}
}

func (l *Lion) AddParams(params [][]float64) {
	for _, p := range params {
		slot := &ParamSlot{
			Data: p,
			Grad: make([]float64, len(p)),
			M1:   make([]float64, len(p)),
		}
		l.Slots = append(l.Slots, slot)
	}
}

func (l *Lion) Step() {
	l.step++
	for _, s := range l.Slots {
		if len(s.Grad) == 0 {
			continue
		}
		for i := range s.Data {
			g := s.Grad[i]
			s.M1[i] = l.beta1*s.M1[i] + (1.0-l.beta1)*g
			s.Data[i] -= l.lr * s.M1[i]
		}
	}
}

func (l *Lion) ZeroGrad() {
	for _, s := range l.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (l *Lion) SetLearningRate(lr float64) {
	l.lr = lr
}

// OptimizerManager manages multiple optimizers.
type OptimizerManager struct {
	optimizers []Optimizer
}

func NewOptimizerManager() *OptimizerManager {
	return &OptimizerManager{make([]Optimizer, 0)}
}

func (m *OptimizerManager) Add(opt Optimizer) {
	m.optimizers = append(m.optimizers, opt)
}

func (m *OptimizerManager) Step() {
	for _, opt := range m.optimizers {
		opt.Step()
	}
}

func (m *OptimizerManager) ZeroGrad() {
	for _, opt := range m.optimizers {
		opt.ZeroGrad()
	}
}

func (m *OptimizerManager) SetLearningRate(lr float64) {
	for _, opt := range m.optimizers {
		opt.SetLearningRate(lr)
	}
}
