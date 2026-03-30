package meta_learner

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"self_improving_agents/core"
)

type Learner struct {
	state             core.MetaLearningState
	learningRate      float64
	strategies        map[string]*StrategyMetrics
	difficultyWeights map[string]float64
}

type StrategyMetrics struct {
	Attempts    int
	TotalGain   float64
	AvgGain     float64
	LastAttempt time.Time
}

func NewLearner() *Learner {
	return &Learner{
		state: core.MetaLearningState{
			EffectiveStrategies:         make(map[string]float64),
			DiminishingReturnsThreshold: 0.01,
			CurrentFocus:                "general",
			LearningRate:                0.01,
		},
		learningRate:      0.1,
		strategies:        make(map[string]*StrategyMetrics),
		difficultyWeights: make(map[string]float64),
	}
}

func (l *Learner) Update(history []core.ImprovementCycle) {
	l.state.ImprovementHistory = history

	if len(history) == 0 {
		return
	}

	l.updateStrategyMetrics(history)
	l.adjustLearningRate(history)
	l.updateDifficultyWeights(history)
	l.detectDiminishingReturns(history)
}

func (l *Learner) updateStrategyMetrics(history []core.ImprovementCycle) {
	for _, cycle := range history {
		for _, weakness := range cycle.WeaknessesAddressed {
			taskType := weakness.TaskType
			if _, ok := l.strategies[taskType]; !ok {
				l.strategies[taskType] = &StrategyMetrics{}
			}

			metrics := l.strategies[taskType]
			metrics.Attempts++
			metrics.TotalGain += cycle.Improvement
			metrics.AvgGain = metrics.TotalGain / float64(metrics.Attempts)
			metrics.LastAttempt = time.Now()

			l.state.EffectiveStrategies[taskType] = metrics.AvgGain
		}
	}

	l.updateCurrentFocus()
}

func (l *Learner) updateCurrentFocus() {
	var bestType string
	var bestGain float64 = -math.MaxFloat64

	for taskType, gain := range l.state.EffectiveStrategies {
		if gain > bestGain {
			bestGain = gain
			bestType = taskType
		}
	}

	if bestType != "" {
		l.state.CurrentFocus = bestType
	}
}

func (l *Learner) adjustLearningRate(history []core.ImprovementCycle) {
	if len(history) < 2 {
		return
	}

	recentImprovements := make([]float64, 0)
	for i := len(history) - min(5, len(history)); i < len(history); i++ {
		recentImprovements = append(recentImprovements, history[i].Improvement)
	}

	avgRecentImprovement := sum(recentImprovements) / float64(len(recentImprovements))

	threshold := l.state.DiminishingReturnsThreshold
	if avgRecentImprovement < threshold {
		l.learningRate *= 0.9
		if l.learningRate < 0.001 {
			l.learningRate = 0.001
		}
	} else if avgRecentImprovement > threshold*10 {
		l.learningRate *= 1.1
		if l.learningRate > 1.0 {
			l.learningRate = 1.0
		}
	}

	l.state.LearningRate = float32(l.learningRate)
}

func (l *Learner) updateDifficultyWeights(history []core.ImprovementCycle) {
	for _, cycle := range history {
		for _, weakness := range cycle.WeaknessesAddressed {
			taskType := weakness.TaskType
			currentWeight := l.difficultyWeights[taskType]

			improvementRatio := cycle.Improvement / max(weakness.Difficulty, 0.01)

			if improvementRatio > 1.0 {
				l.difficultyWeights[taskType] = currentWeight + 0.1
			} else if improvementRatio < 0.5 {
				l.difficultyWeights[taskType] = currentWeight - 0.1
			}

			if l.difficultyWeights[taskType] < 0.1 {
				l.difficultyWeights[taskType] = 0.1
			}
			if l.difficultyWeights[taskType] > 2.0 {
				l.difficultyWeights[taskType] = 2.0
			}
		}
	}
}

func (l *Learner) detectDiminishingReturns(history []core.ImprovementCycle) {
	if len(history) < 5 {
		return
	}

	windowSize := 3
	recentWindow := history[len(history)-windowSize:]
	olderWindow := history[len(history)-2*windowSize : len(history)-windowSize]

	recentAvg := avgImprovement(recentWindow)
	olderAvg := avgImprovement(olderWindow)

	if olderAvg > 0 && recentAvg/olderAvg < 0.5 {
		l.state.DiminishingReturnsThreshold *= 0.5
		if l.state.DiminishingReturnsThreshold < 0.001 {
			l.state.DiminishingReturnsThreshold = 0.001
		}
	}
}

func (l *Learner) GetRecommendedStrategy(taskType string) string {
	if metrics, ok := l.strategies[taskType]; ok && metrics.AvgGain > 0 {
		return fmt.Sprintf("Continue focused practice on %s (avg improvement: %.4f per cycle)", taskType, metrics.AvgGain)
	}
	return fmt.Sprintf("Explore new approaches for %s", taskType)
}

func (l *Learner) GetState() core.MetaLearningState {
	return l.state
}

func (l *Learner) GetEffectiveStrategies() map[string]float64 {
	effective := make(map[string]float64)
	for k, v := range l.state.EffectiveStrategies {
		effective[k] = v
	}
	return effective
}

func (l *Learner) Save(path string) error {
	data, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal meta-learner state: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

func (l *Learner) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read meta-learner state: %w", err)
	}
	return json.Unmarshal(data, l)
}

func sum(vals []float64) float64 {
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return s
}

func avgImprovement(cycles []core.ImprovementCycle) float64 {
	if len(cycles) == 0 {
		return 0
	}
	total := 0.0
	for _, c := range cycles {
		total += c.Improvement
	}
	return total / float64(len(cycles))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
