package weakness_detector

import (
	"fmt"
	"math"
	"strings"

	"self_improving_agents/core"
)

type Detector struct {
	errorLog     []ErrorRecord
	taskPatterns map[string]int
}

type ErrorRecord struct {
	Task       string
	TaskType   string
	Correct    bool
	Attempts   int
	TimeSpent  float64
	Difficulty float64
}

func NewDetector() *Detector {
	return &Detector{
		errorLog:     make([]ErrorRecord, 0),
		errors:       make(map[string]int),
		taskPatterns: make(map[string]int),
	}
}

func (d *Detector) RecordResult(task, taskType string, correct bool, attempts int, timeSpent, difficulty float64) {
	d.errorLog = append(d.errorLog, ErrorRecord{
		Task:       task,
		TaskType:   taskType,
		Correct:    correct,
		Attempts:   attempts,
		TimeSpent:  timeSpent,
		Difficulty: difficulty,
	})
	d.taskPatterns[taskType]++
	if !correct {
		d.errors[taskType]++
	}
}

func (d *Detector) Detect(agent core.AgentInterface, task string, validationSetSize int) ([]core.Weakness, error) {
	var weaknesses []core.Weakness

	// Analyze error patterns by task type
	typeStats := make(map[string]*taskStats)
	for _, rec := range d.errorLog {
		if _, ok := typeStats[rec.TaskType]; !ok {
			typeStats[rec.TaskType] = &taskStats{total: 0, errors: 0, totalDifficulty: 0}
		}
		ts := typeStats[rec.TaskType]
		ts.total++
		if !rec.Correct {
			ts.errors++
		}
		ts.totalDifficulty += rec.Difficulty
	}

	// Identify weaknesses from error patterns
	for taskType, stats := range typeStats {
		if stats.total < 3 {
			continue // Need minimum samples
		}

		errorRate := float64(stats.errors) / float64(stats.total)
		avgDifficulty := stats.totalDifficulty / float64(stats.total)

		if errorRate > 0.2 { // More than 20% error rate
			weakness := d.analyzeWeakness(taskType, errorRate, avgDifficulty, stats)
			weaknesses = append(weaknesses, weakness)
		}
	}

	// Analyze confusion patterns (tasks where agent spent too long or needed many attempts)
	confusion := d.detectConfusion()
	for _, c := range confusion {
		weaknesses = append(weaknesses, c)
	}

	// Analyze difficulty thresholds
	difficultyGaps := d.detectDifficultyGaps()
	for _, g := range difficultyGaps {
		weaknesses = append(weaknesses, g)
	}

	// If no specific weaknesses found, return general ones
	if len(weaknesses) == 0 {
		weaknesses = append(weaknesses, core.Weakness{
			ID:             "general",
			Description:    "General performance gap - try varied difficulty",
			TaskType:       "general",
			Difficulty:     0.5,
			Frequency:      len(d.errorLog),
			SuggestedFocus: "Balanced practice across difficulty levels",
		})
	}

	return weaknesses, nil
}

type taskStats struct {
	total           int
	errors          int
	totalDifficulty float64
}

func (d *Detector) analyzeWeakness(taskType string, errorRate, avgDifficulty float64, stats *taskStats) core.Weakness {
	description := d.getWeaknessDescription(taskType, errorRate)
	suggestedFocus := d.getSuggestedFocus(taskType)

	return core.Weakness{
		ID:             fmt.Sprintf("weakness_%s_%d", taskType, int(errorRate*100)),
		Description:    description,
		TaskType:       taskType,
		Difficulty:     math.Min(1.0, avgDifficulty+0.1),
		Frequency:      stats.errors,
		Examples:       d.getExamples(taskType, 3),
		SuggestedFocus: suggestedFocus,
	}
}

func (d *Detector) getWeaknessDescription(taskType string, errorRate float64) string {
	switch {
	case strings.Contains(taskType, "arithmetic"):
		if errorRate > 0.5 {
			return "Severe difficulty with arithmetic operations"
		}
		return "Difficulty with arithmetic computation"
	case strings.Contains(taskType, "algebra"):
		if errorRate > 0.5 {
			return "Severe difficulty with algebraic manipulation"
		}
		return "Difficulty solving algebraic equations"
	case strings.Contains(taskType, "reasoning"):
		return "Weak logical reasoning and step-by-step problem solving"
	case strings.Contains(taskType, "word_problem"):
		return "Difficulty translating word problems to mathematical expressions"
	case strings.Contains(taskType, "fraction"):
		return "Difficulty with fraction operations"
	case strings.Contains(taskType, "geometry"):
		return "Weak spatial reasoning for geometric problems"
	default:
		return fmt.Sprintf("Difficulty with %s tasks (%.0f%% error rate)", taskType, errorRate*100)
	}
}

func (d *Detector) getSuggestedFocus(taskType string) string {
	switch {
	case strings.Contains(taskType, "arithmetic"):
		return "Practice arithmetic with varied numbers, focus on order of operations"
	case strings.Contains(taskType, "algebra"):
		return "Study algebraic manipulation rules, practice isolating variables"
	case strings.Contains(taskType, "reasoning"):
		return "Break problems into smaller steps, verify each step"
	case strings.Contains(taskType, "word_problem"):
		return "Identify key quantities, translate to equations step by step"
	case strings.Contains(taskType, "fraction"):
		return "Review fraction arithmetic, common denominators"
	case strings.Contains(taskType, "geometry"):
		return "Practice spatial visualization, learn geometric formulas"
	default:
		return "Targeted practice on " + taskType + " problems"
	}
}

func (d *Detector) getExamples(taskType string, n int) []string {
	examples := map[string][]string{
		"arithmetic":   {"Calculate: 23 * 47 + 91 / 13", "Solve: 156 - 89 + 34 * 2", "What is 15% of 840?"},
		"algebra":      {"Solve: 3x + 7 = 22", "If y = 2x + 1 and x = 4, find y", "Simplify: 2(x + 3) - x"},
		"reasoning":    {"If all A are B, and all B are C, what about A and C?", "Find the pattern: 2, 6, 12, 20, ?", "Which is larger: 3/7 or 4/9?"},
		"word_problem": {"A train travels 60mph for 2.5 hours. How far?", "John has 3x coins, gives 5 away, has 16 left. Find x", "Area of rectangle is 72, width is 8. Find perimeter"},
		"fraction":     {"Add: 3/4 + 5/6", "What is 2/3 of 3/4?", "Compare: 5/8 and 2/3"},
		"geometry":     {"Find area of triangle with base 8, height 5", "A circle has radius 7. Find circumference", "Find hypotenuse of right triangle with legs 5 and 12"},
	}
	if exs, ok := examples[taskType]; ok {
		if len(exs) >= n {
			return exs[:n]
		}
		return exs
	}
	return []string{"Practice " + taskType + " problems"}
}

func (d *Detector) detectConfusion() []core.Weakness {
	var weaknesses []core.Weakness
	confusedTasks := make(map[string]int)

	for _, rec := range d.errorLog {
		if !rec.Correct && rec.Attempts > 1 {
			confusedTasks[rec.TaskType]++
		}
	}

	for taskType, count := range confusedTasks {
		if count >= 2 {
			weaknesses = append(weaknesses, core.Weakness{
				ID:             fmt.Sprintf("confusion_%s", taskType),
				Description:    fmt.Sprintf("Repeated failures in %s - agent struggles to self-correct", taskType),
				TaskType:       taskType,
				Difficulty:     0.6,
				Frequency:      count,
				SuggestedFocus: "Practice with hints or step-by-step decomposition",
			})
		}
	}

	return weaknesses
}

func (d *Detector) detectDifficultyGaps() []core.Weakness {
	var weaknesses []core.Weakness
	byDifficulty := make(map[int][]ErrorRecord)

	for _, rec := range d.errorLog {
		level := int(rec.Difficulty * 3) // 0=Easy, 1=Medium, 2=Hard
		byDifficulty[level] = append(byDifficulty[level], rec)
	}

	// Detect sudden jumps in error rate
	for level := 1; level < 3; level++ {
		lower := byDifficulty[level-1]
		upper := byDifficulty[level]
		if len(lower) < 3 || len(upper) < 3 {
			continue
		}

		lowerErrors := 0
		for _, r := range lower {
			if !r.Correct {
				lowerErrors++
			}
		}
		upperErrors := 0
		for _, r := range upper {
			if !r.Correct {
				upperErrors++
			}
		}

		lowerRate := float64(lowerErrors) / float64(len(lower))
		upperRate := float64(upperErrors) / float64(len(upper))

		if upperRate-lowerRate > 0.3 { // Sudden 30% jump in error rate
			diffName := "easy"
			if level == 1 {
				diffName = "easy-to-medium"
			} else {
				diffName = "medium-to-hard"
			}
			weaknesses = append(weaknesses, core.Weakness{
				ID:             fmt.Sprintf("gap_%s", diffName),
				Description:    fmt.Sprintf("Performance gap at %s transition", diffName),
				TaskType:       "difficulty_gap",
				Difficulty:     float64(level) / 2.0,
				Frequency:      len(upper),
				SuggestedFocus: "Bridge the difficulty gap with intermediate problems",
			})
		}
	}

	return weaknesses
}

// AnalyzePerformanceTrend computes whether performance is improving or degrading
func (d *Detector) AnalyzePerformanceTrend(windowSize int) (improving bool, trend float64) {
	if len(d.errorLog) < windowSize {
		return true, 0.0
	}

	recent := d.errorLog[len(d.errorLog)-windowSize:]
	older := d.errorLog[len(d.errorLog)-2*windowSize : len(d.errorLog)-windowSize]

	olderErrors := 0
	for _, r := range older {
		if !r.Correct {
			olderErrors++
		}
	}
	recentErrors := 0
	for _, r := range recent {
		if !r.Correct {
			recentErrors++
		}
	}

	olderRate := float64(olderErrors) / float64(len(older))
	recentRate := float64(recentErrors) / float64(len(recent))

	trend = recentRate - olderRate
	improving = recentRate < olderRate
	return
}
