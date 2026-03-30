package data_generator

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"self_improving_agents/core"
	"self_improving_agents/integrations/transformers"
)

type Generator struct {
	transformerAdapter transformers.Adapter
	rng                *rand.Rand
}

func NewGenerator(transformerAdapter transformers.Adapter) *Generator {
	return &Generator{
		transformerAdapter: transformerAdapter,
		rng:                rand.New(rand.NewSource(42)),
	}
}

func (g *Generator) Generate(weaknesses []core.Weakness, examplesPerWeakness int, difficultyProgression float64) ([]string, error) {
	var trainingData []string

	for _, weakness := range weaknesses {
		difficulty := weakness.Difficulty
		for i := 0; i < examplesPerWeakness; i++ {
			adjustedDifficulty := math.Min(1.0, difficulty+float64(i)*difficultyProgression*0.01)
			example := g.generateForTaskType(weakness.TaskType, adjustedDifficulty)
			trainingData = append(trainingData, example)
		}
	}

	return trainingData, nil
}

func (g *Generator) generateForTaskType(taskType string, difficulty float64) string {
	switch {
	case strings.Contains(taskType, "arithmetic"):
		return g.generateArithmetic(difficulty)
	case strings.Contains(taskType, "algebra"):
		return g.generateAlgebra(difficulty)
	case strings.Contains(taskType, "reasoning"):
		return g.generateReasoning(difficulty)
	case strings.Contains(taskType, "word_problem"):
		return g.generateWordProblem(difficulty)
	case strings.Contains(taskType, "fraction"):
		return g.generateFraction(difficulty)
	case strings.Contains(taskType, "geometry"):
		return g.generateGeometry(difficulty)
	default:
		return g.generateArithmetic(difficulty)
	}
}

func (g *Generator) generateArithmetic(difficulty float64) string {
	ops := []string{"+", "-", "*", "/"}
	op := ops[g.rng.Intn(len(ops))]

	switch {
	case difficulty < 0.3:
		a := g.rng.Intn(20) + 1
		b := g.rng.Intn(20) + 1
		if op == "-" {
			if a < b {
				a, b = b, a
			}
		}
		if op == "/" {
			b = g.rng.Intn(10) + 1
			a = b * (g.rng.Intn(10) + 1)
		}
		return fmt.Sprintf("Calculate: %d %s %d", a, op, b)

	case difficulty < 0.6:
		a := g.rng.Intn(100) + 10
		b := g.rng.Intn(50) + 5
		c := g.rng.Intn(10) + 1
		if op == "-" {
			if a < b {
				a, b = b, a
			}
		}
		if op == "/" {
			b = g.rng.Intn(10) + 1
			a = b * (g.rng.Intn(10) + 1)
		}
		op2 := ops[g.rng.Intn(len(ops))]
		return fmt.Sprintf("Calculate: %d %s %d %s %d", a, op, b, op2, c)

	default:
		a := g.rng.Intn(1000) + 100
		b := g.rng.Intn(100) + 10
		percentage := (g.rng.Intn(30) + 5)
		return fmt.Sprintf("What is %d%% of %d? (Then add %d)", percentage, a, b)
	}
}

func (g *Generator) generateAlgebra(difficulty float64) string {
	switch {
	case difficulty < 0.3:
		x := g.rng.Intn(10) + 1
		a := g.rng.Intn(5) + 1
		b := a*x + g.rng.Intn(20)
		return fmt.Sprintf("Solve for x: %dx + %d = %d", a, b-a*x, b)

	case difficulty < 0.6:
		x := g.rng.Intn(10) + 1
		a := g.rng.Intn(3) + 2
		b := g.rng.Intn(10) + 1
		c := a*x + b
		return fmt.Sprintf("Solve for x: %dx + %d = %d, then find 2x + 5", a, b, c)

	default:
		x := g.rng.Intn(10) + 1
		y := g.rng.Intn(10) + 1
		a := g.rng.Intn(3) + 2
		b := g.rng.Intn(3) + 2
		return fmt.Sprintf("If x = %d and y = %d, evaluate: %dx + %dy - %d", x, y, a, b, a*x+b*y)
	}
}

func (g *Generator) generateReasoning(difficulty float64) string {
	switch {
	case difficulty < 0.3:
		pattern := g.rng.Intn(3)
		switch pattern {
		case 0:
			n := g.rng.Intn(10) + 2
			return fmt.Sprintf("Find the next number in the pattern: 2, 4, 6, 8, %d, ?", n*2+2)
		case 1:
			return "If all cats are animals, and all animals need water, do cats need water?"
		default:
			return "Which is larger: 3/7 or 4/9?"
		}

	case difficulty < 0.6:
		patterns := []string{
			"Find the next number: 1, 1, 2, 3, 5, 8, ?, ?",
			"If a train leaves at 9am traveling 60mph and another leaves at 11am traveling 80mph, when do they meet?",
			"What comes next: 1, 4, 9, 16, ?",
		}
		return patterns[g.rng.Intn(len(patterns))]

	default:
		problems := []string{
			"A tank fills in 4 hours and drains in 6 hours. How long to fill with drain open?",
			"Three people can paint 3 walls in 3 hours. How many walls can 6 people paint in 6 hours?",
			"If you have a 3-gallon and a 5-gallon jug, how do you measure exactly 4 gallons?",
		}
		return problems[g.rng.Intn(len(problems))]
	}
}

func (g *Generator) generateWordProblem(difficulty float64) string {
	switch {
	case difficulty < 0.3:
		distance := g.rng.Intn(50) + 20
		speed := g.rng.Intn(20) + 10
		hours := distance / speed
		return fmt.Sprintf("A car travels at %d mph for %d hours. How far does it go?", speed, hours)

	case difficulty < 0.6:
		price := g.rng.Intn(50) + 10
		quantity := g.rng.Intn(10) + 2
		discount := g.rng.Intn(20) + 5
		return fmt.Sprintf("A book costs $%d. You buy %d books with a %d%% discount. What is the total?", price, quantity, discount)

	default:
		problems := []string{
			"John has 3x coins, gives 5 away, and has 16 left. Find x.",
			"A rectangle has area 72 and width 8. Find its perimeter.",
			"Two numbers differ by 5, and their product is 84. Find both numbers.",
		}
		return problems[g.rng.Intn(len(problems))]
	}
}

func (g *Generator) generateFraction(difficulty float64) string {
	switch {
	case difficulty < 0.3:
		a := g.rng.Intn(9) + 1
		b := g.rng.Intn(9) + 1
		c := g.rng.Intn(9) + 1
		d := g.rng.Intn(9) + 1
		if a == b {
			a++
		}
		if c == d {
			c++
		}
		return fmt.Sprintf("Add: %d/%d + %d/%d (simplify your answer)", a, b, c, d)

	case difficulty < 0.6:
		ops := []string{"add", "subtract", "multiply"}
		op := ops[g.rng.Intn(len(ops))]
		a := g.rng.Intn(9) + 1
		b := g.rng.Intn(9) + 1
		c := g.rng.Intn(9) + 1
		d := g.rng.Intn(9) + 1
		if b == 0 {
			b = 1
		}
		if d == 0 {
			d = 1
		}
		return fmt.Sprintf("%s: %d/%d %s %d/%d", op, a, b, op, c, d)

	default:
		problems := []string{
			"Simplify: (2/3) / (4/5)",
			"What is 2/3 of 3/4? Express as a simplified fraction.",
			"Compare: 5/8 and 2/3 (which is larger?)",
		}
		return problems[g.rng.Intn(len(problems))]
	}
}

func (g *Generator) generateGeometry(difficulty float64) string {
	switch {
	case difficulty < 0.3:
		base := g.rng.Intn(10) + 3
		height := g.rng.Intn(10) + 3
		return fmt.Sprintf("Find the area of a triangle with base %d and height %d", base, height)

	case difficulty < 0.6:
		radius := g.rng.Intn(10) + 2
		return fmt.Sprintf("A circle has radius %d. Find its circumference and area", radius)

	default:
		problems := []string{
			"Find the hypotenuse of a right triangle with legs 5 and 12",
			"A cylinder has radius 3 and height 7. Find its volume",
			"Find the surface area of a cube with side length 5",
		}
		return problems[g.rng.Intn(len(problems))]
	}
}
