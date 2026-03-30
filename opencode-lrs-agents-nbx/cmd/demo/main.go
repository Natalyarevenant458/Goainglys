package main

import (
	"fmt"
	"math/rand"

	lrs "github.com/goainglys/lrs-agents-nbx"
)

func main() {
	fmt.Println("=== LRS Agents NBX Demo ===")
	fmt.Println()

	// Seed random
	rand.Seed(42) //nolint:staticcheck // for reproducible demo

	// ============================================================
	// Demo 1: Basic Scheduler Usage
	// ============================================================
	fmt.Println("--- Demo 1: Learning Rate Schedulers ---")
	fmt.Println()

	// Create different schedulers
	schedulers := map[string]lrs.LRScheduler{
		"OneCycleLR": lrs.NewOneCycleLR(0.001,
			lrs.WithMaxLR(0.01),
			lrs.WithTotalSteps(1000),
			lrs.WithDivFactor(25),
		),
		"CyclicLR": lrs.NewCyclicLR(0.001,
			lrs.WithMaxLR(0.01),
			lrs.WithStepSize(200),
		),
		"CosineAnnealingWR": lrs.NewCosineAnnealingWarmRestarts(0.001,
			lrs.WithT0(200),
			lrs.WithTMult(2.0),
		),
		"PolynomialLR": lrs.NewPolynomialLR(0.001,
			lrs.WithWarmupSteps(100),
			lrs.WithTotalSteps(1000),
			lrs.WithPower(1.0),
		),
		"ExponentialWarmup": lrs.NewExponentialWarmup(0.001,
			lrs.WithWarmupSteps(100),
			lrs.WithTotalSteps(1000),
		),
	}

	// Test each scheduler
	for name, sched := range schedulers {
		fmt.Printf("Testing %s:\n", name)
		lr := sched.GetLR()
		fmt.Printf("  Step 0: LR = %.8f\n", lr)

		// Step a few times
		for i := 1; i <= 10; i++ {
			lr = sched.Step()
		}
		fmt.Printf("  Step 10: LR = %.8f\n", lr)

		// Reset and show full trajectory
		sched.Reset()
		fmt.Printf("  Full trajectory (first 10 steps): ")
		for i := 0; i < 10; i++ {
			lr = sched.Step()
			fmt.Printf("%.6f ", lr)
		}
		fmt.Println()
		sched.Reset()
		fmt.Println()
	}

	// ============================================================
	// Demo 2: LRS-Aware Optimizer
	// ============================================================
	fmt.Println("--- Demo 2: LRS-Aware Optimizer ---")
	fmt.Println()

	// Create model parameters
	weights1 := make([]float64, 10)
	weights2 := make([]float64, 20)
	for i := range weights1 {
		weights1[i] = rand.NormFloat64() * 0.1
	}
	for i := range weights2 {
		weights2[i] = rand.NormFloat64() * 0.1
	}

	// Create optimizer
	baseOptimizer := lrs.NewAdamScale(0.001, 0.9, 0.999, 1e-8, 0.01)
	baseOptimizer.AddParams([][]float64{weights1, weights2})

	// Create scheduler
	scheduler := lrs.NewOneCycleLR(0.001,
		lrs.WithMaxLR(0.01),
		lrs.WithTotalSteps(500),
	)

	// Wrap with LRS optimizer
	lrsOptimizer := lrs.NewLRSOptimizer(baseOptimizer, scheduler)

	// Simulate training
	fmt.Println("Training with LRS-aware optimizer:")
	for step := 0; step < 20; step++ {
		// Step
		lr := lrsOptimizer.Step()

		if step%5 == 0 {
			fmt.Printf("  Step %d: LR = %.8f\n", step, lr)
		}
	}

	// Show LR history
	history := lrsOptimizer.GetHistoryLR()
	fmt.Printf("\nLR History (last 10): ")
	start := len(history) - 10
	if start < 0 {
		start = 0
	}
	for i := start; i < len(history); i++ {
		fmt.Printf("%.6f ", history[i])
	}
	fmt.Println("\n")

	// ============================================================
	// Demo 3: LRSAwareTrainer
	// ============================================================
	fmt.Println("--- Demo 3: LRSAwareTrainer ---")
	fmt.Println()

	// Create a dummy model
	model := lrs.NewDummyModel(64)

	// Create optimizer
	opt := lrs.NewAdamScale(0.001, 0.9, 0.999, 1e-8, 0.0)
	opt.AddParams(model.GetParameters())

	// Create scheduler
	sched := lrs.NewCosineAnnealingWarmRestarts(0.001,
		lrs.WithT0(100),
		lrs.WithTMult(1.5),
	)

	// Create trainer config
	config := &lrs.TrainerConfig{
		MaxSteps:      100,
		PrintInterval: 20,
		TargetLoss:    0.001,
	}

	// Create trainer
	trainer := lrs.NewLRSAwareTrainer(model, opt, sched, config)

	// Train
	fmt.Println("Training with LRSAwareTrainer:")
	trainer.Train(100)

	// Get metrics
	metrics := trainer.GetMetrics()
	fmt.Printf("\nFinal Metrics:\n")
	fmt.Printf("  Loss variance: %.6f\n", metrics.GetLossVariance())
	fmt.Printf("  Gradient efficiency: %.6f\n", metrics.GetGradientEfficiency())
	fmt.Printf("  LR effectiveness: %.6f\n", metrics.GetLREffectiveness())
	fmt.Println()

	// ============================================================
	// Demo 4: Adaptive Scheduler Agent
	// ============================================================
	fmt.Println("--- Demo 4: Adaptive Scheduler Agent ---")
	fmt.Println()

	// Create model
	model2 := lrs.NewDummyModel(64)

	// Create candidates
	candidates := []*lrs.SchedulerCandidate{
		{
			Name:      "OneCycle",
			Scheduler: lrs.NewOneCycleLR(0.001, lrs.WithMaxLR(0.01), lrs.WithTotalSteps(200)),
		},
		{
			Name:      "Cyclic",
			Scheduler: lrs.NewCyclicLR(0.001, lrs.WithMaxLR(0.01), lrs.WithStepSize(50)),
		},
		{
			Name:      "Polynomial",
			Scheduler: lrs.NewPolynomialLR(0.001, lrs.WithWarmupSteps(20), lrs.WithTotalSteps(200)),
		},
	}

	// Create optimizer
	opt2 := lrs.NewAdamScale(0.001, 0.9, 0.999, 1e-8, 0.0)
	opt2.AddParams(model2.GetParameters())

	// Create adaptive config
	adaptiveConfig := &lrs.AdaptiveConfig{
		MaxSteps:             200,
		EvalInterval:         20,
		SwitchThreshold:      0.5,
		ImprovementThreshold: 1.0,
	}

	// Create adaptive agent
	adaptiveAgent := lrs.NewAdaptiveSchedulerAgent(model2, opt2, candidates, adaptiveConfig)

	// Run training episode
	fmt.Println("Running Adaptive Scheduler Agent:")
	adaptiveAgent.RunEpisode(200)

	fmt.Printf("\nFinal scheduler: %s\n", adaptiveAgent.GetCurrentSchedulerName())
	fmt.Println()

	// ============================================================
	// Demo 5: NBX Protocol
	// ============================================================
	fmt.Println("--- Demo 5: Neural Block Exchange (NBX) ---")
	fmt.Println()

	// Create blocks
	block1 := lrs.NewBlock("block_1", "embedding", []float64{0.1, 0.2, 0.3, 0.4, 0.5})
	block2 := lrs.NewBlock("block_2", "attention", []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8})
	block3 := lrs.NewBlock("block_3", "ffn", []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})

	// Set gradients (simulated)
	block1.Gradients = []float64{0.01, 0.02, -0.01, 0.03, -0.02}
	block2.Gradients = []float64{0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02}
	block3.Gradients = []float64{0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.003, -0.001}

	// Compute gradient norms
	norm1 := block1.ComputeGradientNorm()
	norm2 := block2.ComputeGradientNorm()
	norm3 := block3.ComputeGradientNorm()

	fmt.Println("Block Gradient Norms:")
	fmt.Printf("  %s: %.6f\n", block1.ID, norm1)
	fmt.Printf("  %s: %.6f\n", block2.ID, norm2)
	fmt.Printf("  %s: %.6f\n", block3.ID, norm3)
	fmt.Println()

	// Create block exchange
	exchange := lrs.NewBlockExchange(0, 4, lrs.Ring)
	exchange.RegisterBlock(block1)
	exchange.RegisterBlock(block2)
	exchange.RegisterBlock(block3)

	// Test block selector
	selector := lrs.NewBlockSelector()
	blocks := exchange.GetAllBlocks()
	selected := selector.SelectTopK(blocks, 2)

	fmt.Println("Selected blocks for exchange (top 2 by gradient norm):")
	for _, b := range selected {
		fmt.Printf("  %s (grad norm: %.6f)\n", b.ID, b.Metadata.GradNorm)
	}
	fmt.Println()

	// Test exchange scheduler
	schedulerNBX := lrs.NewExchangeScheduler(lrs.Ring, 4, 0)
	partners := schedulerNBX.GetExchangePartners()
	fmt.Printf("Ring topology partners for worker 0: %v\n", partners)

	schedulerNBX.Step()
	partners = schedulerNBX.GetExchangePartners()
	fmt.Printf("After one step: %v\n", partners)
	fmt.Println()

	// Test collaborative optimizer with trust
	trustMerger := lrs.NewTrustBasedMerger()
	collaborativeOpt := lrs.NewCollaborativeOptimizer(trustMerger)

	// Simulate receiving gradients from workers
	remoteGrad1 := []float64{0.015, 0.025, -0.015, 0.035, -0.025}
	remoteGrad2 := []float64{0.012, -0.018, 0.028, -0.012, 0.022}

	trust1 := lrs.ComputeTrustScore(block1.Gradients, remoteGrad1)
	trust2 := lrs.ComputeTrustScore(block1.Gradients, remoteGrad2)

	fmt.Println("Trust scores:")
	fmt.Printf("  Worker 1: %.4f\n", trust1)
	fmt.Printf("  Worker 2: %.4f\n", trust2)

	collaborativeOpt.UpdateTrustScore(1, trust1)
	collaborativeOpt.UpdateTrustScore(2, trust2)

	// Merge simulated gradients
	grads := [][]float64{block1.Gradients, remoteGrad1, remoteGrad2}
	weights := []float64{1.0, trust1, trust2}

	mergedGrad := trustMerger.MergedGradients(grads, weights)
	fmt.Println("\nMerged gradients (weighted by trust):")
	fmt.Printf("  Local:     %v\n", block1.Gradients)
	fmt.Printf("  Remote 1: %v\n", remoteGrad1)
	fmt.Printf("  Remote 2: %v\n", remoteGrad2)
	fmt.Printf("  Merged:   %v\n", mergedGrad)
	fmt.Println()

	// ============================================================
	// Demo 6: Full NBX Training Agent
	// ============================================================
	fmt.Println("--- Demo 6: NBX Training Agent ---")
	fmt.Println()

	// Create a model for NBX
	modelNBX := lrs.NewDummyModel(32)

	// Create NBX agent
	nbxAgent := lrs.NewNBXAgent(0, 4, lrs.Mesh, modelNBX)

	// Simulate a few training steps
	fmt.Println("Running NBX training steps:")
	for step := 0; step < 5; step++ {
		// Forward pass (simulated)
		input := make([]float64, 32)
		for i := range input {
			input[i] = rand.Float64()*2 - 1
		}

		// Forward
		output := modelNBX.Forward(input)

		// Compute loss
		target := make([]float64, len(output))
		loss := 0.0
		for i := range output {
			target[i] = 0.5
			diff := output[i] - target[i]
			loss += diff * diff
		}
		loss /= float64(len(output))

		// Backward (simulated)
		gradOut := make([]float64, len(output))
		for i := range output {
			gradOut[i] = 2 * (output[i] - target[i]) / float64(len(output))
		}
		modelNBX.Backward(gradOut)

		// Update with NBX
		updateFunc := func(mergedGrad []float64) {
			// In real implementation, would apply merged gradients
			_ = mergedGrad
		}

		_ = nbxAgent.RunTrainingStep(
			func() float64 { return loss },
			func() { modelNBX.Backward(gradOut) },
			updateFunc,
		)

		fmt.Printf("  Step %d: Loss = %.6f\n", step, loss)
	}

	fmt.Println()
	fmt.Println("=== Demo Complete ===")
}
