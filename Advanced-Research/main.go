package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	fmt.Println("=== Goainglys ML Platform - Advanced Research ===")
	fmt.Println()

	// Set random seed for reproducibility
	rand.Seed(42)

	// ============================================================
	// 1. Mixture of Experts Demo
	// ============================================================
	fmt.Println("--- Mixture of Experts (MoE) ---")

	// Configure MoE
	moeConfig := MoEConfig{
		NumExperts: 4,
		TopK:       2,
		ExpertDim:  64,
		Capacity:   1.5,
	}

	// Create MoE layer
	inputDim := 16
	outputDim := 8
	moeLayer := NewMoELayer(inputDim, outputDim, moeConfig)

	// Create sample input (batch of 4 samples)
	batchSize := 4
	x := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		x[i] = make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			x[i][j] = rand.NormFloat64()
		}
	}

	// Forward pass
	output := moeLayer.Forward(x)

	// Print results
	fmt.Printf("Input shape: %d x %d\n", batchSize, inputDim)
	fmt.Printf("Output shape: %d x %d\n", len(output), len(output[0]))
	fmt.Printf("Sample output (first row): ")
	for j := 0; j < min(4, len(output[0])); j++ {
		fmt.Printf("%.4f ", output[0][j])
	}
	fmt.Println("...")

	// Get routing statistics
	stats := moeLayer.GetMoEStats()
	fmt.Printf("Expert load stats: avg=%.2f, max=%.2f, min=%.2f\n",
		stats["avg_load"], stats["max_load"], stats["min_load"])

	// Train step (simplified)
	gradOutput := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		gradOutput[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			gradOutput[i][j] = rand.NormFloat64() * 0.1
		}
	}
	moeLayer.Backward(x, gradOutput, 0.01)
	fmt.Println("Training step completed.")

	// Training loop
	fmt.Println("Running training loop...")
	for epoch := 0; epoch < 3; epoch++ {
		// Fresh forward pass
		x := make([][]float64, batchSize)
		for i := 0; i < batchSize; i++ {
			x[i] = make([]float64, inputDim)
			for j := 0; j < inputDim; j++ {
				x[i][j] = rand.NormFloat64()
			}
		}

		output := moeLayer.Forward(x)

		// Compute loss (target is random)
		target := make([][]float64, batchSize)
		loss := 0.0
		for i := 0; i < batchSize; i++ {
			target[i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				target[i][j] = rand.NormFloat64()
				diff := output[i][j] - target[i][j]
				loss += diff * diff
			}
		}
		loss /= float64(batchSize * outputDim)

		// Backward
		gradOutput := make([][]float64, batchSize)
		for i := 0; i < batchSize; i++ {
			gradOutput[i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				gradOutput[i][j] = 2 * (output[i][j] - target[i][j]) / float64(outputDim)
			}
		}
		moeLayer.Backward(x, gradOutput, 0.01)

		fmt.Printf("  Epoch %d: loss = %.6f\n", epoch+1, loss)
	}

	fmt.Println()

	// ============================================================
	// 2. Diffusion Model Demo
	// ============================================================
	fmt.Println("--- Diffusion Model (DDPM) ---")

	// Configure DDPM
	ddpmConfig := DDPMConfig{
		Timesteps: 100,
		BetaStart: 0.0001,
		BetaEnd:   0.02,
		ModelDim:  16,
	}

	// Create diffusion model
	diffusion := NewDDPM(ddpmConfig)

	// Create clean data
	cleanData := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		cleanData[i] = make([]float64, ddpmConfig.ModelDim)
		for j := 0; j < ddpmConfig.ModelDim; j++ {
			cleanData[i][j] = math.Sin(float64(i) * 0.5 * float64(j+1))
		}
	}

	// Forward diffusion (add noise)
	t := 50 // midpoint timestep
	noisyData, trueNoise := diffusion.ForwardDiffusion(cleanData, t)

	fmt.Printf("Clean data range: [%.4f, %.4f]\n",
		minFloatMatrix(cleanData), maxFloatMatrix(cleanData))
	fmt.Printf("Noisy data (t=%d) range: [%.4f, %.4f]\n", t,
		minFloatMatrix(noisyData), maxFloatMatrix(noisyData))
	fmt.Printf("Noise range: [%.4f, %.4f]\n",
		minFloatMatrix(trueNoise), maxFloatMatrix(trueNoise))

	// Training step
	fmt.Println("Running training steps...")
	for epoch := 0; epoch < 3; epoch++ {
		// Random timestep
		t := rand.Intn(ddpmConfig.Timesteps - 1)

		loss := diffusion.TrainStep(cleanData, t, 0.01)
		fmt.Printf("  Epoch %d (t=%d): loss = %.6f\n", epoch+1, t, loss)
	}

	// Sampling
	fmt.Println("Generating samples...")
	samples := diffusion.Sample(2)
	fmt.Printf("Generated sample shape: %d x %d\n", len(samples), len(samples[0]))
	fmt.Printf("Sample values: ")
	for j := 0; j < min(4, len(samples[0])); j++ {
		fmt.Printf("%.4f ", samples[0][j])
	}
	fmt.Println("...")

	// DDIM sampling
	fmt.Println("DDIM sampling (faster)...")
	ddimSamples := diffusion.DDIMSampler(2, 10)
	fmt.Printf("DDIM samples: ")
	for j := 0; j < min(4, len(ddimSamples[0])); j++ {
		fmt.Printf("%.4f ", ddimSamples[0][j])
	}
	fmt.Println("...")

	fmt.Println()

	// ============================================================
	// 3. Reinforcement Learning Demo
	// ============================================================
	fmt.Println("--- Reinforcement Learning (PPO) ---")

	// Configure PPO
	stateDim := 8
	numActions := 4
	gamma := 0.99
	clipEpsilon := 0.2
	lr := 0.001

	// Create PPO agent
	ppo := NewPPO(stateDim, numActions, lr, gamma, clipEpsilon)

	// Create replay buffer
	replayBuffer := NewReplayBuffer(1000, 0.6, 0.4)

	// Generate sample trajectories
	fmt.Println("Collecting experience...")
	for episode := 0; episode < 5; episode++ {
		state := make([]float64, stateDim)
		for i := range state {
			state[i] = rand.Float64()*2 - 1 // Random state in [-1, 1]
		}

		// Episode steps
		steps := 10
		for step := 0; step < steps; step++ {
			// Select action
			action := ppo.SelectActionPPO(state)

			// Environment interaction (simulated)
			nextState := make([]float64, stateDim)
			for i := range nextState {
				nextState[i] = state[i] + rand.NormFloat64()*0.1
				if nextState[i] > 1 {
					nextState[i] = 1
				}
				if nextState[i] < -1 {
					nextState[i] = -1
				}
			}

			// Reward (simple: higher state values = better)
			reward := 0.0
			for _, v := range nextState {
				reward += v
			}
			reward /= float64(stateDim)

			done := step == steps-1

			// Add to replay buffer
			replayBuffer.Add(state, nextState, action, reward, done)

			state = nextState
		}
	}

	fmt.Printf("Replay buffer size: %d\n", replayBuffer.Size())

	// Get action probabilities for a sample state
	sampleState := make([]float64, stateDim)
	for i := range sampleState {
		sampleState[i] = rand.Float64()
	}
	probs := ppo.GetActionProbabilities(sampleState)
	fmt.Printf("Sample action probabilities: ")
	for i, p := range probs {
		fmt.Printf("a%d=%.3f ", i, p)
	}
	fmt.Println()

	// Sample from buffer and train
	_, _, _, _, _, _, indices := replayBuffer.Sample(min(4, replayBuffer.Size()))
	if indices != nil {
		fmt.Printf("Sampled batch size: %d\n", len(indices))

		// Update priorities
		tdErrors := make([]float64, len(indices))
		for i := range tdErrors {
			tdErrors[i] = rand.Float64()
		}
		replayBuffer.UpdatePriorities(indices, tdErrors)
	}

	// PPO training
	fmt.Println("Running PPO training...")
	for epoch := 0; epoch < 3; epoch++ {
		// Create sample trajectory with Done flag
		trajectory := make([]TrajectoryWithDone, 5)

		for i := range trajectory {
			state := make([]float64, stateDim)
			for j := range state {
				state[j] = rand.Float64()*2 - 1
			}
			trajectory[i].State = state
			trajectory[i].Action = rand.Intn(numActions)
			trajectory[i].Reward = rand.Float64()
			trajectory[i].Done = i == len(trajectory)-1
		}

		loss := ppo.TrainPPO(trajectory)
		fmt.Printf("  Epoch %d: loss = %.6f\n", epoch+1, loss)
	}

	// Policy gradient training
	fmt.Println("Running REINFORCE training...")
	pg := NewPolicyGradient(stateDim, numActions, lr, gamma)
	for epoch := 0; epoch < 3; epoch++ {
		// Create sample trajectory
		traj := make([]TrajectoryStep, 5)

		for i := range traj {
			state := make([]float64, stateDim)
			for j := range state {
				state[j] = rand.Float64()*2 - 1
			}
			traj[i].State = state
			traj[i].Action = rand.Intn(numActions)
			traj[i].Reward = rand.Float64()
		}

		loss := pg.Train(traj)
		fmt.Printf("  Epoch %d: loss = %.6f\n", epoch+1, loss)
	}

	fmt.Println()

	// ============================================================
	// 4. Graph Neural Network Demo
	// ============================================================
	fmt.Println("--- Graph Neural Network (GNN) ---")

	// Create sample graphs
	graph1 := CreateSampleGraph()
	graph2 := CreateSampleGraphWithCycle()

	fmt.Printf("Graph 1: %d nodes, %d edges\n", graph1.NumNodes, graph1.NumEdges)
	fmt.Printf("Graph 2: %d nodes, %d edges\n", graph2.NumNodes, graph2.NumEdges)

	// Create GNN
	gnnInputDim := 8
	gnnHiddenDim := 16
	gnnOutputDim := 4
	numHops := 3

	gnn := NewGraphNetwork(gnnInputDim, gnnHiddenDim, gnnOutputDim, numHops)

	// Graph-level prediction
	graphEmbedding := gnn.Forward(graph1)
	fmt.Printf("Graph embedding shape: %d\n", len(graphEmbedding))
	fmt.Printf("Graph embedding: ")
	for i := 0; i < min(4, len(graphEmbedding)); i++ {
		fmt.Printf("%.4f ", graphEmbedding[i])
	}
	fmt.Println("...")

	// Node-level prediction
	nodePredictions := gnn.ForwardNodeClassification(graph2)
	fmt.Printf("Node predictions shape: %d x %d\n", len(nodePredictions), len(nodePredictions[0]))
	fmt.Printf("Node 0 prediction: ")
	for i := 0; i < min(4, len(nodePredictions[0])); i++ {
		fmt.Printf("%.4f ", nodePredictions[0][i])
	}
	fmt.Println("...")

	// GraphSAGE convolution demo
	fmt.Println("Testing GraphSAGE convolution...")
	sageConv := NewGraphConv(gnnInputDim, gnnHiddenDim, relu)
	sageOutput := sageConv.Forward(graph1)
	fmt.Printf("GraphSAGE output: %d x %d\n", len(sageOutput), len(sageOutput[0]))

	// Graph Attention (GAT) demo
	fmt.Println("Testing Graph Attention (GAT)...")
	gat := NewGraphAttention(gnnInputDim, gnnHiddenDim, 2, relu)
	gatOutput := gat.Forward(graph1)
	fmt.Printf("GAT output: %d x %d\n", len(gatOutput), len(gatOutput[0]))

	// Readout pooling demo
	fmt.Println("Testing readout pooling...")
	readoutMean := NewReadoutPooling("mean")
	readoutSum := NewReadoutPooling("sum")
	readoutMax := NewReadoutPooling("max")

	nodeEmbeddings := make([][]float64, graph1.NumNodes)
	for i := range nodeEmbeddings {
		nodeEmbeddings[i] = make([]float64, gnnHiddenDim)
		for j := range nodeEmbeddings[i] {
			nodeEmbeddings[i][j] = rand.Float64()
		}
	}

	meanPooled := readoutMean.Forward(nodeEmbeddings)
	sumPooled := readoutSum.Forward(nodeEmbeddings)
	maxPooled := readoutMax.Forward(nodeEmbeddings)

	_ = meanPooled
	_ = sumPooled
	_ = maxPooled
	fmt.Printf("Mean/Sum/Max pooling: %d dims each\n", len(meanPooled))

	// Compute losses
	fmt.Println("Computing losses...")
	labels := []int{0, 1, 2, 1, 0}
	nodeLoss := gnn.ComputeNodeLoss(graph1, labels)
	graphLoss := gnn.ComputeGraphLoss(graph2, 1)
	fmt.Printf("Node classification loss: %.6f\n", nodeLoss)
	fmt.Printf("Graph classification loss: %.6f\n", graphLoss)

	// Message passing demo
	fmt.Println("Testing multi-hop message passing...")
	mp := NewMessagePassing(gnnInputDim, gnnHiddenDim, numHops)
	mpOutput := mp.Forward(graph1)
	fmt.Printf("Message passing output: %d x %d\n", len(mpOutput), len(mpOutput[0]))

	_ = sageOutput
	_ = gatOutput
	_ = mpOutput

	fmt.Println()

	// ============================================================
	// Summary
	// ============================================================
	fmt.Println("=== All Advanced Research Modules Working ===")
	fmt.Println()
	fmt.Println("Summary:")
	fmt.Println("  1. Mixture of Experts - Routing, expert selection, parallel processing")
	fmt.Println("  2. Diffusion Model - DDPM forward/noise, reverse denoising, DDIM sampling")
	fmt.Println("  3. Reinforcement Learning - PPO, REINFORCE, prioritized replay buffer")
	fmt.Println("  4. Graph Neural Network - GraphSAGE, GAT, message passing, readout pooling")
	fmt.Println()
	fmt.Println("All implementations use pure Go float64 with no external dependencies.")
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func minFloatMatrix(m [][]float64) float64 {
	if len(m) == 0 || len(m[0]) == 0 {
		return 0
	}
	minVal := m[0][0]
	for i := range m {
		for j := range m[i] {
			if m[i][j] < minVal {
				minVal = m[i][j]
			}
		}
	}
	return minVal
}

func maxFloatMatrix(m [][]float64) float64 {
	if len(m) == 0 || len(m[0]) == 0 {
		return 0
	}
	maxVal := m[0][0]
	for i := range m {
		for j := range m[i] {
			if m[i][j] > maxVal {
				maxVal = m[i][j]
			}
		}
	}
	return maxVal
}
