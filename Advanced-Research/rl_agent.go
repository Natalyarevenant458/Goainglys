package main

import (
	"math"
	"math/rand"
)

// ReplayBuffer stores experience tuples for RL
type ReplayBuffer struct {
	states     [][]float64
	actions    []int
	rewards    []float64
	nextStates [][]float64
	dones      []bool
	priorities []float64
	capacity   int
	size       int
	alpha      float64 // Priority exponent
	beta       float64 // Importance sampling exponent
}

// NewReplayBuffer creates a new prioritized replay buffer
func NewReplayBuffer(capacity int, alpha, beta float64) *ReplayBuffer {
	return &ReplayBuffer{
		states:     make([][]float64, 0, capacity),
		actions:    make([]int, 0, capacity),
		rewards:    make([]float64, 0, capacity),
		nextStates: make([][]float64, 0, capacity),
		dones:      make([]bool, 0, capacity),
		priorities: make([]float64, 0, capacity),
		capacity:   capacity,
		alpha:      alpha,
		beta:       beta,
	}
}

// Add adds an experience to the buffer
func (rb *ReplayBuffer) Add(state, nextState []float64, action int, reward float64, done bool) {
	if rb.capacity > 0 && rb.size >= rb.capacity {
		// Remove oldest entry
		rb.states = rb.states[1:]
		rb.actions = rb.actions[1:]
		rb.rewards = rb.rewards[1:]
		rb.nextStates = rb.nextStates[1:]
		rb.dones = rb.dones[1:]
		rb.priorities = rb.priorities[1:]
		rb.size--
	}

	// Compute priority (max priority if buffer not empty, else 1.0)
	maxPriority := 1.0
	if len(rb.priorities) > 0 {
		for _, p := range rb.priorities {
			if p > maxPriority {
				maxPriority = p
			}
		}
	}

	// Add new experience
	rb.states = append(rb.states, copySlice(state))
	rb.actions = append(rb.actions, action)
	rb.rewards = append(rb.rewards, reward)
	rb.nextStates = append(rb.nextStates, copySlice(nextState))
	rb.dones = append(rb.dones, done)
	rb.priorities = append(rb.priorities, math.Pow(maxPriority, rb.alpha))
	rb.size++
}

// Sample samples a batch of experiences with prioritized sampling
func (rb *ReplayBuffer) Sample(batchSize int) (states, nextStates [][]float64, actions []int, rewards []float64, dones []bool, weights []float64, indices []int) {
	if rb.size == 0 {
		return nil, nil, nil, nil, nil, nil, nil
	}

	// Compute sampling probabilities
	sumPriority := 0.0
	for _, p := range rb.priorities {
		sumPriority += p
	}

	probs := make([]float64, rb.size)
	for i, p := range rb.priorities {
		probs[i] = p / sumPriority
	}

	// Sample indices based on priorities
	indices = make([]int, 0, batchSize)
	sampledProbs := make([]float64, 0, batchSize)

	for len(indices) < batchSize && len(indices) < rb.size {
		r := rand.Float64()
		cumProb := 0.0
		for i, p := range probs {
			cumProb += p
			if r <= cumProb {
				indices = append(indices, i)
				sampledProbs = append(sampledProbs, probs[i])
				break
			}
		}
	}

	// Compute importance sampling weights
	n := float64(rb.size)
	weights = make([]float64, len(indices))
	for i, idx := range indices {
		weight := math.Pow(n*probs[idx], -rb.beta)
		weights[i] = weight
	}

	// Normalize weights
	maxWeight := weights[0]
	for _, w := range weights[1:] {
		if w > maxWeight {
			maxWeight = w
		}
	}
	for i := range weights {
		weights[i] /= maxWeight
	}

	// Extract experiences
	states = make([][]float64, len(indices))
	actions = make([]int, len(indices))
	rewards = make([]float64, len(indices))
	nextStates = make([][]float64, len(indices))
	dones = make([]bool, len(indices))

	for i, idx := range indices {
		states[i] = rb.states[idx]
		actions[i] = rb.actions[idx]
		rewards[i] = rb.rewards[idx]
		nextStates[i] = rb.nextStates[idx]
		dones[i] = rb.dones[idx]
	}

	return states, nextStates, actions, rewards, dones, weights, indices
}

// UpdatePriorities updates the priorities of sampled experiences
func (rb *ReplayBuffer) UpdatePriorities(indices []int, tdErrors []float64) {
	for i, idx := range indices {
		priority := math.Abs(tdErrors[i]) + 1e-5
		rb.priorities[idx] = math.Pow(priority, rb.alpha)
	}
}

// Size returns the current size of the buffer
func (rb *ReplayBuffer) Size() int {
	return rb.size
}

// ValueNetwork is a simple 2-layer MLP for state value estimation
type ValueNetwork struct {
	w1 [][]float64
	b1 []float64
	w2 [][]float64
	b2 []float64
}

// NewValueNetwork creates a new value network
func NewValueNetwork(inputDim, hiddenDim int) *ValueNetwork {
	w1 := make([][]float64, inputDim)
	for i := range w1 {
		w1[i] = make([]float64, hiddenDim)
		for j := range w1[i] {
			w1[i][j] = rand.NormFloat64() * 0.01
		}
	}

	b1 := make([]float64, hiddenDim)
	for i := range b1 {
		b1[i] = 0.0
	}

	w2 := make([][]float64, hiddenDim)
	for i := range w2 {
		w2[i] = make([]float64, 1)
		for j := range w2[i] {
			w2[i][j] = rand.NormFloat64() * 0.01
		}
	}

	b2 := make([]float64, 1)
	b2[0] = 0.0

	return &ValueNetwork{w1: w1, b1: b1, w2: w2, b2: b2}
}

// Forward computes the value estimate for a state
func (vn *ValueNetwork) Forward(state []float64) float64 {
	inputDim := len(state)
	hiddenDim := len(vn.b1)

	// First layer: linear + ReLU
	hidden := make([]float64, hiddenDim)
	for j := 0; j < hiddenDim; j++ {
		sum := vn.b1[j]
		for i := 0; i < inputDim; i++ {
			sum += state[i] * vn.w1[i][j]
		}
		hidden[j] = relu(sum)
	}

	// Output layer: linear
	output := vn.b2[0]
	for i := 0; i < hiddenDim; i++ {
		output += hidden[i] * vn.w2[i][0]
	}

	return output
}

// PolicyGradient implements the REINFORCE algorithm
type PolicyGradient struct {
	// Policy network (simple softmax policy)
	policyW [][]float64 // stateDim x numActions
	policyB []float64   // numActions

	// Value network for baseline
	valueNet *ValueNetwork

	// Hyperparameters
	learningRate float64
	gamma         float64 // discount factor
	numActions    int
	stateDim      int
}

// TrajectoryStep represents a single step in a trajectory
type TrajectoryStep struct {
	State  []float64
	Action int
	Reward float64
}

// NewPolicyGradient creates a new REINFORCE agent
func NewPolicyGradient(stateDim, numActions int, lr, gamma float64) *PolicyGradient {
	policyW := make([][]float64, stateDim)
	for i := range policyW {
		policyW[i] = make([]float64, numActions)
		for j := range policyW[i] {
			policyW[i][j] = rand.NormFloat64() * 0.01
		}
	}

	policyB := make([]float64, numActions)
	for i := range policyB {
		policyB[i] = 0.0
	}

	return &PolicyGradient{
		policyW:      policyW,
		policyB:      policyB,
		valueNet:     NewValueNetwork(stateDim, 32),
		learningRate: lr,
		gamma:        gamma,
		numActions:   numActions,
		stateDim:     stateDim,
	}
}

// Forward computes action probabilities
func (pg *PolicyGradient) Forward(state []float64) []float64 {
	// Compute logits
	logits := make([]float64, pg.numActions)
	for a := 0; a < pg.numActions; a++ {
		logits[a] = pg.policyB[a]
		for i := 0; i < pg.stateDim; i++ {
			logits[a] += state[i] * pg.policyW[i][a]
		}
	}

	// Softmax
	exp := make([]float64, pg.numActions)
	sum := 0.0
	for a := 0; a < pg.numActions; a++ {
		exp[a] = math.Exp(logits[a] - maxFloatSlice(logits))
		sum += exp[a]
	}

	probs := make([]float64, pg.numActions)
	for a := 0; a < pg.numActions; a++ {
		probs[a] = exp[a] / sum
	}

	return probs
}

// SelectAction selects an action based on current policy
func (pg *PolicyGradient) SelectAction(state []float64) int {
	probs := pg.Forward(state)

	// Sample from distribution
	r := rand.Float64()
	cumProb := 0.0
	for a := 0; a < pg.numActions; a++ {
		cumProb += probs[a]
		if r <= cumProb {
			return a
		}
	}

	return pg.numActions - 1
}

// ComputeLogProb computes log probability of taking an action
func (pg *PolicyGradient) ComputeLogProb(state []float64, action int) float64 {
	probs := pg.Forward(state)
	if probs[action] <= 1e-8 {
		return -20.0 // log(1e-9)
	}
	return math.Log(probs[action])
}

// Train performs a policy gradient update
func (pg *PolicyGradient) Train(trajectory []TrajectoryStep) float64 {
	// Compute returns (discounted cumulative rewards)
	returns := make([]float64, len(trajectory))
	G := 0.0
	for t := len(trajectory) - 1; t >= 0; t-- {
		G = trajectory[t].Reward + pg.gamma*G
		returns[t] = G
	}

	// Compute baseline
	baselines := make([]float64, len(trajectory))
	for t, traj := range trajectory {
		baselines[t] = pg.valueNet.Forward(traj.State)
	}

	// Compute advantages
	advantages := make([]float64, len(trajectory))
	for t := 0; t < len(trajectory); t++ {
		advantages[t] = returns[t] - baselines[t]
	}

	// Compute policy gradient loss
	totalLoss := 0.0
	for t, traj := range trajectory {
		logProb := pg.ComputeLogProb(traj.State, traj.Action)
		totalLoss += -logProb * advantages[t]
	}

	// Update policy (simplified gradient descent)
	epsilon := 1e-4
	pg.policyGradientUpdate(trajectory, advantages, epsilon)

	// Update value network
	pg.valueGradientUpdate(trajectory, returns, epsilon)

	return totalLoss / float64(len(trajectory))
}

// policyGradientUpdate updates policy network weights
func (pg *PolicyGradient) policyGradientUpdate(trajectory []TrajectoryStep, advantages []float64, epsilon float64) {

	for i := 0; i < pg.stateDim; i++ {
		for a := 0; a < pg.numActions; a++ {
			// Numerical gradient
			orig := pg.policyW[i][a]
			lossPlus := pg.computePolicyLoss(trajectory, advantages, i, a, orig+epsilon)
			lossMinus := pg.computePolicyLoss(trajectory, advantages, i, a, orig-epsilon)
			grad := (lossPlus - lossMinus) / (2 * epsilon)

			pg.policyW[i][a] -= pg.learningRate * grad
		}
	}

	// Update bias
	for a := 0; a < pg.numActions; a++ {
		orig := pg.policyB[a]
		lossPlus := pg.computePolicyLossBias(trajectory, advantages, a, orig+epsilon)
		lossMinus := pg.computePolicyLossBias(trajectory, advantages, a, orig-epsilon)
		grad := (lossPlus - lossMinus) / (2 * epsilon)

		pg.policyB[a] -= pg.learningRate * grad
	}
}

func (pg *PolicyGradient) computePolicyLoss(trajectory []TrajectoryStep, advantages []float64, i, a int, weight float64) float64 {
	// Temporarily set weight
	orig := pg.policyW[i][a]
	pg.policyW[i][a] = weight

	loss := 0.0
	for t := range trajectory {
		logProb := pg.ComputeLogProb(trajectory[t].State, trajectory[t].Action)
		loss += -logProb * advantages[t]
	}

	// Restore
	pg.policyW[i][a] = orig

	return loss / float64(len(trajectory))
}

func (pg *PolicyGradient) computePolicyLossBias(trajectory []TrajectoryStep, advantages []float64, a int, bias float64) float64 {
	orig := pg.policyB[a]
	pg.policyB[a] = bias

	loss := 0.0
	for t := range trajectory {
		logProb := pg.ComputeLogProb(trajectory[t].State, trajectory[t].Action)
		loss += -logProb * advantages[t]
	}

	pg.policyB[a] = orig

	return loss / float64(len(trajectory))
}

// valueGradientUpdate updates value network
func (pg *PolicyGradient) valueGradientUpdate(trajectory []TrajectoryStep, returns []float64, epsilon float64) {
	lr := pg.learningRate * 0.1 // Smaller learning rate for value net

	// Update w1
	for i := 0; i < pg.stateDim; i++ {
		for j := 0; j < len(pg.valueNet.b1); j++ {
			orig := pg.valueNet.w1[i][j]
			lossPlus := pg.computeValueLoss(trajectory, returns, i, j, orig+epsilon)
			lossMinus := pg.computeValueLoss(trajectory, returns, i, j, orig-epsilon)
			grad := (lossPlus - lossMinus) / (2 * epsilon)
			pg.valueNet.w1[i][j] -= lr * grad
		}
	}

	// Update b1
	for j := 0; j < len(pg.valueNet.b1); j++ {
		orig := pg.valueNet.b1[j]
		lossPlus := pg.computeValueLossB1(trajectory, returns, j, orig+epsilon)
		lossMinus := pg.computeValueLossB1(trajectory, returns, j, orig-epsilon)
		grad := (lossPlus - lossMinus) / (2 * epsilon)
		pg.valueNet.b1[j] -= lr * grad
	}

	// Update w2
	for i := 0; i < len(pg.valueNet.w2); i++ {
		for j := 0; j < len(pg.valueNet.b2); j++ {
			orig := pg.valueNet.w2[i][j]
			lossPlus := pg.computeValueLossW2(trajectory, returns, i, j, orig+epsilon)
			lossMinus := pg.computeValueLossW2(trajectory, returns, i, j, orig-epsilon)
			grad := (lossPlus - lossMinus) / (2 * epsilon)
			pg.valueNet.w2[i][j] -= lr * grad
		}
	}
}

func (pg *PolicyGradient) computeValueLoss(trajectory []TrajectoryStep, returns []float64, i, j int, weight float64) float64 {
	orig := pg.valueNet.w1[i][j]
	pg.valueNet.w1[i][j] = weight

	loss := 0.0
	for t := range trajectory {
		pred := pg.valueNet.Forward(trajectory[t].State)
		diff := pred - returns[t]
		loss += diff * diff
	}

	pg.valueNet.w1[i][j] = orig

	return loss / float64(len(trajectory))
}

func (pg *PolicyGradient) computeValueLossB1(trajectory []TrajectoryStep, returns []float64, j int, bias float64) float64 {
	orig := pg.valueNet.b1[j]
	pg.valueNet.b1[j] = bias

	loss := 0.0
	for t := range trajectory {
		pred := pg.valueNet.Forward(trajectory[t].State)
		diff := pred - returns[t]
		loss += diff * diff
	}

	pg.valueNet.b1[j] = orig

	return loss / float64(len(trajectory))
}

func (pg *PolicyGradient) computeValueLossW2(trajectory []TrajectoryStep, returns []float64, i, j int, weight float64) float64 {
	orig := pg.valueNet.w2[i][j]
	pg.valueNet.w2[i][j] = weight

	loss := 0.0
	for t := range trajectory {
		pred := pg.valueNet.Forward(trajectory[t].State)
		diff := pred - returns[t]
		loss += diff * diff
	}

	pg.valueNet.w2[i][j] = orig

	return loss / float64(len(trajectory))
}

// PPO implements Proximal Policy Optimization
type PPO struct {
	policy        *PolicyGradient
	clipEpsilon   float64 // Clipping parameter for PPO
	gamma         float64
	lambda        float64 // GAE lambda
	valueCoeff    float64 // Value loss coefficient
	entropyCoeff  float64 // Entropy bonus coefficient
}

// TrajectoryWithDone represents a trajectory step with done flag
type TrajectoryWithDone struct {
	State  []float64
	Action int
	Reward float64
	Done   bool
}

// NewPPO creates a new PPO agent
func NewPPO(stateDim, numActions int, lr, gamma, clipEpsilon float64) *PPO {
	return &PPO{
		policy:        NewPolicyGradient(stateDim, numActions, lr, gamma),
		clipEpsilon:   clipEpsilon,
		gamma:         gamma,
		lambda:        0.95,
		valueCoeff:    0.5,
		entropyCoeff:  0.01,
	}
}

// GAE computes Generalized Advantage Estimation
func (ppo *PPO) GAE(trajectory []TrajectoryWithDone, values []float64) []float64 {
	advantages := make([]float64, len(trajectory))

	// Last advantage is 0 (no next state)
	advantage := 0.0

	for t := len(trajectory) - 1; t >= 0; t-- {
		if t == len(trajectory)-1 {
			nextValue := 0.0 // Terminal state
			tdError := trajectory[t].Reward + ppo.gamma*nextValue - values[t]
			advantage = tdError
		} else {
			nextValue := values[t+1]
			tdError := trajectory[t].Reward + ppo.gamma*nextValue - values[t]
			advantage = tdError + ppo.gamma*ppo.lambda*advantage
		}
		advantages[t] = advantage
	}

	return advantages
}

// TrainPPO performs a PPO update
func (ppo *PPO) TrainPPO(trajectory []TrajectoryWithDone) float64 {
	// Compute values for current policy
	values := make([]float64, len(trajectory))
	for t, traj := range trajectory {
		values[t] = traj.State[0] // Simplified: just use first dimension as value
		// In practice, use value network
		for i := 1; i < len(traj.State); i++ {
			values[t] += traj.State[i]
		}
		values[t] /= float64(len(traj.State))
	}

	// Compute advantages using GAE
	advantages := ppo.GAE(trajectory, values)

	// Compute returns (for value function training)
	returns := make([]float64, len(trajectory))
	for t := 0; t < len(trajectory); t++ {
		returns[t] = advantages[t] + values[t]
	}

	// PPO loss with clipped surrogate objective
	loss := ppo.computePPOLoss(trajectory, advantages)

	// Update policy (convert trajectory types)
	simpleTraj := make([]TrajectoryStep, len(trajectory))
	for i, t := range trajectory {
		simpleTraj[i] = TrajectoryStep{
			State:  t.State,
			Action: t.Action,
			Reward: t.Reward,
		}
	}
	epsilon := 1e-4
	ppo.policy.policyGradientUpdate(simpleTraj, advantages, epsilon)

	// Update value network
	ppo.policy.valueGradientUpdate(simpleTraj, returns, epsilon)

	return loss
}

func (ppo *PPO) computePPOLoss(trajectory []TrajectoryWithDone, advantages []float64) float64 {
	policyLoss := 0.0

	for t := range trajectory {
		logProb := ppo.policy.ComputeLogProb(trajectory[t].State, trajectory[t].Action)

		// Compute surrogate loss (simplified)
		ratio := math.Exp(logProb)
		surr1 := ratio * advantages[t]
		surr2 := clamp(ratio, 1-ppo.clipEpsilon, 1+ppo.clipEpsilon) * advantages[t]

		policyLoss += -minFloat(surr1, surr2)
	}

	// Add entropy bonus for exploration
	entropy := 0.0
	for _, traj := range trajectory {
		probs := ppo.policy.Forward(traj.State)
		for _, p := range probs {
			if p > 1e-8 {
				entropy -= p * math.Log(p)
			}
		}
	}
	entropy /= float64(len(trajectory))

	return policyLoss/float64(len(trajectory)) - ppo.entropyCoeff*entropy
}

// Helper functions
func copySlice(s []float64) []float64 {
	c := make([]float64, len(s))
	copy(c, s)
	return c
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func maxFloatSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := s[0]
	for i := 1; i < len(s); i++ {
		if s[i] > m {
			m = s[i]
		}
	}
	return m
}

func clamp(val, min, max float64) float64 {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// GetActionProbabilities returns the action probabilities for a state
func (ppo *PPO) GetActionProbabilities(state []float64) []float64 {
	return ppo.policy.Forward(state)
}

// SelectActionPPO selects an action using PPO policy
func (ppo *PPO) SelectActionPPO(state []float64) int {
	return ppo.policy.SelectAction(state)
}

// Experience stores a single experience tuple for the buffer
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// PrioritizedReplayBuffer is an alias for ReplayBuffer
type PrioritizedReplayBuffer = ReplayBuffer
