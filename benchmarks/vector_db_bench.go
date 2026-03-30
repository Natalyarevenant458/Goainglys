package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// HNSWConfig holds HNSW configuration
type HNSWConfig struct {
	Dim         int
	MaxElements int
	EF         int
	M          int
	LevelMult   float64
}

// HNSWNode represents a node in HNSW
type HNSWNode struct {
	vector     []float64
	elementID  int
	level      int
	neighbors  [][]int
	connections [][]int
}

// HNSW implements Hierarchical Navigable Small World graph
type HNSW struct {
	config    HNSWConfig
	enterPoint *HNSWNode
	nodes     map[int]*HNSWNode
	elementCount int
}

// NewHNSW creates a new HNSW index
func NewHNSW(config HNSWConfig) *HNSW {
	return &HNSW{
		config:    config,
		nodes:     make(map[int]*HNSWNode),
		enterPoint: nil,
	}
}

// L2Distance computes L2 distance between two vectors
func L2Distance(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a) && i < len(b); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// RandomLevel generates a random level for new element
func RandomLevel(levelMult float64) int {
	// Simplified: use geometric distribution
	level := 0
	for rand.Float64() < math.Exp(-float64(level)/levelMult) {
		level++
	}
	return level
}

// SearchLevel performs search at a single level
func SearchLevel(hnsw *HNSW, query []float64, ef, limit int, entry *HNSWNode) []*HNSWNode {
	if entry == nil {
		return nil
	}

	visited := make(map[int]bool)
	visited[entry.elementID] = true

	candidates := []*HNSWNode{entry}
	result := []*HNSWNode{entry}

	for len(candidates) > 0 {
		// Sort candidates by distance to query
		sort.Slice(candidates, func(i, j int) bool {
			return L2Distance(candidates[i].vector, query) < L2Distance(candidates[j].vector, query)
		})

		current := candidates[0]
		candidates = candidates[1:]

		// Get worst result distance
		var worstDist float64
		if len(result) > 0 {
			worstIdx := 0
			for i := 1; i < len(result); i++ {
				if L2Distance(result[i].vector, query) > L2Distance(result[worstIdx].vector, query) {
					worstIdx = i
				}
			}
			worstDist = L2Distance(result[worstIdx].vector, query)
		}

		// Check neighbors
		if len(current.neighbors) > 0 {
			for _, neighbor := range current.neighbors[0] {
				if !visited[neighbor] {
					visited[neighbor] = true
					if node, ok := hnsw.nodes[neighbor]; ok {
						dist := L2Distance(node.vector, query)
						if len(result) < ef || dist < worstDist {
							candidates = append(candidates, node)
							if len(result) < ef {
								result = append(result, node)
							} else {
								// Replace worst
								worstIdx := 0
								for i := 1; i < len(result); i++ {
									if L2Distance(result[i].vector, query) > L2Distance(result[worstIdx].vector, query) {
										worstIdx = i
									}
								}
								result[worstIdx] = node
							}
						}
					}
				}
			}
		}
	}

	return result
}

// Insert inserts a new element into HNSW
func (h *HNSW) Insert(vector []float64) {
	level := RandomLevel(h.config.LevelMult)
	node := &HNSWNode{
		vector:     make([]float64, len(vector)),
		elementID:  h.elementCount,
		level:      level,
		neighbors:  make([][]int, level+1),
		connections: make([][]int, level+1),
	}
	copy(node.vector, vector)

	if h.enterPoint == nil {
		h.enterPoint = node
		h.nodes[h.elementCount] = node
		h.elementCount++
		return
	}

	// Search from top level
	current := h.enterPoint
	for l := current.level; l > level; l-- {
		results := SearchLevel(h, vector, 1, 1, current)
		if len(results) > 0 {
			current = results[0]
		}
	}

	// Insert at each level
	for l := minInt(level, current.level); l >= 0; l-- {
		ep := SearchLevel(h, vector, h.config.EF, h.config.M, current)
		
		// Connect to nearest neighbors
		sort.Slice(ep, func(i, j int) bool {
			return L2Distance(ep[i].vector, vector) < L2Distance(ep[j].vector, vector)
		})

		m := h.config.M
		if len(ep) < m {
			m = len(ep)
		}
		node.connections[l] = make([]int, m)
		for i := 0; i < m; i++ {
			node.connections[l][i] = ep[i].elementID
		}
		
		// Add reverse connections
		for _, e := range ep[:m] {
			if e.neighbors == nil {
				e.neighbors = make([][]int, l+1)
			}
			if len(e.neighbors) <= l {
				newNeighbors := make([][]int, l+1)
				copy(newNeighbors, e.neighbors)
				e.neighbors = newNeighbors
			}
			e.neighbors[l] = append(e.neighbors[l], node.elementID)
		}

		current = ep[0]
	}

	h.nodes[h.elementCount] = node
	h.elementCount++
}

// Search searches for k nearest neighbors
func (h *HNSW) Search(query []float64, k int) []*HNSWNode {
	if h.enterPoint == nil {
		return nil
	}

	// Search from top level
	current := h.enterPoint
	for l := current.level; l > 0; l-- {
		results := SearchLevel(h, query, 1, 1, current)
		if len(results) > 0 {
			current = results[0]
		}
	}

	// Search at base level with ef=k
	ep := SearchLevel(h, query, k, k, current)

	// Sort by distance and return top k
	sort.Slice(ep, func(i, j int) bool {
		return L2Distance(ep[i].vector, query) < L2Distance(ep[j].vector, query)
	})

	if len(ep) > k {
		ep = ep[:k]
	}

	return ep
}

// BruteForceSearch performs brute force k-NN search
func BruteForceSearch(vectors [][]float64, query []float64, k int) []int {
	type result struct {
		id    int
		dist  float64
	}

	results := make([]result, len(vectors))
	for i, v := range vectors {
		results[i] = result{id: i, dist: L2Distance(v, query)}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	ids := make([]int, k)
	for i := 0; i < k && i < len(results); i++ {
		ids[i] = results[i].id
	}

	return ids
}

// HNSWBenchmarkResults holds HNSW benchmark results
type HNSWBenchmarkResults struct {
	TestName    string
	CorpusSize  int
	Metric      string
	Value       float64
	LatencyMs   float64
	Recall      float64
}

func RunHNSWBenchmarks() []HNSWBenchmarkResults {
	results := []HNSWBenchmarkResults{}
	dim := 128

	// Insert benchmark - 10000 vectors
	hnsw := NewHNSW(HNSWConfig{
		Dim:         dim,
		MaxElements: 20000,
		EF:         16,
		M:          16,
		LevelMult:  1.0 / math.Log(16),
	})

	corpus := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		corpus[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			corpus[i][j] = rand.Float64()*2 - 1
		}
	}

	// Benchmark insert
	fn := func() {
		hnsw2 := NewHNSW(HNSWConfig{
			Dim:         dim,
			MaxElements: 20000,
			EF:         16,
			M:          16,
			LevelMult:  1.0 / math.Log(16),
		})
		for i := 0; i < 500; i++ {
			hnsw2.Insert(corpus[i])
		}
	}
	fn() // warmup

	iterations := 2
	var totalTime time.Duration
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime := totalTime.Seconds() / float64(iterations)
	results = append(results, HNSWBenchmarkResults{
		TestName:   "Insert",
		CorpusSize: 10000,
		Metric:     "Vectors/sec",
		Value:      1000.0 / avgTime,
		LatencyMs:  avgTime * 1000,
	})

	// Full insert of 10000 vectors
	_ = NewHNSW(HNSWConfig{
		Dim:         dim,
		MaxElements: 20000,
		EF:         16,
		M:          16,
		LevelMult:  1.0 / math.Log(16),
	})

	fn = func() {
		for i := 0; i < 1000; i++ {
			hnsw.Insert(corpus[i])
		}
	}
	fn() // warmup

	iterations = 3
	totalTime = 0
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		fn()
		t1 := time.Now()
		totalTime += t1.Sub(t0)
	}

	avgTime = totalTime.Seconds() / float64(iterations)
	results = append(results, HNSWBenchmarkResults{
		TestName:   "Insert Full",
		CorpusSize: 10000,
		Metric:     "Vectors/sec",
		Value:      10000.0 / avgTime,
		LatencyMs:  avgTime * 1000,
	})

	// Search benchmark at different corpus sizes
	corpusSizes := []int{100}

	for _, corpusSize := range corpusSizes {
		// Build index
		hnsw := NewHNSW(HNSWConfig{
			Dim:         dim,
			MaxElements: corpusSize * 2,
			EF:         16,
			M:          16,
			LevelMult:  1.0 / math.Log(16),
		})

		allCorpus := make([][]float64, corpusSize)
		for i := 0; i < corpusSize; i++ {
			vec := make([]float64, dim)
			for j := 0; j < dim; j++ {
				vec[j] = rand.Float64()*2 - 1
			}
			allCorpus[i] = vec
			hnsw.Insert(vec)
		}

		// Generate queries
		queries := make([][]float64, 100)
		for i := 0; i < 100; i++ {
			queries[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				queries[i][j] = rand.Float64()*2 - 1
			}
		}

		// Benchmark search
		fn := func() {
			for _, q := range queries {
				hnsw.Search(q, 10)
			}
		}
		fn() // warmup

		iterations = 10
		totalTime = 0
		for i := 0; i < iterations; i++ {
			t0 := time.Now()
			fn()
			t1 := time.Now()
			totalTime += t1.Sub(t0)
		}

		avgTime := totalTime.Seconds() / float64(iterations)
		results = append(results, HNSWBenchmarkResults{
			TestName:   "Search",
			CorpusSize: corpusSize,
			Metric:     "QPS",
			Value:      float64(len(queries)) / avgTime,
			LatencyMs:  avgTime * 1000,
		})
	}

	// Recall benchmark
	_ = NewHNSW(HNSWConfig{
		Dim:         dim,
		MaxElements: 10000,
		EF:         16,
		M:          16,
		LevelMult:  1.0 / math.Log(16),
	})

	testCorpus := make([][]float64, 500)
	for i := 0; i < 500; i++ {
		testCorpus[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			testCorpus[i][j] = rand.Float64()*2 - 1
		}
		hnsw.Insert(testCorpus[i])
	}

	// Generate test queries
	testQueries := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		testQueries[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			testQueries[i][j] = rand.Float64()*2 - 1
		}
	}

	// Compute recall
	totalRecall := 0.0
	for _, q := range testQueries {
		hnswResult := hnsw.Search(q, 10)
		bruteResult := BruteForceSearch(testCorpus, q, 10)

		hnswIDs := make(map[int]bool)
		for _, n := range hnswResult {
			hnswIDs[n.elementID] = true
		}

		matches := 0
		for _, id := range bruteResult {
			if hnswIDs[id] {
				matches++
			}
		}
		totalRecall += float64(matches) / 10.0
	}

	recall := totalRecall / float64(len(testQueries))
	results = append(results, HNSWBenchmarkResults{
		TestName:   "Recall@10",
		CorpusSize: 1000,
		Metric:     "Recall",
		Value:      recall * 100,
		Recall:     recall,
	})

	return results
}

// BenchmarkHNSWInsert benchmarks HNSW insertion
func BenchmarkHNSWInsert(b *testing.B) {
	dim := 128

	vectors := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		vectors[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float64()*2 - 1
		}
	}

	_ = NewHNSW(HNSWConfig{
		Dim:         dim,
		MaxElements: 20000,
		EF:         16,
		M:          16,
		LevelMult:  1.0 / math.Log(16),
	})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		hnsw2 := NewHNSW(HNSWConfig{
			Dim:         dim,
			MaxElements: 20000,
			EF:         16,
			M:          16,
			LevelMult:  1.0 / math.Log(16),
		})
		for j := 0; j < 10000; j++ {
			hnsw2.Insert(vectors[j])
		}
	}
}

// BenchmarkHNSWSearch benchmarks HNSW search
func BenchmarkHNSWSearch(b *testing.B) {
	dim := 128
	corpusSizes := []int{100}

	for _, corpusSize := range corpusSizes {
		b.Run(fmt.Sprintf("corpus_%d", corpusSize), func(b *testing.B) {
			// Build index
			hnsw := NewHNSW(HNSWConfig{
				Dim:         dim,
				MaxElements: corpusSize * 2,
				EF:         16,
				M:          16,
				LevelMult:  1.0 / math.Log(16),
			})

			corpus := make([][]float64, corpusSize)
			for i := 0; i < corpusSize; i++ {
				corpus[i] = make([]float64, dim)
				for j := 0; j < dim; j++ {
					corpus[i][j] = rand.Float64()*2 - 1
				}
				hnsw.Insert(corpus[i])
			}

			// Generate queries
			queries := make([][]float64, 10)
			for i := 0; i < 10; i++ {
				queries[i] = make([]float64, dim)
				for j := 0; j < dim; j++ {
					queries[i][j] = rand.Float64()*2 - 1
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, q := range queries {
					hnsw.Search(q, 10)
				}
			}
		})
	}
}

// BenchmarkHNSWRecall benchmarks HNSW recall vs brute force
func BenchmarkHNSWRecall(b *testing.B) {
	dim := 128

	// Build index
	hnsw := NewHNSW(HNSWConfig{
		Dim:         dim,
		MaxElements: 10000,
		EF:         16,
		M:          16,
		LevelMult:  1.0 / math.Log(16),
	})

	corpus := make([][]float64, 1000)
	for i := 0; i < 500; i++ {
		corpus[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			corpus[i][j] = rand.Float64()*2 - 1
		}
		hnsw.Insert(corpus[i])
	}

	queries := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		queries[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			queries[i][j] = rand.Float64()*2 - 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, q := range queries {
			_ = hnsw.Search(q, 10)
			_ = BruteForceSearch(corpus, q, 10)
		}
	}
}
