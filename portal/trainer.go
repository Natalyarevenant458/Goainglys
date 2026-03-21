package main

import (
        "fmt"
        "math"
        "math/rand"
        "sort"
        "sync"
        "time"
)

// ─────────────────────────────────────────────────────────────────────
// Training Corpora
// ─────────────────────────────────────────────────────────────────────

var corpora = map[string]string{
        "shakespeare": `First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak. First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved. First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know it, we know it. First Citizen: Let us kill him, and we will have corn at our own price.
Second Citizen: One word, good citizens. First Citizen: We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they would yield us but the superfluity,
while it were wholesome, we might guess they relieved us humanely; but they think we are too dear:
the leanness that afflicts us, the object of our misery, is as an inventory to particularise
their abundance; our sufferance is a gain to them. Let us revenge this with our pikes.
To be, or not to be, that is the question: Whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune, or to take arms against a sea of troubles,
and by opposing end them? To die, to sleep, no more; and by a sleep to say we end
the heart-ache and the thousand natural shocks that flesh is heir to.
All the world is a stage, and all the men and women merely players;
they have their exits and their entrances, and one man in his time plays many parts.
Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.
The evil that men do lives after them; the good is oft interred with their bones.
What a piece of work is a man! how noble in reason! how infinite in faculty!
in form and moving how express and admirable! in action how like an angel!
in apprehension how like a god! the beauty of the world! the paragon of animals!`,

        "code": `package main

import fmt math sort strings

func fibonacci n int int {
        if n less than two { return n }
        return fibonacci n minus one plus fibonacci n minus two
}
func isPrime n int bool {
        if n less two { return false }
        for i two; i times i less n; i plus plus {
                if n mod i equals zero { return false }
        }
        return true
}
func bubbleSort arr []int []int {
        n := len arr
        for i zero; i less n minus one; i plus plus {
                for j zero; j less n minus i minus one; j plus plus {
                        if arr j greater arr j plus one {
                                arr j, arr j plus one = arr j plus one, arr j
                        }
                }
        }
        return arr
}
func reverseString s string string {
        runes := []rune s
        for i, j := zero, len runes minus one; i less j; i, j = i plus one, j minus one {
                runes i, runes j = runes j, runes i
        }
        return string runes
}
func wordCount text string map string int {
        counts := make map string int
        words := strings Fields text
        for _, w := range words {
                counts strings ToLower w plus plus
        }
        return counts
}
func main {
        nums := []int{5, 2, 8, 1, 9, 3}
        sorted := bubbleSort nums
        for i, n := range sorted {
                fmt Printf sorted index equals value newline i n
        }
        for i zero; i less ten; i plus plus {
                fmt Printf fib equals prime newline i fibonacci i isPrime i
        }
}`,

        "math": `The fundamental theorem of calculus connects differentiation and integration.
If a function f is continuous on the closed interval a to b and F is an antiderivative,
then the definite integral from a to b equals F of b minus F of a.
The derivative of the product of f and g equals f prime times g plus f times g prime.
For any real number x the exponential function e to the x equals the infinite sum
of x to the n divided by n factorial where n ranges from zero to infinity.
The sum of interior angles of a triangle is one hundred eighty degrees.
Euler identity states that e to the power of i times pi plus one equals zero.
The Pythagorean theorem states that a squared plus b squared equals c squared.
In a right triangle with legs a and b and hypotenuse c this relation always holds.
The quadratic formula gives x equals negative b plus or minus the square root
of b squared minus four a c all divided by two a.
A matrix multiplied by its inverse gives the identity matrix.
The determinant of a two by two matrix a b c d equals a times d minus b times c.
Linear algebra studies vector spaces and linear transformations between them.
Probability theory defines the probability of an event A as a number between zero and one.
Bayes theorem relates conditional probabilities P of A given B equals P of B given A
times P of A divided by P of B. The expected value is the probability weighted average.
Normal distribution is symmetric around its mean with standard deviation sigma.`,
}

// ─────────────────────────────────────────────────────────────────────
// Vocabulary
// ─────────────────────────────────────────────────────────────────────

type Vocab struct {
        charToIdx map[rune]int
        idxToChar []rune
        Size      int
}

func buildVocab(text string) *Vocab {
        seen := map[rune]bool{}
        for _, ch := range text {
                seen[ch] = true
        }
        chars := make([]rune, 0, len(seen))
        for ch := range seen {
                chars = append(chars, ch)
        }
        sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })
        v := &Vocab{charToIdx: make(map[rune]int), idxToChar: chars, Size: len(chars)}
        for i, ch := range chars {
                v.charToIdx[ch] = i
        }
        return v
}

func (v *Vocab) Encode(text string) []int {
        out := make([]int, 0, len(text))
        for _, ch := range text {
                if idx, ok := v.charToIdx[ch]; ok {
                        out = append(out, idx)
                }
        }
        return out
}

// ─────────────────────────────────────────────────────────────────────
// Matrix — row-major float64
// ─────────────────────────────────────────────────────────────────────

type Mat struct {
        d    []float64
        r, c int
}

func newMat(r, c int) *Mat            { return &Mat{d: make([]float64, r*c), r: r, c: c} }
func (m *Mat) at(i, j int) float64    { return m.d[i*m.c+j] }
func (m *Mat) set(i, j int, v float64) { m.d[i*m.c+j] = v }
func (m *Mat) acc(i, j int, v float64) { m.d[i*m.c+j] += v }
func (m *Mat) clone() *Mat {
        n := newMat(m.r, m.c)
        copy(n.d, m.d)
        return n
}

func randMatN(r, c int, std float64) *Mat {
        m := newMat(r, c)
        for i := range m.d {
                u1, u2 := rand.Float64()+1e-10, rand.Float64()
                m.d[i] = math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2) * std
        }
        return m
}

// [ar×ac] @ [br×bc] → [ar×bc]
func matMul(a, b *Mat) *Mat {
        out := newMat(a.r, b.c)
        for i := 0; i < a.r; i++ {
                for k := 0; k < a.c; k++ {
                        av := a.d[i*a.c+k]
                        if av == 0 {
                                continue
                        }
                        bBase, oBase := k*b.c, i*out.c
                        for j := 0; j < b.c; j++ {
                                out.d[oBase+j] += av * b.d[bBase+j]
                        }
                }
        }
        return out
}

// a @ b.T — [ar×ac] @ [br×ac].T → [ar×br]
func matMulTB(a, b *Mat) *Mat {
        out := newMat(a.r, b.r)
        for i := 0; i < a.r; i++ {
                for j := 0; j < b.r; j++ {
                        var s float64
                        for k := 0; k < a.c; k++ {
                                s += a.d[i*a.c+k] * b.d[j*b.c+k]
                        }
                        out.d[i*out.c+j] = s
                }
        }
        return out
}

// a.T @ b — [ac×ar] @ [ar×bc] → [ac×bc]
func matTMul(a, b *Mat) *Mat {
        out := newMat(a.c, b.c)
        for k := 0; k < a.r; k++ {
                for i := 0; i < a.c; i++ {
                        av := a.d[k*a.c+i]
                        if av == 0 {
                                continue
                        }
                        bBase, oBase := k*b.c, i*out.c
                        for j := 0; j < b.c; j++ {
                                out.d[oBase+j] += av * b.d[bBase+j]
                        }
                }
        }
        return out
}

func addBiasRows(m *Mat, b []float64) {
        for i := 0; i < m.r; i++ {
                base := i * m.c
                for j := 0; j < m.c; j++ {
                        m.d[base+j] += b[j]
                }
        }
}

func reluIP(m *Mat) {
        for i := range m.d {
                if m.d[i] < 0 {
                        m.d[i] = 0
                }
        }
}

func reluGrad(grad, fwd *Mat) *Mat {
        out := newMat(grad.r, grad.c)
        for i := range grad.d {
                if fwd.d[i] > 0 {
                        out.d[i] = grad.d[i]
                }
        }
        return out
}

func colSums(m *Mat) []float64 {
        s := make([]float64, m.c)
        for i := 0; i < m.r; i++ {
                for j := 0; j < m.c; j++ {
                        s[j] += m.d[i*m.c+j]
                }
        }
        return s
}

func softmaxCE(logits *Mat, targets []int) (*Mat, float64) {
        probs := newMat(logits.r, logits.c)
        var loss float64
        for i := 0; i < logits.r; i++ {
                base := i * logits.c
                maxV := logits.d[base]
                for j := 1; j < logits.c; j++ {
                        if v := logits.d[base+j]; v > maxV {
                                maxV = v
                        }
                }
                var sum float64
                for j := 0; j < logits.c; j++ {
                        v := math.Exp(logits.d[base+j] - maxV)
                        probs.d[base+j] = v
                        sum += v
                }
                for j := 0; j < logits.c; j++ {
                        probs.d[base+j] /= sum
                }
                loss -= math.Log(math.Max(probs.d[base+targets[i]], 1e-10))
        }
        return probs, loss / float64(logits.r)
}

func ceGrad(probs *Mat, targets []int) *Mat {
        grad := probs.clone()
        scale := 1.0 / float64(probs.r)
        for i := 0; i < probs.r; i++ {
                grad.acc(i, targets[i], -1)
        }
        for i := range grad.d {
                grad.d[i] *= scale
        }
        return grad
}

func top1Acc(logits *Mat, targets []int) float64 {
        correct := 0
        for i := 0; i < logits.r; i++ {
                base := i * logits.c
                best, bestV := 0, logits.d[base]
                for j := 1; j < logits.c; j++ {
                        if v := logits.d[base+j]; v > bestV {
                                bestV = v
                                best = j
                        }
                }
                if best == targets[i] {
                        correct++
                }
        }
        return float64(correct) / float64(logits.r)
}

// ─────────────────────────────────────────────────────────────────────
// Adam Optimizer
// ─────────────────────────────────────────────────────────────────────

type adamSlot struct{ m, v []float64 }

type Adam struct {
        lr, b1, b2, eps float64
        t               int
        slots           []adamSlot
}

func newAdam(lr float64) *Adam {
        return &Adam{lr: lr, b1: 0.9, b2: 0.999, eps: 1e-8}
}

func (a *Adam) track(slices ...[]float64) {
        for _, s := range slices {
                a.slots = append(a.slots, adamSlot{
                        m: make([]float64, len(s)),
                        v: make([]float64, len(s)),
                })
        }
}

func (a *Adam) step(idx int, param, grad []float64) {
        a.t++
        sl := &a.slots[idx]
        b1t := 1 - math.Pow(a.b1, float64(a.t))
        b2t := 1 - math.Pow(a.b2, float64(a.t))
        for i := range param {
                g := grad[i]
                sl.m[i] = a.b1*sl.m[i] + (1-a.b1)*g
                sl.v[i] = a.b2*sl.v[i] + (1-a.b2)*g*g
                param[i] -= a.lr * (sl.m[i] / b1t) / (math.Sqrt(sl.v[i]/b2t) + a.eps)
        }
}

// ─────────────────────────────────────────────────────────────────────
// CharLM — 2-layer MLP character language model
// ─────────────────────────────────────────────────────────────────────

type CharLM struct {
        V, C, E, H int // vocabSize, contextLen, embedDim, hiddenDim
        embed       *Mat
        W1, W2      *Mat
        b1, b2      []float64
        opt         *Adam
        freezeBase  bool
}

func newCharLM(V, C, E, H int, lr float64, freezeBase bool) *CharLM {
        inputDim := C * E
        m := &CharLM{
                V: V, C: C, E: E, H: H,
                embed:      randMatN(V, E, 0.05),
                W1:         randMatN(inputDim, H, math.Sqrt(2.0/float64(inputDim))),
                b1:         make([]float64, H),
                W2:         randMatN(H, V, math.Sqrt(2.0/float64(H))),
                b2:         make([]float64, V),
                opt:        newAdam(lr),
                freezeBase: freezeBase,
        }
        // slot indices: 0=embed, 1=W1, 2=b1, 3=W2, 4=b2
        m.opt.track(m.embed.d, m.W1.d, m.b1, m.W2.d, m.b2)
        return m
}

// trainStep: one forward + backward pass, returns loss and top-1 accuracy
func (m *CharLM) trainStep(contexts [][]int, targets []int) (loss, acc float64) {
        batch := len(contexts)
        inputDim := m.C * m.E

        // ── Forward ────────────────────────────────────────────────────────
        // gather embeddings → x [batch × inputDim]
        x := newMat(batch, inputDim)
        for b, ctx := range contexts {
                for c, ch := range ctx {
                        src := ch * m.E
                        dst := b*inputDim + c*m.E
                        copy(x.d[dst:dst+m.E], m.embed.d[src:src+m.E])
                }
        }
        // hidden layer
        h1 := matMul(x, m.W1)
        addBiasRows(h1, m.b1)
        reluIP(h1)
        // output
        logits := matMul(h1, m.W2)
        addBiasRows(logits, m.b2)

        // ── Loss & accuracy ────────────────────────────────────────────────
        probs, l := softmaxCE(logits, targets)
        loss = l
        acc = top1Acc(logits, targets)

        // ── Backward ───────────────────────────────────────────────────────
        dlogits := ceGrad(probs, targets)

        // output layer
        dW2 := matTMul(h1, dlogits) // [H × V]
        db2 := colSums(dlogits)
        m.opt.step(3, m.W2.d, dW2.d)
        m.opt.step(4, m.b2, db2)

        if !m.freezeBase {
                // hidden layer
                dh1 := reluGrad(matMulTB(dlogits, m.W2), h1) // [batch × H]
                dW1 := matTMul(x, dh1)
                db1 := colSums(dh1)
                m.opt.step(1, m.W1.d, dW1.d)
                m.opt.step(2, m.b1, db1)

                // embed scatter
                dx := matMulTB(dh1, m.W1) // [batch × inputDim]
                dEmbed := newMat(m.V, m.E)
                for b, ctx := range contexts {
                        for c, ch := range ctx {
                                src := b*inputDim + c*m.E
                                dst := ch * m.E
                                for e := 0; e < m.E; e++ {
                                        dEmbed.d[dst+e] += dx.d[src+e]
                                }
                        }
                }
                m.opt.step(0, m.embed.d, dEmbed.d)
        }
        return
}

// ─────────────────────────────────────────────────────────────────────
// Architecture presets
// ─────────────────────────────────────────────────────────────────────

type ArchPreset struct {
        EmbedDim   int
        HiddenDim  int
        ContextLen int
        FreezeBase bool
}

var archPresets = map[string]ArchPreset{
        "mlp-small":     {EmbedDim: 8, HiddenDim: 64, ContextLen: 4, FreezeBase: false},
        "mlp-base":      {EmbedDim: 16, HiddenDim: 128, ContextLen: 6, FreezeBase: false},
        "gpt-nano":      {EmbedDim: 32, HiddenDim: 256, ContextLen: 8, FreezeBase: false},
        "lora-finetune": {EmbedDim: 16, HiddenDim: 128, ContextLen: 6, FreezeBase: true},
}

// ─────────────────────────────────────────────────────────────────────
// Training Session
// ─────────────────────────────────────────────────────────────────────

type SessionConfig struct {
        Name      string  `json:"name"`
        ModelType string  `json:"model_type"`
        Dataset   string  `json:"dataset"`
        LR        float64 `json:"learning_rate"`
        BatchSize int     `json:"batch_size"`
        MaxSteps  int     `json:"max_steps"`
}

type SessionStatus string

const (
        StatusQueued    SessionStatus = "queued"
        StatusTraining  SessionStatus = "training"
        StatusCompleted SessionStatus = "completed"
        StatusStopped   SessionStatus = "stopped"
)

type SessionMetric struct {
        Step     int     `json:"step"`
        Loss     float64 `json:"loss"`
        Accuracy float64 `json:"accuracy"`
        LR       float64 `json:"lr"`
        Ts       int64   `json:"ts"`
}

type TrainingSession struct {
        ID          string          `json:"id"`
        Config      SessionConfig   `json:"config"`
        Status      SessionStatus   `json:"status"`
        Progress    float64         `json:"progress"`
        CurrentStep int             `json:"current_step"`
        TotalSteps  int             `json:"total_steps"`
        BestLoss    float64         `json:"best_loss"`
        FinalAcc    float64         `json:"final_accuracy"`
        Metrics     []SessionMetric `json:"metrics"`
        StartedAt   time.Time       `json:"started_at"`
        CompletedAt *time.Time      `json:"completed_at,omitempty"`
        SavedModel  string          `json:"saved_model,omitempty"`

        mu   sync.RWMutex
        stop chan struct{}
}

func (s *TrainingSession) snapshot() TrainingSession {
        s.mu.RLock()
        defer s.mu.RUnlock()
        cp := *s
        cp.Metrics = make([]SessionMetric, len(s.Metrics))
        copy(cp.Metrics, s.Metrics)
        return cp
}

// ─────────────────────────────────────────────────────────────────────
// Session Manager
// ─────────────────────────────────────────────────────────────────────

type SessionManager struct {
        mu       sync.RWMutex
        sessions []*TrainingSession
        seq      int
        models   *ModelStore
}

func NewSessionManager(ms *ModelStore) *SessionManager {
        return &SessionManager{models: ms}
}

func (sm *SessionManager) Start(cfg SessionConfig) *TrainingSession {
        sm.mu.Lock()
        sm.seq++
        s := &TrainingSession{
                ID:         fmt.Sprintf("session-%d", sm.seq),
                Config:     cfg,
                Status:     StatusQueued,
                TotalSteps: cfg.MaxSteps,
                BestLoss:   math.MaxFloat64,
                StartedAt:  time.Now(),
                stop:       make(chan struct{}),
        }
        sm.sessions = append(sm.sessions, s)
        sm.mu.Unlock()
        go sm.runSession(s)
        return s
}

func (sm *SessionManager) Stop(id string) bool {
        sm.mu.RLock()
        defer sm.mu.RUnlock()
        for _, s := range sm.sessions {
                if s.ID == id {
                        s.mu.RLock()
                        st := s.Status
                        s.mu.RUnlock()
                        if st == StatusTraining || st == StatusQueued {
                                select {
                                case <-s.stop:
                                default:
                                        close(s.stop)
                                }
                                return true
                        }
                }
        }
        return false
}

func (sm *SessionManager) List() []TrainingSession {
        sm.mu.RLock()
        ptrs := make([]*TrainingSession, len(sm.sessions))
        copy(ptrs, sm.sessions)
        sm.mu.RUnlock()
        out := make([]TrainingSession, len(ptrs))
        for i, p := range ptrs {
                out[i] = p.snapshot()
        }
        return out
}

func (sm *SessionManager) Get(id string) *TrainingSession {
        sm.mu.RLock()
        defer sm.mu.RUnlock()
        for _, s := range sm.sessions {
                if s.ID == id {
                        return s
                }
        }
        return nil
}

func (sm *SessionManager) runSession(s *TrainingSession) {
        s.mu.Lock()
        s.Status = StatusTraining
        s.mu.Unlock()

        cfg := s.Config
        preset, ok := archPresets[cfg.ModelType]
        if !ok {
                preset = archPresets["mlp-base"]
        }
        corpus, ok := corpora[cfg.Dataset]
        if !ok {
                corpus = corpora["shakespeare"]
        }
        vocab := buildVocab(corpus)
        tokens := vocab.Encode(corpus)

        if len(tokens) < preset.ContextLen+2 {
                s.mu.Lock()
                s.Status = StatusStopped
                s.mu.Unlock()
                return
        }

        lr := cfg.LR
        if lr <= 0 {
                lr = 0.001
        }
        bs := cfg.BatchSize
        if bs <= 0 {
                bs = 32
        }
        totalSteps := cfg.MaxSteps
        if totalSteps <= 0 {
                totalSteps = 300
        }

        model := newCharLM(vocab.Size, preset.ContextLen, preset.EmbedDim, preset.HiddenDim, lr, preset.FreezeBase)

        // LoRA: warm up base model before freezing
        if preset.FreezeBase {
                model.freezeBase = false
                for ws := 0; ws < 100; ws++ {
                        ctxs, tgts := sampleBatch(tokens, preset.ContextLen, bs)
                        model.trainStep(ctxs, tgts)
                }
                model.freezeBase = true
                // reset optimizer state for W2, b2 only
                model.opt.slots[3] = adamSlot{m: make([]float64, len(model.W2.d)), v: make([]float64, len(model.W2.d))}
                model.opt.slots[4] = adamSlot{m: make([]float64, len(model.b2)), v: make([]float64, len(model.b2))}
                model.opt.t = 0
        }

        // initial smoothed values
        smoothLoss := math.Log(float64(vocab.Size))
        smoothAcc := 1.0 / float64(vocab.Size)

        for step := 1; step <= totalSteps; step++ {
                select {
                case <-s.stop:
                        s.mu.Lock()
                        s.Status = StatusStopped
                        now := time.Now()
                        s.CompletedAt = &now
                        s.mu.Unlock()
                        return
                default:
                }

                ctxs, tgts := sampleBatch(tokens, preset.ContextLen, bs)
                loss, acc := model.trainStep(ctxs, tgts)

                alpha := 0.08
                smoothLoss = alpha*loss + (1-alpha)*smoothLoss
                smoothAcc = alpha*acc + (1-alpha)*smoothAcc

                rLoss := math.Round(smoothLoss*10000) / 10000
                rAcc := math.Round(smoothAcc*10000) / 10000

                s.mu.Lock()
                s.CurrentStep = step
                s.Progress = float64(step) / float64(totalSteps)
                if rLoss < s.BestLoss {
                        s.BestLoss = rLoss
                }
                s.FinalAcc = rAcc
                s.Metrics = append(s.Metrics, SessionMetric{
                        Step:     step,
                        Loss:     rLoss,
                        Accuracy: rAcc,
                        LR:       lr,
                        Ts:       time.Now().UnixMilli(),
                })
                if len(s.Metrics) > 500 {
                        s.Metrics = s.Metrics[len(s.Metrics)-500:]
                }
                s.mu.Unlock()

                time.Sleep(50 * time.Millisecond) // ~20 steps/sec
        }

        // Save completed model to registry
        name := cfg.Name
        if name == "" {
                name = fmt.Sprintf("%s-run-%d", cfg.ModelType, sm.seq)
        }
        params := int64(vocab.Size*preset.EmbedDim +
                preset.ContextLen*preset.EmbedDim*preset.HiddenDim + preset.HiddenDim +
                preset.HiddenDim*vocab.Size + vocab.Size)
        saved := sm.models.Create(&RegistryModel{
                Name:        name,
                Description: fmt.Sprintf("Trained on %s (%d steps, %s arch, lr=%.4f)", cfg.Dataset, totalSteps, cfg.ModelType, lr),
                Framework:   "Go-Native",
                Stage:       "REGISTERED",
                Accuracy:    math.Round(smoothAcc*1000) / 1000,
                Loss:        math.Round(smoothLoss*10000) / 10000,
                Parameters:  params,
        })

        s.mu.Lock()
        s.Status = StatusCompleted
        s.SavedModel = saved.ID
        now := time.Now()
        s.CompletedAt = &now
        s.mu.Unlock()
}

func sampleBatch(tokens []int, contextLen, batchSize int) ([][]int, []int) {
        maxStart := len(tokens) - contextLen - 1
        contexts := make([][]int, batchSize)
        targets := make([]int, batchSize)
        for b := 0; b < batchSize; b++ {
                start := rand.Intn(maxStart)
                ctx := make([]int, contextLen)
                copy(ctx, tokens[start:start+contextLen])
                contexts[b] = ctx
                targets[b] = tokens[start+contextLen]
        }
        return contexts, targets
}
