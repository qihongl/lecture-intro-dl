# Deep learning architectures

90 minutes including manually controlled demonstrations and discussion. 51 teaching slides; 1 optional reference slides. Formula reference pages are in a separate document.

## Presenter controls

Arrow keys / Space advance one build; left arrow reverses a build. R resets the current slide and its demo. G opens the slide index, N opens synchronized presenter notes, F enters full screen, E enables local text editing. Ctrl/Cmd+S in edit mode saves a self-contained edited copy. No demonstration runs automatically. All assets and fonts are embedded; internet is needed only to follow reference links.

## Timing

| Section | Time | Slides |
|---|---|---|
| Recap | 0–5 | 1–3 |
| Multilayer networks | 5–24.5 | 4–16 |
| Recurrent networks & temporal context | 24.5–42 | 17–25 |
| Gated recurrence | 42–51 | 26–30 |
| Convolutional networks | 51–68.5 | 31–39 |
| Transformers & attention | 68.5–87 | 40–49 |
| Consolidation | 87–90 | 50–51 |
| Optional reference | Outside the 90 min | 52–52 |

## Demonstration solutions

RNN: h0=0; input sequence [1,0,0,0]; h=tanh(x+0.8hprev) gives [0.761594,0.543617,0.409403,0.316293]. Negative cue reverses every sign.

Gate: g×0.8; g=0,0.5,1 gives0,0.4,0.8.

LSTM: previous c=0.8, candidate=−0.6. Retain → c=0.8,h=0.664037; partial → c=0.1,h=0.099668; replace → c=−0.6,h=−0.537050; hide → c=0.8,h=0. Endpoints0/1 are idealized.

GRU (Olah write convention): candidate=tanh(−0.6+0.8r×0.8); h=(1−z)×0.8+z×candidate. With z=0.5,r=1: candidate=0.039979,h=0.419989. With z=0, h=0.8 for every r; z=1 writes the candidate.

Convolution: a vertical stripe in column3 of a5×5 zero/one image; kernel rows [−1,0,+1]; stride1,no padding,no bias. Every row of the raw output is [3,0,−3]; ReLU gives [3,0,0]. Shift stripe to column4 → each raw row [0,3,0]. Horizontal kernel responds0 to this vertical stripe.

Attention: fourth query q=[1,0], keys=[[0,0],[2,0],[0,1],[1,0]], values=[[0,0],[1,0],[0,1],[0.5,0.5]], scaled by√2. Scores=[0,1.414214,0,0.707107]; weights=[0.122830,0.505229,0.122830,0.249112]; mixture=[0.629785,0.247386]. For query row1 (The), causal masking leaves only token1; its weight is1.

The semantic matrix, TCM overlap sketch, delay curve and two-head weights are teaching illustrations, not trained models or measured brain data. Published figures are preserved separately with source provenance.

## 1. Deep learning architectures

Recap · 0–0.5 min

Open with the title only. Today we ask what changes when we change the connections between familiar artificial neurons.

## 2. The same ingredients

Recap · 0.5–2.5 min

Build input circles, weighted connections, then the output. Recap x, w and y from the previous lecture; h will mean a hidden representation. No biological equivalence is implied. Ask students to explain the effect of a negative weight rather than count arrows.

## 3. Architecture changes the information flow

Recap · 2.5–5 min

Reveal the four sketches one at a time. LSTM and GRU belong inside the recurrence family. Explain that architecture constrains what information is accessible and what computations share weights. These are design choices, not a ranking from obsolete to best.

## 4. A hidden layer creates a representation

Multilayer networks · 5–6.5 min

Trace a single input forward. The highlighted paths indicate flow, not measured activation magnitudes. Each circle is an artificial unit. Remind students that the hidden layer is learned together with the output.

Question: If we freeze the hidden layer, what can training still change?

Expected answer: The output weights and bias can change. The hidden representation stays fixed.

## 5. A multilayer perceptron (MLP)

Multilayer networks · 6.5–7.5 min

Reveal the second hidden layer before the output. Do not name individual units as semantic features unless demonstrated. Multiple layers permit compositional nonlinear functions; more layers alone do not guarantee better generalization.

## 6. XOR: one straight boundary is not enough

Multilayer networks · 7.5–8.5 min

Recap the same four Boolean XOR cases as the introductory lecture: (0,0) and (1,1) target 0; (0,1) and (1,0) target 1. Coordinates label the two inputs; dot colors and numbers are target outputs. First show the points, then the example line at input sum 0.5. The side above the line predicts 1, so (1,1) is wrong. This is one illustration, not a proof by testing a single line. Explain that both classes occupy opposite corners: the line would need to keep both ends of one diagonal together while excluding both ends of the other. No single affine decision boundary can do so.

## 7. Two hidden units ask simpler questions

Multilayer networks · 8.5–9.5 min

Reuse the hand-set step-function network from the introductory lecture exactly. h1=step(x1+x2−0.5); h2=step(x1+x2−1.5); step(z)=1 for z≥0, otherwise 0. Both input weights are +1 in each hidden unit. Keep the complete two-input, two-hidden-unit, one-output architecture visible. Reveal the h1 decision region while highlighting h1 and its two incoming connections; then reveal h2 while transferring that highlight to h2 and its incoming connections. The output and its incoming connections remain muted until the next slide combines these decisions. Panel colors and dot numbers now indicate the hidden unit output, not the XOR target. h1 outputs 0,1,1,1 and h2 outputs 0,0,0,1 for (0,0),(0,1),(1,0),(1,1). Each hidden unit makes a simple straight-boundary decision; neither alone solves XOR. The step functions convert continuous weighted sums into distinct features. Exact equations are in the separate formula reference document.

## 8. Combine the hidden decisions to solve XOR

Multilayer networks · 9.5–10.5 min

Use the original output unit: predicted y=step(h1−2h2−0.5). Thus hidden (0,0) produces 0, (1,0) produces 1, and (1,1) produces 0. Keep exactly the same network layout and notation as the preceding slide. Reveal the lower h1 threshold while highlighting h1 and its path to the output, then reveal the upper h2 threshold while highlighting h2 and its path to the output. Finally reveal the class-1 region between them while highlighting the output and both hidden-to-output connections. The captions accept at least one, reject both, and exactly one appear with their corresponding stages. The network remains visible throughout. The final output is 1 exactly when 0.5≤x1+x2<1.5, solving all four Boolean cases. The lower boundary is included because step(0)=1; the upper boundary is excluded because h2 turns on there and suppresses the output. Filled regions show this hand-set model’s predictions on real-valued inputs; XOR itself is defined here only on the four Boolean cases. This is not a learned model and no uncertainty interval applies. Without nonlinear hidden activations, stacked affine computations reduce to one affine score, leaving the original XOR limitation unchanged. Spend about three minutes across this recap.

## 9. Learn this decision boundary with two hidden units

Multilayer networks · 10.5–13.5 min

Present only the target screenshot, task title and standard Playground link. Students choose the inputs, activation and training settings themselves; the assignment constrains the architecture to two hidden units. The link has no preset parameters. Instructor reference, to use only if students need help: one layer of two Tanh units, raw x1/x2 inputs, XOR data, learning rate 0.1, noise 0, no regularization, batch size 30, 50% training split, seed 0.70561. A prior recorded run at epoch 5257 gave training loss 0.142 and test loss 0.222 (outputs/playground-exercises/xor-two-units.jpg). This reference is an approximate diagonal band on the continuous Playground XOR dataset, not a perfect fit to every point. Do not require zero loss or disclose the reference settings before students explore. The provided screenshot is embedded without alteration. Three minutes are reserved in the 90-minute lecture; allow longer practice outside class if desired.

## 10. Semantic memory

Multilayer networks · 13.5–14.5 min

Use the four photographs and examples from the supplied lecture, slides 56–58. Begin with the pictures alone. Reveal their names, then shared properties. Knowing that a canary is a bird and can sing is semantic knowledge; remembering the particular occasion on which one first saw it is episodic memory. Keep that contrast in narration. These photographs are examples of concepts, not the visual inputs to the later localist-item network. Source deck: https://docs.google.com/presentation/d/1Rp3A6j6XdjeE8fW5WssMQgnmg0nnPFt4hB4SoqHLpQQ/edit . Definition: Tulving (1972), summarized by Tulving (1987), https://garfield.library.upenn.edu/classics1987/A1987K743800001.pdf .

## 11. Concepts have a similarity structure

Multilayer networks · 14.5–16.5 min

Combine the earlier concept-space illustration with the new-bird generalization example. Build 0: show the familiar birds and flowers in the same schematic space. Build 1: reveal only the sentence Similar concepts evoke similar neural patterns on the right. Build 2: reveal the new bird photograph beneath that sentence, without a label or species name; invite students to apply their existing bird knowledge. Build 3: replace the left illustration with a large excerpt of the supplied news screenshot, preserving its headline, date and introductory paragraph. Keep the new bird photo and the sentence visible. The screenshot is from Natali Anderson, New Songbird Species Discovered in Bolivia, Sci.News, January 5, 2026: https://www.sci.news/biology/beni-greenlet-hylophilus-moxensis-14464.html . The Beni greenlet, Hylophilus moxensis, was formally described in 2026; the population had been observed decades earlier. Photograph credited in that article to Tini Wijpkema. The photo comes from the supplied lecture and the screenshot is the user’s attachment. Photo positions stand schematically for activity patterns, not measured brain responses or a numerical simulation. Each illustrative axis is one unit’s activity. Similarity supports expectations about a new bird but does not guarantee every bird flies or sings. The later eight-item Rumelhart model uses localist item inputs and does not directly accept an unseen bird photograph; generalization to novel items requires shared feature inputs or a learned mapping into the representation. Source lecture: https://docs.google.com/presentation/d/1Rp3A6j6XdjeE8fW5WssMQgnmg0nnPFt4hB4SoqHLpQQ/edit , slides 59–61 and 91–94.

## 12. Broad and fine semantic distinctions

Multilayer networks · 16.5–17.5 min

Use the bird/flower concept-space figure from supplied slides 62–64 to motivate a model. The sketch is an illustration, not infant neural recordings. Mandler and McDonough (1993) studied conceptual categorization in 7–11-month-olds: 9- and 11-month-olds differentiated animals from vehicles, but did not differentiate dogs from fish or rabbits in that task. Do not attribute the exact robin/canary versus animal/plant picture to that experiment, and do not claim that infants universally lack fine perceptual discrimination. Their paper explicitly contrasts conceptual and perceptual tasks. https://doi.org/10.1016/S0885-2014(93)80003-C . Rogers et al. (2004) discuss selective progressive loss of conceptual knowledge: distinctive information can be more vulnerable than broad shared information. https://web.stanford.edu/~jlmcc/papers/RogersETAL04.pdf . Their recurrent semantic-dementia model is not identical to the feedforward Rumelhart architecture taught here. The point is a cognitive pattern for a learning account to explain, not proof that this exact network is the human mechanism. Source deck: https://docs.google.com/presentation/d/1Rp3A6j6XdjeE8fW5WssMQgnmg0nnPFt4hB4SoqHLpQQ/edit .

## 13. Semantic knowledge as a prediction task

Multilayer networks · 17.5–19.5 min

Published Figure 2.2, printed p.31, author manuscript PDF p.43. This is Rogers and McClelland’s adaptation of the Rumelhart feedforward model, with the entire set of units shown. Item inputs are localist: one unit for each of eight items, with the selected item set to 1 and the others set to 0. A separate input pool selects one of four relations: ISA, is, can, has. The item pool projects to a learned distributed representation layer; this representation and the relation pool project to the hidden layer, which predicts attributes. Where connections are indicated, all units in the source pool connect to all units in the destination pool. The black item and relation units indicate the canary/can query; black attribute units show its target completions grow, move, fly and sing. Do not interpret the item unit as the model’s only input or as one biological canary neuron. Source caption contains an apparent robin/canary typo; the selected unit is labelled canary in the figure. Preserve the original diagram unchanged.

Source: Rogers & McClelland (2004), Semantic Cognition · Chs. 2–3 · Figure 2.2

## 14. Read a distributed representation

Multilayer networks · 19.5–20.5 min

Use the illustrative matrix only to explain how to read the forthcoming original figure. Values are hand specified, not trained or biological recordings. Two plants share a pattern; two birds share a different one. A unit can contribute to multiple items. This matrix previews the Representation layer in the Rogers–McClelland architecture, not its separate later Hidden layer. The forthcoming source figure shows eight learned Representation-layer units.

Question: Does a similar pattern require one “plant neuron”?

Expected answer: No. Similarity can be distributed across several units.

## 15. Learning separates semantic structure

Multilayer networks · 20.5–23 min

Published Figure 3.1, book p.47, author manuscript PDF p.59. Rows: eight items; bars: activations of the eight units in the Representation layer of Figure 2.2, not the later Hidden layer or the Attribute outputs; columns: epochs 50, 100, 150. Activate one Item input at a time; each row is the resulting representation pattern. The ordering of the eight units stays fixed across rows and epochs. Reveal the three columns in sequence. Focus on pine/oak versus rose/daisy. Deterministic model snapshots, not participant means; no confidence interval applies. Consolidate the former semantic-summary slide here: learning a shared property-prediction task organizes conceptual similarity. The item input is localist, while the learned representation is distributed. This is a computational account, not a measurement of human neural activity, and later recurrent semantic models should not be conflated with this feedforward model.

Source: Rogers & McClelland (2004), Semantic Cognition · Chs. 2–3

## 16. Learning differentiates concepts gradually

Multilayer networks · 23–24.5 min

Include the user-supplied 2D and 3D MDS trajectory figure from slides 86–87, preserving all curves, colors, axis labels and the shared legend. It is a model illustration from the supplied lecture; no new simulation is being claimed. This classroom simulation includes Shark rather than the book figure’s Sunfish and uses a different training schedule; do not present it as the exact same run as Rogers and McClelland Figure 3.1. The plotted activity is labelled Hidden Representation in the source simulation. MDS means multidimensional scaling: its coordinates summarize the similarities between many-unit representation patterns. These axes are summary coordinates, not individual neurons and not training epochs. Each colored curve traces one concept’s representation across training snapshots. Initially show the 2D view and legend. Reveal the 3D view with the next click, then explain the interpretation. The original GIF is a rotating view of fixed trajectories; do not autoplay it or present rotation as learning. The image starts with a common red cluster and shows separation into broader and finer distinctions. The curves do not establish a literal measurement of infant brain development. Source deck: https://docs.google.com/presentation/d/1Rp3A6j6XdjeE8fW5WssMQgnmg0nnPFt4hB4SoqHLpQQ/edit , slides 86–87; attached figure supplied by the user. See also Saxe, McClelland and Ganguli (2019), https://doi.org/10.1073/pnas.1820226116 .

## 17. The same word can need different context

Recurrent networks & temporal context · 24.5–26 min

Begin with bank alone and pause to let students notice the ambiguity. First reveal river on the left, its connection to bank, and the river sentence together. Then reveal money on the left with a second bank and the money sentence. The final build states what the preceding context contributes. These are everyday teaching examples, not the sentences in Elman’s experiment. Invite students to explain what information a memoryless word classifier would be missing.

## 18. Recurrent neural network (RNN)

Recurrent networks & temporal context · 26–28 min

Start with Olah’s rolled RNN, then show the single-tanh repeating module. Introduce h as a vector of hidden-unit activities. A recurrence does not mean copying the raw sentence. Distinguish stored activity from trained weights, which do not change on each forward step. Use Olah’s rolled RNN first, then his single-tanh repeating module. In these diagrams h is the hidden state; a separate prediction head is omitted. A denotes the repeated recurrent computation, not an individual neuron.

Source: Christopher Olah (2015) · Understanding LSTM Networks

## 19. Unroll the network through time

Recurrent networks & temporal context · 28–30 min

Reveal three time steps manually using the same authored architecture and node positions as the later Elman example. Every column is a moment, not a new layer with new weights. Inputs x1, x2 and x3 appear above hidden states h1, h2 and h3; prediction vectors appear below. The separate h0 node is the initial hidden state before x1, consistent with the numerical demo and the Elman example. Each circle represents a vector, not one biological neuron. The hidden-to-hidden links carry the previous state; recurrent and prediction weights are reused at every time step. The Elman slide adds word labels and next-word targets to this identical diagram. Shared weights make variable-length sequences possible. This is an authored schematic of recurrence, not a reproduced source figure.

Source: Christopher Olah (2015) · Understanding LSTM Networks

## 20. Watch a context update

Recurrent networks & temporal context · 30–32.5 min

A one-dimensional, hand-set RNN: h0=0, h_t=tanh(x_t+0.8h_(t−1)); no bias. Step through +1, 0, 0, 0, then use Change first input to repeat with −1, 0, 0, 0. After the first 0, the hidden state is approximately +0.544 or −0.544: the same current input produces different states because the first input changed the previous state. tanh outputs between −1 and 1. Explain tanh as a smooth response that stays between −1 and 1; no derivative is needed. This is an exact forward calculation, not a trained language model. Spend time reading the plot axes and the numeric equation.

Question: After input 0, is the state the same if we started with +1 or −1?

Expected answer: No. The first input still affects the hidden state.

## 21. Finding word boundaries without pauses

Recurrent networks & temporal context · 32.5–34 min

Saffran, Aslin & Newport (1996), Statistical Learning by 8-Month-Old Infants, Science 274:1926–1928. https://pubmed.ncbi.nlm.nih.gov/8943209/ ; doi:10.1126/science.274.5294.1926. Begin with the continuous orthographic example bidakupadotigolabubidaku from the paper. The infants heard synthesized continuous speech with no pauses, stress differences or other acoustic boundary cues. Build 1 expands the first two three-syllable words to show next-syllable predictability: within-word transitions 1.0; between-word transitions 0.33 in experiment 1, where four words occurred in random order with no immediate repetitions. Percentages summarize the familiarization corpus, not observations from the short displayed snippet or a fitted neural network. Build 2 marks bidaku and padoti as the underlying word groups; the marks are explanatory overlays, not pauses in the auditory input. Twenty-four infants in experiment 1 distinguished familiar words from nonwords. A separate group of twenty-four infants in experiment 2 distinguished words from part-words spanning a word boundary, listening longer to part-words (7.60 versus 6.77 seconds); this is the test finding on the slide. Experiment 2 used different four-word inventories in its two counterbalanced conditions. The tested part-words had a boundary transition probability of 0.33, followed by a within-word transition of 1.0. Do not present the illustrated experiment-1 sequence as its exact test material. Do not claim that the study demonstrated an Elman network in infants, that all natural boundaries have lower transition probability, or that statistical learning is the only cue to natural speech segmentation. Transition: how might a network learn such sequential regularities? The next slide introduces Elman’s prediction-learning approach. The next-word example is from Elman’s separate artificial-sentence experiment, not a simulation of Saffran’s infant study.

Source: Saffran, Aslin & Newport (1996) · Statistical Learning by 8-Month-Old Infants

## 22. Elman: predict the next word

Recurrent networks & temporal context · 34–36 min

Reuse the exact unrolled architecture, node positions, connections and x/h/ŷ notation introduced on the earlier unrolling slide. Connect the preceding infant example to the general question of learning sequence structure through prediction; this is a separate modeling example, not a reproduction of that infant experiment. Training compares next-word predictions with the following word and updates the shared weights. Initially show that familiar architecture. Build 1 adds boy, chases, dog above the input vectors; build 2 adds next-word targets under the output vectors; build 3 highlights the hidden-state row to introduce the forthcoming representation analysis. The illustration is not an assertion that these predictions are already correct. The third target is omitted with an ellipsis because the continuation is not specified. Each circle denotes an input, hidden-state or output vector, not one unit for an entire word. Elman’s original context units copy the previous hidden activities; the unrolled h-to-h links express the same memory dependency, with the copied state implicit. This notation does not claim that the original sigmoid model uses the toy tanh demo’s parameters or modern softmax. Elman’s 1990 paper has several demonstrations; focus on the simple-sentence word prediction experiment. Artificial sequences provided regularities without noun/verb category labels as targets. This is a cognitive modeling example, not a brain-recording experiment.

Source: Elman (1990), Finding Structure in Time · Cognitive Science

## 23. Structure appears in hidden states

Recurrent networks & temporal context · 36–38 min

Published Figure 7, article p.200 (PDF p.22). This is a hierarchical clustering tree of words, not a network wiring diagram, brain anatomy, or clusters of individual hidden units. After training the simple recurrent next-word prediction model, freeze its weights and run the word stream through it. For every word occurrence, record the activations of the 150 hidden units after processing that word in context. Average those vectors separately for each of the 29 word types; then hierarchically cluster the 29 mean activation vectors. Leaves label the words that produced the vectors. Joins closer to the word labels indicate more similar average patterns; joins farther left merge more dissimilar clusters. Read mouse/cat and woman/girl as concrete examples. The unit order is the same in every vector. Noun and verb category annotations were not the next-word training targets. The original paper cautions that averaging can induce clustering artifacts; individual-token analyses are needed too. Figure is preserved, not a new clustering calculation.

Source: Elman (1990), Finding Structure in Time · Cognitive Science

## 24. Human recall favors nearby studied items

Recurrent networks & temporal context · 38–39.5 min

Introduce the human free-recall finding before the temporal context model. Participants study a list and later recall its items without being required to preserve study order. One original curve is shown: Howard & Kahana Figure 1, panel E, reproduced from Kahana (1996). This curve collapses over conditions of Murdock & Okada (1970). The other five panels are omitted. Axis labels are repositioned for this single-panel excerpt; plotted points, connecting lines, ticks, and error bars are retained from the source. It is data motivating TCM, not a plotted fit from our demo. Lag zero is excluded. Original error bars are retained with their original meaning; do not relabel them as newly computed 95% CIs. Source caption identifies the conditions. After explaining the observed nearby-item advantage, transition with: how could a changing internal context produce this pattern? The next slide introduces the temporal context mechanism. The horizontal variable is signed study-list position lag, not elapsed time or the order of earlier recall responses. Recalling study item 5 then study item 6 gives lag +1; item 5 then item 4 gives lag −1. Positive lags move forward through the studied list. The vertical axis is conditional response probability among available transitions. Point out the forward asymmetry after explaining the axes.

Source: Howard & Kahana (2002), Temporal Context Model · J. Mathematical Psychology

## 25. Temporal context explains nearby recall

Recurrent networks & temporal context · 39.5–42 min

Start from the human result on the preceding slide: why does recalling one item favor recall of its study-list neighbors? Howard and Kahana’s TCM: encoding and retrieval both change context; reinstated context helps cue neighboring list items. Teaching sketch shows apple, chair, river, lamp with overlapping contexts; it is not a full implementation or fitted recall simulation. Shared evolving-state idea links this to recurrence, but TCM has its own memory and retrieval mechanisms. Within a group, each bar position denotes a context component. Strength is encoded by darkness, not height; the bars have equal height. The same component positions are used across all four groups. Reveal the context vectors, explain their overlap for nearby studied items, then recall river and show its reinstated context cueing neighbors. Link the mechanism explicitly back to the human lag-CRP curve. This completes the four-minute data-to-model sequence. The simplified sketch explains the proximity intuition; it does not by itself derive the forward asymmetry or the numerical human curve. In the full model, retrieved context is an asymmetric retrieval cue, as explained in Howard and Kahana (2002).

Question: If recalling “river” brings back its context, which items gain a useful cue?

Expected answer: Items studied near “river” share more of that context.

Source: Howard & Kahana (2002), Temporal Context Model · J. Mathematical Psychology

## 26. Keeping a cue across a delay

Gated recurrence · 42–44 min

Distinguish two difficulties: forward state retention and backward credit assignment. The y-axis is the activation of one illustrative memory unit: the cue sets h₀ = 1, then hₜ = 0.65 × hₜ₋₁ while no new input enters. The unit is a simple linear recurrent unit with recurrent weight 0.65 and zero bias. A value of 1 is its starting activation; 0.5 is half that activation; 0 is no remaining cue signal in this scalar unit. This is an arbitrary activation scale, not biological firing rate or percentage of facts remembered. The x-axis counts time steps since the cue. This hand-set scalar decay illustrates only retention; it is not recall probability, measured human memory, a trained RNN result, or a training gradient. Vanishing gradients arise from repeated derivative factors during training, not simply because memory fades. Do not say all RNNs necessarily forget everything.

## 27. A gate is a learned amount

Gated recurrence · 44–45.5 min

Use a circle and arrow for the gate computation. Move g between 0, 0.5 and 1: g×0.8 yields 0, 0.4, 0.8. Explain that this is a continuous learned amount, not a physical switch. The slider is a manual intervention, not gate training.

## 28. Long short-term memory (LSTM)

Gated recurrence · 45.5–48 min

Use Olah’s highlighted diagrams in order: cell update path, forget gate, input gate and candidate, cell update, output gate. Main-slide crops retain the complete cell diagram and omit the dense equations beside it. Yellow boxes are learned neural-network layers; pink circles are elementwise operations. Lines carry vectors; merging means concatenation and branching means copying. Explain these symbols as each becomes relevant. Olah uses uppercase C for cell state; the scalar demo uses lowercase c for the same quantity. Modern forget-gate LSTM, not the exact original 1997 cell. The original LSTM is Hochreiter & Schmidhuber (1997); forget gates were introduced later (Gers et al., 2000). c and h are vectors. Multiplication is elementwise. A hidden state is not a biological neural firing model.

Question: Can a stored cell value differ from the visible hidden output?

Expected answer: Yes. The output gate controls how much of tanh(cell) is exposed.

Source: Christopher Olah (2015) · Diagram; Hochreiter & Schmidhuber (1997); Gers et al. (2000)

## 29. Preserve a value, then replace it

Gated recurrence · 48–49.5 min

Use controlled presets: retain (f=1,i=0,o=1), partial write (f=0.5,i=0.5), replace (f=0,i=1), hide output (f=1,i=0,o=0). Fixed previous c=0.8 and candidate=−0.6. Values 0 and 1 are idealized gate endpoints; finite sigmoid logits approach rather than reach them. This is an arithmetic illustration, not an LSTM trained on a delay task.

## 30. Gated recurrent unit (GRU)

Gated recurrence · 49.5–51 min

Use Olah’s write convention: h=(1−z) h_prev+z candidate. z near 0 retains old state, and z near 1 writes the candidate. This complements the original Cho retention variable; do not mix the two conventions. The original diagram’s cell is shown intact, while its adjacent equations are reserved for the reference slide. Here candidate=tanh(x+0.8 r h_prev), x=−0.6, h_prev=0.8. Reset affects candidate computation; it does not directly erase the old-state route. The gates are manually controlled; actual GRUs learn their gate-producing weights. Keep the main numerical display focused on old state, candidate and their blended result. The candidate computation belongs in the separate formula reference. To test the question, set z=0, vary r, and observe h remains 0.8 while the candidate may change.

Question: Set writing to 0. Does changing reset change the stored state?

Expected answer: No. With writing at 0, the old state is retained regardless of the candidate.

Source: Christopher Olah (2015) · Diagram; Cho et al. (2014)

## 31. Same bird, different location

Convolutional networks · 51–51.5 min

Introduce the convolution section with the robin photograph used earlier in the semantic-memory slides. Show exactly the same crop and size at three different positions: left, upper right, then lower center. Each manual advance moves the photo; previous reverses the move. Leave only one instance visible so students compare position without a second changing property. Students can still recognize the bird at these different locations. Use this visual intuition to motivate recognizing a local pattern wherever it occurs; the next slide simplifies that problem to a moving vertical stripe. The photo includes its original background, so this is a translated photograph, not a bird segmented onto novel scenes. It illustrates robust recognition, not measured neural invariance or a claim that all CNNs are perfectly translation invariant. Source photograph is from the supplied lecture, as recorded in semantic-user-provenance.json.

## 32. A visual pattern can move

Convolutional networks · 51.5–53 min

Compare two binary images with a shifted vertical stripe. Avoid claiming complete translation invariance: convolution is approximately translation equivariant away from boundary effects, and classifier robustness depends on architecture/training.

## 33. One patch produces one number

Convolutional networks · 53–56 min

Start with the 5×5 binary image and its highlighted 3×3 window. Build 1 extracts the patch, places it beside the fixed filter, and shows all nine cellwise products. The blue box identifies the corresponding entry in all three grids; use Next cell or click a cell to trace its contribution. Build 2 adds those nine products into one output. Change the patch to encounter a positive response, zero response and negative response. Input intensity is grayscale; blue and orange indicate signed numerical contributions. The filter is hand specified, stride 1, no padding, no bias. This implements the cross-correlation convention used by CNN libraries. The brief outline movement is lecturer-triggered and respects reduced-motion settings.

## 34. Build a feature map one position at a time

Convolutional networks · 56–59 min

The filter weights stay fixed. At build 1 the first output is revealed. Next position slides the 3×3 window by one pixel and fills exactly one corresponding cell; previously filled cells remain visible. Previous position reverses this sequence. Reveal map fills all nine positions at once if needed. Restart clears the progress back to the first position. Build 2 adds ReLU beside the raw map so negative responses visibly become zero. Changing or shifting the input resets the scan. For the original vertical stripe, each raw row is [3,0,−3] and each ReLU row is [3,0,0]. After shifting right, raw rows are [0,3,0]. After the next shift the stripe reaches the right edge; repeated shifts may remove it. Blank output cells have not yet been visited. The source image, active window and highlighted output remain linked.

## 35. Two filters, two different feature maps

Convolutional networks · 59–61 min

Use the same input image for the vertical and horizontal filters. Reveal the vertical branch at build 1 and horizontal branch at build 2. Each filter has one separately labeled raw response map; no ReLU is applied on this comparison slide. A vertical stripe produces rows [3,0,−3] for the vertical filter and zeros for the horizontal filter. A horizontal stripe reverses their roles: the horizontal map has rows [3,3,3], [0,0,0], [−3,−3,−3]. Use the Vertical, Horizontal and Corner presets, or edit individual input pixels by clicking the grayscale image. Every edit recomputes both maps. These filters detect the shown polarity of an edge; a sign reversal gives a negative response. Training would normally learn the filter values. This is an exact teaching demonstration, not a trained vision model.

## 36. AlexNet: a convolutional neural network

Convolutional networks · 61–63 min

Redrawn from the instructor-provided architecture diagram in Ignacio Quintero’s AlexNet blog post: https://medium.com/@igquinteroch/understanding-alexnet-the-2012-breakthrough-that-redefined-ai-d0e267e2470a . Build 0: RGB input. Build 1: five convolutional feature-map volumes and pooling after Conv1 and Conv2. Build 2: pooling after Conv5 and the three fully connected layers. Build 3: the ImageNet result. Dimensions are height × width × channels; FC numbers count units. Volumes depict stacks of spatial feature maps, not individual neurons, and their drawn proportions are schematic. The 227-pixel input convention follows the supplied diagram; with an unpadded 11×11 stride-4 first convolution it produces 55×55 maps. The original paper reports 224-pixel crops; implementations differ in input/padding conventions. Max pooling reduces 55→27, 27→13 and 13→6; final pooled representation is 6×6×256. ReLU, normalization, dropout and two-GPU grouping are omitted from the overview. Source: Krizhevsky, Sutskever & Hinton (2012), ImageNet Classification with Deep Convolutional Neural Networks, https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf . Five convolutional plus three fully connected learned layers. The winning 2012 entry was an ensemble of AlexNet-family networks, not one network in isolation; its top-5 test error was 15.3%. Top-5 error means the correct class is absent from the five highest-ranked predictions. The result helped establish large trained CNNs as a practical approach to image recognition.

Source: Krizhevsky, Sutskever & Hinton (2012) · ImageNet classification with deep CNNs

## 37. Compare a visual model with cortex

Convolutional networks · 63–64.5 min

Yamins et al. compared hierarchical convolutional models with macaque IT responses; also studied V4. Freeze the task model for the neural comparison, fit a separate linear readout, and test on novel images. Readout fitting uses neural training data even though task optimization need not. Two distinct goals: recognize objects versus predict neural responses. The anatomical human-brain image is a generic symbol for the neural-recording branch, not measured activation data or a localization map. Image: NIAID Visual & Medical Arts, Ryan Kissinger (2024), Brain Lateral, NIH BioArt BIOART-000060, public domain, courtesy of NIAID; https://bioart.niaid.nih.gov/bioart/60. The Yamins visual study used macaque recordings despite this generic human-brain icon. Follow two branches from the same image: model → representations → predicted responses, and participant brain → recorded responses. Predictions and recordings meet at Compare; neither branch generates the other. Fitting uses training images and evaluation uses different images.

Question: Why test the readout on different images?

Expected answer: To test generalization instead of measuring how well it fits the training images.

Source: Yamins et al. (2014) · PNAS · doi:10.1073/pnas.1403112111

## 38. Task performance and IT predictivity

Convolutional networks · 64.5–67 min

Published Figure 1A–B, p.8620. The figure fills the slide; there is no right-hand explanatory text. Explain aloud: right means better categorization, up means more IT response variance explained, and each point is a different model. Focus first on A; B extends the comparison to other models. x=balanced categorization accuracy, chance0.5; y=median explained variance across168 IT sites. Green=random selection; blue=categorization optimization; orange=IT fitting optimization. In B, red squares trace HMO optimization, black circles are controls/published models, and light-blue circles repeat task-optimized models. The purple category ideal observer is not an image-computable model. Reveal axes, then discuss trend. Preserve all original color coding and annotations. No newly calculated uncertainty. This result concerns tested hierarchical model families.

Source: Yamins et al. (2014) · PNAS · doi:10.1073/pnas.1403112111

## 39. What the visual-cortex result supports

Convolutional networks · 67–68.5 min

The bounded statement matters: not a theorem that any optimization, task or architecture always increases brain predictivity. Distinguish correspondence after a fitted readout from a neuron-to-unit identity. Note intermediate model layers were useful for V4, without adding another dense figure.

Source: Yamins et al. (2014) · PNAS · doi:10.1073/pnas.1403112111

## 40. Transformers: represent a word using its context

Transformers & attention · 68.5–69.5 min

Introduce the goal before naming architectural variants: a word is represented using the words around it. Reuse the same four word-like tokens as the attention walkthrough. Begin with the words, reveal their starting vectors on build 1, then on build 2 show information flowing into the highlighted final word and an illustrative changed vector pattern. Only incoming context for this one selected word is drawn; this is not a full attention graph or proof that all transformers use only preceding words. Bar colors are hand-set illustrations, not a trained model or a computed attention output. The example does not claim that it refers to cat or that this is a coreference detector. Attention supplies a learned route for combining contextual information rather than carrying it only through successive recurrent states. The next slide introduces query/key/value computations. GPT-2 adds learned position embeddings before the first block; position information is implicit in this overview. Vector components have learned roles rather than hand-assigned meanings. Reserve the word causal for the later next-word-prediction slide, where the no-peeking rule is visible and explicitly defined. Word-like tokens are used for readability; actual language models often use subword tokens. Source: Vaswani et al. (2017), https://arxiv.org/abs/1706.03762 . A GPT-style causal model has a finite context window. Training can process sequence positions in parallel while autoregressive generation proceeds token by token.

## 41. A token asks for useful context

Transformers & attention · 69.5–72 min

Show a representation vector under each token. For this exact toy calculation the four starting representations are different four-component one-hot basis vectors, so the shared learned linear projections can produce the shown two-component Q, K and V vectors. These basis vectors are an illustrative input to this head, not GPT-2 embeddings. A shared projection module produces a query, key and value for every token. Reveal the three rows in order. Highlight the selected token and its query in blue; its other projections remain in their own token column. Changing the dropdown selects the query used in the next calculation; it does not turn the other tokens into non-query tokens. Queries, keys and values are token-dependent vectors, not the projection weights. Future tokens can have projections, but the next slides explicitly exclude them when forming the selected causal mixture. These are three learned linear projections, with normalization omitted from this teaching diagram. The colored vector strips schematically indicate components, not trained activations. See Transformer Explainer, https://poloclub.github.io/transformer-explainer/ . These verbal descriptions are mnemonics, not literal introspection by the network. In this toy head, all numbers are hand specified, not GPT-2 weights. The arithmetic is exact; the semantic interpretation is illustrative.

## 42. Turn match scores into a weighted mixture

Transformers & attention · 72–74.5 min

Default query=token it (fourth token) in The / cat / saw / it. q=[1,0], k=[[0,0],[2,0],[0,1],[1,0]], values=[[0,0],[1,0],[0,1],[0.5,0.5]]. Score=q·k/√2. Weights=[0.122830,0.505229,0.122830,0.249112]; they sum to1. The mixture is[0.629785,0.247386]. Reveal the scaled match scores, then the softmax weights, then the individual weighted contributions and their sum. Each column retains the same token throughout. The contribution from token j is weight[j] times its value vector, component by component. Values are rounded to three decimals for display; the sum uses full precision. Excluded future tokens have exactly zero contribution. Change query and observe recomputed weights, contributions and sum. This teaching example already applies the causal rule, which is unpacked on the next slide. This is not a trained coreference detector.

Question: Does the output simply copy the value for “cat”?

Expected answer: No. It combines the available values according to their weights.

## 43. Next-word prediction: no peeking ahead

Transformers & attention · 74.5–76 min

Now introduce the term causal: the representation at a position cannot use later inputs. This prevents access to the answer while learning to predict the next token. Read matrix rows as queries and columns as keys. A causal mask is triangular. Start with the cat query row to match the next-word prediction question; saw and it are then unavailable. Choose another row to reveal its available positions. Masked score=−∞ means zero weight after softmax, not a negative contribution. Token representations at time t predict token t+1. Don’t conflate a mask with a sigmoid memory gate.

Question: When predicting after “The cat”, may the model inspect “saw”?

Expected answer: No. That future token must be hidden for this prediction.

## 44. Attention is one part of a block

Transformers & attention · 76–78 min

Replace the former unit-like Attn and MLP circles with rectangular processing modules. Show H as a group of four token-vector strips, with the same positions at the input and output. Arrows in this diagram carry the whole sequence of representations, not a single scalar. Build attention first, then a bank of four independent MLP applications, then residual paths and the output to the next block. The four small MLP boxes depict applications of the same feedforward parameters at different positions; they are not four independently trained MLPs. Residual plus circles are elementwise vector additions across the sequence. Both attention and feedforward updates are added to the representation entering their own sublayer. Explain feedforward parameters are shared across positions but no token mixing occurs inside that sublayer. This diagram simplifies GPT-2’s pre-normalization block. Repeat blocks to form depth; finish with a vocabulary prediction head. Optional equations make the order precise. Keep normalization in presenter notes: layer normalization rescales components within each token representation and has learned scale and offset parameters. It is omitted from the visible simplified diagram. Focus the main explanation on the residual addition that is actually shown.

Source: Vaswani et al. (2017); Radford et al. (2019)

## 45. Several heads, several learned mixtures

Transformers & attention · 78–78.5 min

Two teaching mixtures over the same tokens illustrate distinct weights, not measured GPT head behavior. Do not label one head syntax and another semantics as an empirical fact. Attention weights alone are not a causal explanation of model behavior. The selected query is the token “it”. Each bar is an attention weight for the token directly above it; each head’s weights sum to 1. Blue and orange distinguish heads consistently. The final build shows one output vector for each head, joins those outputs, and applies a learned output projection to produce one attention update. The vector-strip components are schematic and are not computed from the displayed weight rows; each real head also has its own value projection. The output projection can mix and rescale components; it is not simply an average of the head outputs. Both heads attend to the same available tokens but use different learned projections. The two colored rows are separate heads, not different time steps.

## 46. Transformer demos

Transformers & attention · 78.5–82 min

Use Transformer Explainer as the main live class demo: https://poloclub.github.io/transformer-explainer/ . The preview is an unaltered screenshot of its preset example captured on October 14, 2026, while the local GPT-2 download was pending; it is not a claim that custom inference was tested. Open the site ahead of class and allow its model download to finish. Start from its preset sentence, inspect one query row and change the selected head; then connect attention output to the token-wise MLP and next-token probabilities. GPT-2 small is a causal decoder-only model. Use the 3Blue1Brown video, Attention in transformers, step-by-step (Deep Learning Chapter 6), https://www.youtube.com/watch?v=eMlx5fFNoYc , for the geometric intuition. Optional BertViz, https://github.com/jessevig/bertviz , allows attention-head inspection; choose a GPT-2 causal example when comparing with this walkthrough. Its familiar BERT example is bidirectional and is not the same mask as GPT-2. BertViz: Vig (2019), https://aclanthology.org/P19-3007/ . Attention weights alone do not establish a causal account of the model prediction. All three links are clickable in the HTML and printable PDF. Three and a half minutes are reserved for the live demonstration, including the time freed by removing the separate embedding slide, so the complete lecture remains 90 minutes. Screenshot courtesy of the Georgia Tech Polo Club Transformer Explainer team; the tool is linked directly rather than embedded so the slide does not depend on a network connection to load.

Source: Transformer Explainer · Georgia Tech Polo Club

## 47. Compare language models with brain responses

Transformers & attention · 82–83.5 min

Schrimpf et al. benchmarked43 language models against two fMRI datasets and one ECoG dataset; also reading times. Sentence/word aggregation and fitting differ by dataset. This diagram teaches the shared comparison logic, not a claim all stimuli were identical across studies. Neural readout fitting is distinct from language-model pretraining. The anatomical human-brain image is a generic symbol for the neural-recording branch, not measured activation data or a localization map. Image: NIAID Visual & Medical Arts, Ryan Kissinger (2024), Brain Lateral, NIH BioArt BIOART-000060, public domain, courtesy of NIAID; https://bioart.niaid.nih.gov/bioart/60. Use separate model and participant branches. Compare is where predicted responses meet recorded responses. The model does not produce or cause the participant recording. Fit the readout on training stimuli and evaluate it on held-out stimuli.

Source: Schrimpf et al. (2021) · PNAS · doi:10.1073/pnas.2105646118

## 48. Next-word performance and brain predictivity

Transformers & attention · 83.5–86 min

Published Figure3A, article p.5. The x-axis decreases left→right; make this explicit before discussing trend. y is normalized neural predictivity aggregated across three datasets. r=0.44 in the plotted aggregate. The inset separates datasets: ECoG association positive but not significant. Preserve all points and inset. The paper identifies error bars as median absolute deviation over subject scores (Fig.2 caption: here and elsewhere), not95% CIs; do not relabel them. Normalization by a noise ceiling does not mean all raw neural variance is explained. Explain perplexity as the model’s average uncertainty about the next word: a lower score means it assigns more probability to the words that actually follow. It is not classification accuracy. The plotted neural scores are normalized relative to an estimated noise ceiling and aggregated across datasets; keep that technical detail in narration rather than adding another visual definition.

Source: Schrimpf et al. (2021) · PNAS · doi:10.1073/pnas.2105646118

## 49. A predictive model is a testable hypothesis

Transformers & attention · 86–87 min

Some untrained models also had substantial neural predictivity after fitting a readout. This limits a training-only explanation. Treat the work as evidence about candidate representational computations, not direct identification of the brain algorithm or a claim of biological self-attention.

Source: Schrimpf et al. (2021) · PNAS · doi:10.1073/pnas.2105646118

## 50. Choose the information flow

Consolidation · 87–88 min

Build the comparison rows. Architecture is often combined in practice; the rows describe characteristic operations, not exclusive application domains. Every family learns parameters and needs generalization checks. Ask students to describe the information each operation makes available.

## 51. Neural networks as model organisms

Consolidation · 88–90 min

Close the lecture with the two-way exchange between AI and neuroscience. The minute freed by removing the separate closing-question slide is reserved for explaining this figure. Start with an explicit multilayer neural network and the anatomical human brain. Build 1 reveals the blue network-to-brain arrow: Test brain hypotheses. Neural networks are tractable computational test beds: vary architecture, learning objectives or experience; inspect internal representations; perturb units or connections; compare behavioral and neural predictions with independent biological evidence. The phrase model organism is an analogy to experimental systems, not a claim that an artificial network is a biological organism or an exact copy of the brain. Build 2 adds the orange brain-to-network arrow: Inspire better AI. Computational principles learned from neuroscience can suggest architectures, learning rules and memory mechanisms; whether these improve AI must be tested. The two directions describe scientific exchange, not physical causation between the depicted objects. The network drawing is schematic, not a trained model or measured activity. Keep the on-slide wording to the title, two object labels and two arrow captions; narrate the details. References: Richards et al. (2019), A deep learning framework for neuroscience, Nature Neuroscience 22:1761–1770, https://doi.org/10.1038/s41593-019-0520-2 ; Hassabis, Kumaran, Summerfield & Botvinick (2017), Neuroscience-Inspired Artificial Intelligence, Neuron 95:245–258, https://doi.org/10.1016/j.neuron.2017.06.011 . Brain image: NIAID Visual & Medical Arts, Ryan Kissinger (2024), Brain Lateral, NIH BioArt BIOART-000060, public domain, courtesy of NIAID; https://bioart.niaid.nih.gov/bioart/60.

## 52. References

Optional reference · optional

Full primary-source links and figure provenance are provided here and in the instructor notes. The author manuscript for Semantic Cognition predates the2004 book edition. Architecture dates identify seminal sources rather than all modern implementations.

## Full references

- [Christopher Olah (2015) · Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Rogers & McClelland (2004) · Semantic Cognition](https://web.stanford.edu/~jlmcc/papers/RogersMcCBook_7_03.pdf)
- [Saffran, Aslin & Newport (1996) · Statistical learning in infants](https://pubmed.ncbi.nlm.nih.gov/8943209/)
- [Elman (1990) · Finding Structure in Time](https://jeffelman.ucsd.edu/research/publications/)
- [Howard & Kahana (2002) · Temporal Context Model](https://memory.psych.upenn.edu/files/pubs/HowaKaha02.pdf)
- [Hochreiter & Schmidhuber (1997) · LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Gers, Schmidhuber & Cummins (2000) · Learning to Forget](https://doi.org/10.1162/089976600300015015)
- [Cho et al. (2014) · RNN Encoder–Decoder / GRU](https://aclanthology.org/D14-1179/)
- [Yamins et al. (2014) · Visual cortex](https://doi.org/10.1073/pnas.1403112111)
- [Vaswani et al. (2017) · Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al. (2019) · GPT-2](https://cdn.openai.com/better-language-models/language-models.pdf)
- [Schrimpf et al. (2021) · Language and brain](https://doi.org/10.1073/pnas.2105646118)
- [Devlin et al. (2019) · BERT](https://aclanthology.org/N19-1423/)
- [Tulving (1972/1987) · Episodic and semantic memory](https://garfield.library.upenn.edu/classics1987/A1987K743800001.pdf)
- [Mandler & McDonough (1993) · Concept Formation in Infancy](https://doi.org/10.1016/S0885-2014(93)80003-C)
- [Rogers et al. (2004) · Semantic dementia](https://web.stanford.edu/~jlmcc/papers/RogersETAL04.pdf)
- [Saxe, McClelland & Ganguli (2019) · Semantic development](https://doi.org/10.1073/pnas.1820226116)
- [Anderson (2026) · Beni greenlet news report](https://www.sci.news/biology/beni-greenlet-hylophilus-moxensis-14464.html)
- [Krizhevsky, Sutskever & Hinton (2012) · AlexNet](https://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)
- [Richards et al. (2019) · Deep learning for neuroscience](https://doi.org/10.1038/s41593-019-0520-2)
- [Hassabis et al. (2017) · Neuroscience-inspired AI](https://doi.org/10.1016/j.neuron.2017.06.011)
- [NIAID Visual & Medical Arts (2024) · Human brain illustration · Courtesy of NIAID](https://bioart.niaid.nih.gov/bioart/60)