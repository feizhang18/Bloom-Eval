# Metrics

Bloom-Eval defines `16` metrics across `6` cognitive levels.

## Level 1: Memory

- `EFid` (Entity Fidelity): entity overlap and distribution alignment
- `HIRC` (High-Impact Reference Coverage): recall of seminal references
- `FCons` (Factual Consistency): entailment-based factual support

## Level 2: Comprehension

- `OTC` (Outline Topic Coverage): heading/topic coverage against expert outline
- `CF` (Citation Faithfulness): whether claims are supported by cited papers
- `TFSim` (Thematic Focus Similarity): theme overlap and distribution alignment
- `TBal` (Thematic Balance): topic balance within a survey

## Level 3: Application

- `FMI` (Formatting Integrity): citation marker and bibliography consistency
- `DSI` (Document Structure Integrity): required section completeness
- `FAP` (Framework Application): use of a coherent survey framework via GRADE

## Level 4: Analysis

- `STS` (Semantic Tree Similarity): outline tree similarity
- `SCons` (Shape Consistency): depth/breadth consistency
- `SCS` (Structural Clarity Score): structural redundancy penalty

## Level 5: Evaluation

- `CAA` (Critical Analysis Alignment): overlap of critical judgments

## Level 6: Creation

- `FNov` (Framework Novelty): originality of the organizing framework via GRADE
- `ROQ` (Research Outlook Quality): quality of future directions via GRADE

## Release Expectation

Each metric should eventually have:

- an executable implementation
- an input schema
- an output schema
- a prompt file if LLM-assisted
- a reproducibility note

