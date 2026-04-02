# The Tensorial Kernel — Research Document

> **Status:** Conceptual architecture / position paper candidate  
> **Date:** March 2026  
> **Classification:** Novel synthesis in an active research field

---

## Table of Contents

1. [The Problem Being Solved](#1-the-problem-being-solved)
2. [The Architecture](#2-the-architecture)
3. [Core Mechanisms](#3-core-mechanisms)
4. [Contract Language — Formal Stress Test](#4-contract-language--formal-stress-test)
5. [Adversarial Analysis — The Six Redlines](#5-adversarial-analysis--the-six-redlines)
6. [Current State of Research](#6-current-state-of-research)
7. [Novelty Assessment](#7-novelty-assessment)
8. [Practical Applications](#8-practical-applications)
9. [Honest Evaluation — Pros and Cons](#9-honest-evaluation--pros-and-cons)
10. [Hardware Prerequisites and Timeline](#10-hardware-prerequisites-and-timeline)
11. [Recommended Next Steps](#11-recommended-next-steps)
12. [References and Related Work](#12-references-and-related-work)

---

## 1. The Problem Being Solved

### The Foundational Assumption That Is Now Wrong

Every kernel architecture in existence — monolithic, microkernel, exokernel,
unikernel — was designed when the CPU was the fastest and most important
component in the system. The entire kernel model follows from this: the kernel
manages resources on behalf of the CPU, schedules CPU time, and mediates all
data movement through CPU-accessible memory.

That assumption is now false.

| Resource | 2000 | 2026 |
|---|---|---|
| CPU compute | Fastest thing in system | Often the bottleneck *waiting* for data |
| GPU | Not in servers | More FLOPS than CPU by 10–100× |
| NVMe storage | Didn't exist | Faster than 2010 DRAM |
| Network | 1 GbE | 400 GbE SmartNICs with onboard compute |
| AI accelerators | Research curiosity | Dominant compute for most new workloads |
| Interconnect | PCIe 1.0 | CXL 3.1 cache-coherent fabric, 128 GB/s |

The consequence: 70% of AI training time is spent on I/O rather than
computation. Up to 40% of GPU time is wasted waiting for data to arrive from
storage or across the network — data that never needed to touch the CPU or
system RAM at all.

Every existing kernel architecture routes this data through CPU-accessible
memory because that is what kernels know how to manage. The tensorial kernel
asks: what if the kernel's job was not to manage resources at all, but to
maintain *invariants across* resources, letting the hardware manage itself?

---

## 2. The Architecture

### Core Concept

The Tensorial Kernel is a **consistency layer**, not a resource manager. It
does not own resources, schedule time on them, or mediate access to them in
the traditional sense. Instead it:

1. Maintains a verified model of the hardware topology
2. Accepts consistency contracts from applications
3. Configures hardware to satisfy those contracts
4. Steps aside and lets hardware execute directly

The CPU is not privileged after bootstrap. It is one participant among many —
equal in the kernel's eyes to the GPU, NPU, or SmartNIC.

### The Architecture in One Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Applications                                   │
│  declare consistency contracts — what invariants they need       │
└─────────────────────────┬────────────────────────────────────────┘
                          │ contracts (declarative, typed)
┌─────────────────────────▼────────────────────────────────────────┐
│                    Tensorial Kernel                               │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Contract Verifier│  │Topological Mapper│  │ Epoch Manager │  │
│  │ (load-time check)│  │(hardware routing)│  │(teardown/scrub│  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Consistency Fabric (witness chains)           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────┬──────────────────┬──────────────────┬────────────────┘
           │                  │                  │
    ┌──────▼──┐         ┌─────▼───┐       ┌─────▼───┐
    │   CPU   │         │   GPU   │       │   NPU   │   ...
    │(participant)      │(participant)     │(participant)
    └─────────┘         └─────────┘       └─────────┘
         CXL 3.x cache-coherent fabric connecting all
```

### Comparison with Existing Architectures

| Property | Monolithic | Microkernel | Exokernel | **Tensorial** |
|---|---|---|---|---|
| CPU-centric | Yes | Yes | Yes | **No** |
| Resource model | Managed objects | IPC services | Raw hardware | **Invariant contracts** |
| Abstraction unit | Syscall | Message | Library OS | **Projection** |
| Scheduling unit | Process/thread | Process | Application | **Constraint graph** |
| Kernel's primary job | Manage resources | Mediate IPC | Expose hardware | **Maintain consistency** |
| Failure mode | Silent/panic | IPC error | UB | **Typed contract violation** |

---

## 3. Core Mechanisms

### 3.1 Capability Tensors

A resource is not described as "4GB of RAM at address X." It is described as
a **tensor** — a multi-dimensional structure that presents a different typed
view of the same physical resource depending on the observer.

```rust
// The same physical memory region, seen from three perspectives
let mem = CapabilityTensor::new(phys_region);

let cpu_view = mem.project::<CpuAddressSpace>();
// → Virtual address with cache coherency semantics

let gpu_view = mem.project::<GpuAddressSpace>();
// → CUDA unified memory semantics, accessible to GPU kernels

let dma_view = mem.project::<DmaDescriptor>();
// → Scatter-gather list for NIC DMA, bypasses CPU entirely
```

**Why this is novel:** No existing OS describes resources this way. A resource
in Linux is a file descriptor, a socket, a page mapping — a single view. The
tensor model allows the same physical resource to be legally accessed from
multiple hardware perspectives simultaneously, with the kernel's job being to
keep those views consistent rather than to control who can access what.

**Rust's ownership model does real work here:** A projection is an owned Rust
value. `Send` and `Sync` become coherency markers: a type that is `!Send`
cannot be projected across a compute element boundary without an explicit
coherency contract. This is the strongest genuinely novel contribution of the
contract language — applying Rust's existing ownership system as a *coherency
enforcement mechanism*, not merely a memory safety mechanism. The closest
formal analogue is **fractional permissions** (Boyland 2003): a resource can
be split into read-capable fractions and a single write-capable whole, mapping
directly onto `coherent` (multiple readers with ordering) vs. `exclusive`
(single writer). This connection provides a path to a well-founded formal
semantics.

**Critical caveat — physical aliasing:** The borrow checker enforces that two
Rust objects cannot alias. But two independently constructed `CapabilityTensor`
objects over *overlapping physical address ranges* are different Rust objects —
the compiler cannot detect the physical overlap. If two components independently
construct tensors over the same MMIO region and both declare `exclusive`
projections, both contracts are accepted but neither is actually exclusive. The
kernel must perform physical address overlap detection at contract submission
time, which is not a Rust-level guarantee. This is an open implementation
requirement, not a solved problem.

### 3.2 Consistency Contracts

Instead of system calls, applications declare **consistency contracts** —
statements about what invariants must hold across compute elements:

```rust
contract! {
    // These three views must see each other's writes within 100ns
    coherent(cpu_view, gpu_view, nic_view) within 100ns;

    // This projection is exclusive — no concurrent writers
    exclusive(ssd_view);

    // These two views may diverge — application reconciles manually
    relaxed(replica_a, replica_b);
}
```

The kernel verifies at load time that the hardware topology can satisfy the
contract, configures the hardware (IOMMU, CXL fabric, cache coherency
controllers), and steps aside. The contract is the kernel interface — there
are no system calls in the traditional sense.

**Formal semantics (draft — requires rigorous treatment):**

For `coherent(A, B) within T`: for any write `w` to the physical resource via
projection `A` at logical time `t`, the written value must be observable via
projection `B` by time `t + T`. In distributed systems terms this is closest
to **δ-consistency** (Yu & Vahdat 2000) — bounded-time visibility — which is
strictly between linearizability and eventual consistency.

For `exclusive(X)`: at most one write-capable projection of the physical region
exists at any time. Closest formal analogue: the write-capable whole in
fractional permissions (Boyland 2003).

For `relaxed(A, B)`: no ordering or visibility guarantee. Semantically
equivalent to eventual consistency with unbounded convergence time. The
reconciliation mechanism must be defined by the application — the contract
language provides no reconciliation primitives.

**Known semantic gaps (open problems):**

- **`within T` presupposes a global clock**, which does not exist in a
  distributed hardware system. CXL defines ordering but not a shared time
  reference. Clock skew between devices is on the order of 1–10ns and varies
  with thermal state. For tight bounds (`within 10ns`), the guarantee is
  physically ambiguous. The contract language must be grounded in
  hardware-observable happens-before ordering events, not wall-clock time.
- **Coherency transitivity is undefined.** If `coherent(A, B) within 50ns`
  and `coherent(B, C) within 50ns`, is `coherent(A, C) within 100ns` implied?
  If transitive, pairwise constraints grow O(n²) for n projections.
  If not transitive, applications will silently assume transitivity and have
  correctness bugs.
- **`exclusive` granularity is unspecified.** Byte, cache line, page, or
  entire region? Exclusive at one granularity does not imply exclusive at
  another.
- **`relaxed` convergence is unspecified.** "No guarantee" could mean writes
  never propagate (useless) or propagate eventually (eventual consistency).
  These are not equivalent.

**Contract classes (for tractability):**

| Class | Topology | Solve complexity | Who can use |
|---|---|---|---|
| A | Single resource, one user | O(1) | All |
| B | Linear chain | O(n) | All |
| C | Acyclic dependency graph (DAG) | O(n²) claimed* | All |
| D | Arbitrary graph | NP-complete | Privileged only |

*The O(n²) claim for Class C requires proof for the hardware mapping problem
specifically. The abstract constraint graph is a DAG, but mapping it to a
physical hardware topology is a subgraph isomorphism problem, which is
NP-complete in general. The polynomial claim holds only if the hardware topology
has special structure (e.g., tree topology) — which must be proven, not assumed.

**Critical: composition breaks the class hierarchy.** Two independent Class C
contracts (individually acyclic) can compose into a Class D problem if they
share a projection vertex. An adversary can submit many small Class C contracts,
each individually valid, whose composition creates an NP-complete constraint
system. The per-contract class restriction does not bound the complexity of the
*global* constraint system. This is the most serious structural weakness in
the current design and requires a compositional type theory or a global
composition checker with bounded complexity.

### 3.3 Topological Mapper

Replaces the traditional process scheduler. Takes a workload graph and finds
the optimal embedding in the hardware topology that satisfies all consistency
contracts while minimising energy and maximising throughput.

```
Workload graph:         Hardware graph:
A → B → C              CPU0 ─── L3 ─── GPU0
|       |               |               |
D → E → F             DRAM ── CXL ─── NPU
                         |
                        NVMe
```

The mapper finds which hardware element should execute each workload node, and
which physical paths should carry data between them. The CPU may or may not
appear in the critical path — it depends on the workload, not on the
architecture's assumptions.

### 3.4 Witness Chains (Byzantine Fault Tolerance)

A device that claims a write has reached global visibility must be
independently confirmed by **k topologically independent witnesses**:

```rust
struct WitnessChain {
    threshold: usize,           // k — configurable per contract
    witnesses: Vec<WitnessId>,  // must be on disjoint fabric paths
    attestation: AttestType,
}

enum AttestType {
    ReadVerify { delay_ns: u64 },   // witness reads back and confirms
    PerfCounterEpoch,               // cache line eviction confirmed
    CpuGroundTruth,                 // expensive, used as fallback only
}
```

Witnesses must be topologically independent — they cannot share a CXL switch
or fabric segment with the device under verification, as shared buffering
allows correlated false confirmation.

### 3.5 Epoch-Based Projection Teardown

When a projection is released (device preempted, application exits), the
kernel must "unwire" the physical consistency contract. This is harder than
flushing a TLB — it requires convincing multiple independent hardware agents
to simultaneously agree the contract no longer exists.

Three-phase teardown:

1. **Drain:** Mark epoch as `Draining`. No new operations accepted. In-flight
   operations complete (bounded by `max_flight_time` declared at contract
   creation).
2. **Confirm:** Each participant acknowledges drain. Non-responding devices are
   quarantined.
3. **Scrub:** Physical memory is overwritten before the epoch is released for
   reuse. Prevents cross-projection data remanence.

```rust
enum EpochState {
    Active,
    Draining { initiated_at: Timestamp, max_flight_time: Duration },
    Tombstoned { confirmed_at: Timestamp, scrub_pattern: ScrubPattern },
}
```

### 3.6 Legacy Device Quarantine

Devices that do not support CXL flit-based atomicity or witness capability
are placed in **quarantine domains** — isolated IOMMU regions where they
cannot affect the tensorial fabric directly:

```
[Tensorial fabric] ←→ [Quarantine boundary] ←→ [Legacy device]
   Attested                Firewall                  Opaque
   coherency              (IOMMU)                   MMIO
```

Data transfer between the tensorial fabric and a legacy device requires an
explicit tombstone/re-projection cycle through a staging buffer — expensive
but safe and auditable.

### 3.7 Stratified Bootstrap

Cold boot proceeds in three strata, each abandoned once the next is
established:

- **Stratum 0 — Physical:** Hardware power sequencing determines boot order.
  The CXL host bridge (typically CPU-side) must initialise before any CXL
  device can communicate. This is physical, not a software decision.
- **Stratum 1 — CPU-mediated enumeration:** CPU uses MMIO to enumerate
  topology via PCIe config space and ACPI/SRAT tables. CPU holds a
  `TemporalAuthority` capability — time-limited and self-revoking.
- **Stratum 2 — Distributed ratification:** CPU broadcasts a topology
  proposal. Each device signs the topology hash with its attestation key.
  Quorum (≥ ⅔ agreement) establishes the Root Tensor. CPU's
  `TemporalAuthority` is revoked. CPU is now just another participant.

```rust
// TemporalAuthority is !Send and !Sync — cannot outlive bootstrap phase
struct TemporalAuthority {
    _not_send: PhantomData<*mut ()>,
    valid_until: Instant,
}

impl Drop for TemporalAuthority {
    fn drop(&mut self) {
        // Revoke CPU's bootstrap privileges
        // The CPU becomes a fabric participant — no special authority
        self.iommu.revoke_bootstrap_domain();
        self.fabric.demote_cpu_to_participant();
    }
}
```

This is **not a microkernel** because the CPU's authority is self-terminating.
A microkernel's privileged server is permanent. The bootstrap scaffold is
demolished once the fabric can support itself.

---

## 4. Contract Language — Formal Stress Test

*This section records findings from a rigorous peer-review-style stress test
of the contract language. It is the most important section for anyone
considering publication. Findings are classified as Critical (blocks
publication), Important (significant revision required), or Advisory.*

### 4.1 Pathological Contracts

**Case 1 — The Zero-Latency Paradox**

```rust
contract! { coherent(cpu_view, gpu_view) within 0ns; }
```

Syntactically valid (Class B). Physically impossible: minimum CXL round-trip
latency is ~100–500ns. The contract verifier must have lower-bound latency
data per hardware link to reject this at load time. Without it, the contract
is accepted and the system enters permanent oscillation. This reveals a
fundamental gap: the verifier needs both upper-bound feasibility (can the
hardware achieve `within T`?) and lower-bound feasibility (is `T` physically
achievable at all?). The architecture description only addresses the former.

**Case 2 — The Cross-Application Cycle**

Application A declares `coherent(A_cpu, A_gpu) within 50ns` and
`relaxed(A_gpu, shared_nic)`. Application B independently declares
`coherent(B_gpu, shared_nic) within 30ns` and `coherent(shared_nic, A_cpu)
within 20ns`. The composition creates a coherency cycle with a mixed
coherent/relaxed edge. Each contract is individually valid. The composition
is globally inconsistent — but there is no described mechanism for
cross-application contract composition checking. The kernel silently operates
with an inconsistent global constraint.

**Case 3 — The Timing Interference Chain**

Three applications each submit a valid Class B contract routing through
overlapping CXL fabric links. Under light load all three are satisfied. Under
heavy load, shared link contention causes all three to miss their bounds
simultaneously. The contract language has no vocabulary for expressing or
tracking bandwidth consumption on shared fabric elements — two contracts can
be individually satisfiable but mutually incompatible under contention.

**Case 4 — The Phantom Exclusivity Alias**

```rust
// Application A
let t1 = CapabilityTensor::new(region_0x1000_0000_to_0x1FFF_FFFF);
contract! { exclusive(t1.project::<CpuAddressSpace>()); }

// Application B (different tensor object, overlapping physical backing)
let t2 = CapabilityTensor::new(region_0x1000_to_0x2000_subregion);
contract! { exclusive(t2.project::<GpuAddressSpace>()); }
```

Both contracts are accepted. Both applications believe they have exclusive
access. Neither does. The Rust borrow checker cannot detect physical address
overlap between independently constructed tensor objects. Physical aliasing
detection at the kernel level is a hard requirement, not an optimisation.

**Case 5 — The Transitivity Horizon**

```rust
contract! {
    coherent(A, B) within 50ns;
    coherent(B, C) within 50ns;
    // Is coherent(A, C) within 100ns guaranteed? The language doesn't say.
}
```

If the application assumes transitivity (reasonable) but the kernel doesn't
enforce it (underspecified), the application has a latent correctness bug that
manifests only when `A → C` propagation takes the slow path. This is an
underspecification, not an edge case.

### 4.2 The Composition Problem (Critical)

The class hierarchy does not bound the complexity of composed contracts.
Formally: let `G₁` be the constraint graph of contract `C₁` (Class C, acyclic)
and `G₂` be the constraint graph of `C₂` (Class C, acyclic). If `G₁` and `G₂`
share a vertex (a projection referenced by both), then `G₁ ∪ G₂` may contain
a cycle, making the composed problem Class D (NP-complete).

The DoS prevention mechanism (restricting users to Class C) fails at the
system level. An adversary submitting many individually valid Class C contracts
can compose them into a Class D constraint system, paying only Class C credit
per submission. The economic rate limiting doesn't bound composition cost.

**Required fix (one of):**
1. A global composition checker that verifies the composed constraint graph
   remains in the target class — but this checker may itself be NP-hard
2. A prohibition on shared projections across contracts — eliminates most
   interesting multi-application scenarios
3. A compositional type theory where `C₁ ⊕ C₂` is a defined operation with
   a provable complexity bound

### 4.3 The Global Clock Problem (Critical)

`within 100ns` is not physically meaningful as a hard guarantee in a
distributed hardware system. There is no global clock. Each compute element
has its own local clock with drift that varies with temperature. CXL provides
ordering (A's write is ordered before B's read) but not timing. Clock skew
between devices is on the order of 1–10ns.

**The minimal counterexample:**

```rust
contract! { coherent(cpu_view, gpu_view) within 1ns; }
```

This is a valid Class B contract syntactically. It is physically impossible:
propagation delay alone across a 30cm PCIe/CXL trace is ~1ns; store buffer
drain, cache writeback, CXL request, and GPU cache fill each take additional
nanoseconds. The contract verifier, without lower-bound latency data, accepts
it. The hardware never satisfies it. The kernel oscillates.

**Required fix:** Ground `within T` in hardware-observable happens-before
ordering events, with `T` treated as an engineering target verified against
observed hardware latency distributions, not as a hard guarantee. This weakens
the guarantee claim but makes it honest.

### 4.4 Failure Attribution (Important)

In a multi-party coherency contract, violations are non-attributable by design.
Consider a three-device coherency ring where `cpu → nic` visibility takes 120ns
against a 50ns contract. The violation could be caused by GPU forwarding delay,
NIC local buffering, CXL fabric retry, or CPU store buffer drain. The witness
chain confirms that the write eventually arrived — it does not identify which
component caused the delay.

The re-mapping response is therefore blind: it tries a different topological
mapping without knowing which component to route around. This is a fundamental
limitation, not an implementation gap. Attributable violations require a
component with a global view — which the tensorial model eliminated.

**Implication for recovery:** The failure degradation mechanism (re-mapping,
contract relaxation) is heuristic, not targeted. The paper must acknowledge
this explicitly rather than implying that re-mapping resolves the underlying
fault.

### 4.5 Load-Time vs. Runtime Verification Gap (Critical)

The architecture claims contracts are verified at load time against the hardware
topology. But:

- Lower-bound latency feasibility (`within 1ns` is impossible) requires
  knowing achievable hardware latency — which is runtime data
- Composition conflicts with other active contracts require knowledge of all
  currently active contracts — which changes at runtime
- Physical aliasing detection requires knowing all currently allocated tensor
  physical ranges — which changes at runtime

Load-time verification can check structural feasibility (is this contract
syntactically well-formed and graph-class-compatible?). It cannot check
physical feasibility or composition safety without runtime information. The
paper must clearly separate what is verified at load time from what is
monitored at runtime, and what guarantees hold in each case.

### 4.6 Comparison to Existing Formal Models

| Formalism | What it captures | Gap vs. tensorial contracts |
|---|---|---|
| Memory consistency models (TSO, ARM) | Ordering and visibility | No timing bounds; assume shared address space |
| δ-consistency (Yu & Vahdat 2000) | Bounded-time visibility | Assumes global clock; no type-level ownership |
| Session types | Typed communication protocols | Handles protocols, not shared memory consistency |
| Linear logic / linear types | Resource ownership, single use | Atemporal; no visibility or ordering semantics |
| Fractional permissions (Boyland 2003) | Read/write capability splitting | No timing; no cross-device semantics |

**The genuine novelty:** The tensorial contract language is the first proposal
to combine fractional-permissions-style ownership (for projections) with
bounded-visibility consistency guarantees (for coherency) expressed as a
kernel-facing API grounded in the host language's type system. Each piece
exists separately. The combination does not.

**The recommended formalisation path:** Ground the contract language in
fractional permissions (Boyland 2003) for the ownership semantics, and
happens-before ordering from the CXL formal memory model for the consistency
semantics. Treat timing bounds as advisory engineering targets (for
optimisation and degradation decisions) rather than hard correctness
guarantees. This is a meaningful weakening of the original claim but produces
a model that is honest and potentially verifiable.

### 4.7 Publication Readiness Assessment

| Finding | Severity | Status |
|---|---|---|
| `within T` undefined without global clock | **Critical** | Open |
| Composition breaks class hierarchy | **Critical** | Open |
| Physical aliasing not detected by type system | **Critical** | Open |
| Load-time vs. runtime verification conflated | **Critical** | Open |
| Class C polynomial claim unproven for hardware mapping | **Important** | Open |
| Failure attribution is non-deterministic | **Important** | Open |
| Coherency transitivity undefined | **Important** | Open |
| `relaxed` convergence undefined | **Important** | Open |
| `exclusive` granularity unspecified | Advisory | Open |
| Fractional permissions connection unmade | Advisory | Open |

**Verdict:** The contract language is promising but seriously underspecified
for publication at SOSP or EuroSys. It is suitable for a HotOS position paper
if the four Critical items are addressed and the remaining items are explicitly
acknowledged as open problems. The closest path to publication is grounding the
formal semantics in fractional permissions + CXL happens-before, restricting
timing bounds to advisory status, and adding a global composition checker (or
proving composition is restricted).

---

## 5. Adversarial Analysis — The Six Redlines

### Redline 1: The Byzantine Interconnect

**Attack:** A CXL device lies about write visibility, ACKing a coherency
message before data has actually propagated. The consistency fabric is built
on a false foundation.

**Response:** Probabilistic witness chains with topological independence
requirements. Hardware attestation roots (CXL 3.0 integrity features) for
high-stakes contracts. CPU ground-truth verification as expensive fallback.

**Residual risk:** Colluding witness networks. Unmitigable without physical
trust perimeter.

### Redline 2: The Shadow-State Paradox

**Attack:** Two valid contracts are individually satisfiable but mutually
destructive at the physical layer (Rowhammer-style interference, PCIe power
delivery). Adversary submits graphs designed to force exponential solver time.

**Response:** Three layers:
1. Contract language restricted to tractable Class A–C for unprivileged users
2. Topological credits — economic rate limiting, non-refundable during solving
3. Physical interference monitoring via hardware performance counters, with
   forced workload separation when detected

**Residual risk:** Class D contracts (most interesting cases) require offline
verification or privileged access. Tractability vs. expressiveness tradeoff
cannot be fully resolved.

### Redline 3: The Leaky Projection

**Attack:** GPU preempted while holding a projection. Fabric teardown is
non-atomic across distributed hardware agents. Data from previous tenant
accessible during drain window.

**Response:** Epoch-based tombstoning with mandatory physical scrub before
reuse. Drain window is a first-class contract parameter declared at creation
time. Timeout triggers hardware reset of non-responding participants.

**Residual risk:** Drain window exists between `Draining` and `Tombstoned`.
Duration is bounded and contractually declared, but not zero.

### Redline 4: The Analog Jitter (Non-Deterministic Fabric)

**Attack:** Thermal throttling and link-layer retries push actual latency above
contract bounds. Topological mapper enters oscillation, continuously re-mapping
without resolving the underlying physical constraint violation.

**Response:** Bayesian contract relaxation — distributional contracts instead
of point guarantees. Live fabric model updated from hardware performance
counters. Failure classification by autocorrelation (persistent failure vs.
transient jitter). Re-mapping hysteresis with exponential backoff to prevent
oscillation.

**Residual risk:** Fabric model requires calibration. Heuristic classification
thresholds are empirically tuned, not mathematically derived.

### Redline 5: The Dark Silicon Orphan (Legacy Passthrough)

**Attack:** Most 2026 hardware doesn't support CXL atomicity or witness
capability. Without a solution, the architecture collapses to a monolithic shim
for all legacy device interactions.

**Response:** Quarantine domains — IOMMU-enforced physical isolation. Opaque
tensors for legacy devices. Explicit tombstone/re-projection for all data
transfers across the quarantine boundary. Contamination firewall: no tensorial
contract can span a quarantine boundary without automatic downgrade to
`BestEffort`.

**Residual risk:** Every legacy device interaction pays tombstone/re-projection
overhead. For systems where legacy devices are in the critical path, this makes
the tensorial kernel slower than Linux.

### Redline 6: The Meta-Consistency Race (Bootstrap)

**Attack:** At cold boot, nothing is initialised. Multiple compute elements
race to define the Root Tensor. Using a service processor to arbitrate
reinvents the microkernel. Without arbitration, there is no convergence.

**Response:** Stratified bootstrap exploits physical power sequencing as
Stratum 0 arbiter (not software). CPU holds `TemporalAuthority` during Stratum
1 enumeration, then Stratum 2 distributed ratification via hardware attestation
keys establishes the Root Tensor. CPU authority is self-revoking.

**Residual risk:** Requires hardware attestation keys on every device.
Attestation PKI is a trust anchor that the kernel architecture cannot protect.
Bootstrap has a hard timeout with fallback to conventional kernel mode —
requiring maintenance of two kernel paths.

---

## 6. Current State of Research

### The Research Landscape in 2026

The tensorial kernel concept arrived at exactly the right moment. CXL research
has exploded: only 10 papers appeared between 2019–2022, 40 in 2023, and 51 in
2024. The hardware prerequisites are arriving now.

### Directly Related Prior Work

**M3 — TU Dresden (ASPLOS 2016)**
The closest prior work on architecture philosophy. M3 argues for removing the
CPU from the critical path by placing a dedicated Data Transfer Unit (DTU/TCU)
next to every compute element, including accelerators, making them all
first-class OS citizens. M3 achieves "no CPU in critical path" via dedicated
hardware per compute unit. The tensorial kernel achieves the same goal via
consistency contracts over a CXL fabric — different mechanism, same principle.

**Stramash — ASPLOS 2025**
Published the same month as our initial discussion. Introduces the "fused
kernel" design that exploits cache-coherent shared memory among
heterogeneous-ISA CPUs as a first principle, delivering up to 2.1× speedup on
NPB benchmarks. This directly validates the CXL-native assumption that the
tensorial kernel is built on.

**Arrakis — OSDI 2014**
Established the control plane / data plane split. Applications have direct
access to virtualised I/O devices; the kernel only manages protection, not
data movement. The tensorial kernel generalises this beyond networking and
storage to all compute elements.

**Barrelfish / Multikernel — SOSP 2009**
Structures the OS as a distributed system of cores sharing no memory, with
consistency maintained via agreement protocols. The tensorial kernel's
distributed consistency fabric is a direct descendant of this insight, extended
to heterogeneous hardware.

**XSched — OSDI 2025**
Proposes a unified abstraction for preemptive scheduling across heterogeneous
XPUs (GPUs, NPUs, etc.) with hardware-agnostic policies. Addresses the
same heterogeneous scheduling problem from a different angle — scheduling
policies rather than consistency contracts.

**RaBAB-NeuSym Kernel — 2025**
Independently converged on linear types as a kernel resource primitive,
allocating compute and memory as `LinearResource` tokens with single-use
semantics enforced through Linear Logic. Direct parallel to the tensorial
kernel's Rust ownership approach.

**CapsLock — CCS 2025**
Introduces revoke-on-use capability semantics at the hardware level: accessing
a memory object via a capability implicitly invalidates conflicting capabilities.
This is essentially the `CapabilityTensor` projection model, independently
derived and implemented in hardware.

**seL4 — SOSP 2009 / ACM TOCS 2014**
The gold standard for kernel formal verification. Machine-checked proof of
functional correctness (8,700 lines of C), information-flow security, and
binary correctness on multiple architectures. The tensorial kernel has no
comparable formal foundation — seL4 represents the bar that must eventually
be reached.

### Open Problems the Literature Has Identified

The research literature explicitly identifies these as unsolved:

1. **Portable flush/ordering primitives** for CXL-enabled shared memory
   systems — the consistency contract language is a direct proposed answer.
2. **OS/hypervisor integration** for persistent and crash-tolerant
   applications on CXL fabrics — no solution exists.
3. **Schedulability analysis** for heterogeneous accelerator systems in
   real-time contexts — partially addressed by the topological mapper's
   objective function.
4. **Hardware-native multi-tenancy isolation** without hypervisor overhead —
   the quarantine domain mechanism addresses this.

---

## 7. Novelty Assessment

### What Exists (Prior Art)

| Component | Closest prior work | How close |
|---|---|---|
| CPU removed from data path | Arrakis (2014), M3 (2016) | Strong precedent |
| Heterogeneous compute as first-class | M3, Stramash (2025) | Strong precedent |
| CXL as coherency fabric | Active 2024–25 research | Hardware just arriving |
| Linear types as resource primitive | RaBAB-NeuSym, CapsLock (2025) | Independent convergence |
| Distributed kernel consistency | Barrelfish (2009) | Conceptual precedent |
| Read/write capability splitting | Fractional permissions, Boyland (2003) | Direct formal precedent for projection model |
| Bounded-time visibility (δ-consistency) | Yu & Vahdat (2000) | Direct semantic precedent for `coherent within T` |

### What Appears Novel

| Component | Status |
|---|---|
| **Consistency contracts as primary kernel API** | Not found in literature |
| **CapabilityTensor multi-perspective projection** | Not found in literature |
| **Topological mapping replacing scheduling** | Not found in literature |
| **Bayesian contract relaxation for jitter** | Not found in literature |
| **Epoch tombstoning for fabric teardown** | Not found in literature |
| **Stratified bootstrap with self-revoking authority** | Not found in literature |
| **Contract class hierarchy for DoS prevention** | Not found in literature |
| **Rust ownership as cross-device coherency enforcement mechanism** | Not found in literature — strongest novel claim |

### The Honest Novelty Claim

The tensorial kernel is **a novel synthesis in an active field**, not a novel
invention in isolation. The ownership semantics have a formal predecessor in
fractional permissions (Boyland 2003). The timing semantics have a predecessor
in δ-consistency (Yu & Vahdat 2000). What hasn't been proposed is the unifying
abstraction: applying these two formalisms together, grounded in Rust's type
system, as a kernel-facing API over heterogeneous CXL-connected hardware.

Critically, applying Rust's `Send`/`Sync` traits as **coherency markers** —
where `!Send` means "cannot cross a coherency domain boundary without a
contract" — is specific, implementable, and not previously proposed. This is
the strongest individual novel claim in the architecture.

The field has independently validated the core premises across 2024–25 papers,
which means the research community is converging on the problem space without
having proposed this specific solution.

---

## 8. Practical Applications

### 7.1 AI/ML Training Infrastructure (Strongest Case)

**The problem:** 70% of AI training time is spent on I/O rather than
computation. Up to 40% of GPU time is wasted waiting for data. Memory bandwidth
has scaled at 1.6× per two years while compute has scaled at 3× — the gap
grows every generation.

**Why tensorial helps:** Consistency contracts between NVMe, GPU HBM, and NPU
would eliminate CPU-mediated data staging. The topological mapper would
automatically route data through GPUDirect-style paths without application
engineers manually configuring RDMA, DMA descriptors, and memory pinning.

**Concrete opportunity:** Recovering the 40% GPU idle time is worth hundreds of
millions of dollars annually at hyperscaler scale. CXL KV cache offloading
alone has demonstrated up to 21.9× throughput improvement in early research.

**Timeline:** CXL 3.1 is deploying now. NVIDIA Blackwell supports CXL. AMD
MI300X supports CXL. The hardware is arriving in 2026–2027.

### 7.2 High-Frequency Trading (Near-Term Best Fit)

**The problem:** Traditional kernels add 20–50 microseconds of latency for
network packet processing. HFT targets sub-10-microsecond round trips. The
industry has responded by bypassing the kernel entirely (DPDK, RDMA, custom
NIC firmware).

**Why tensorial helps:** The industry is already manually implementing the
tensorial model — direct hardware access with no kernel mediation. The
tensorial kernel would formalise and make this safe: consistency contracts
instead of hand-tuned bypass code, typed projections instead of raw MMIO,
witness chains instead of "we tested it and it seems to work."

**Why this is the near-term opportunity:** Hardware requirements are modest
compared to full CXL 3.0 deployment. The economic incentive is enormous. A
tensorial kernel subsystem for network-to-compute data paths could be
productised on existing RDMA-capable hardware within 3–5 years.

### 7.3 Genomics and Bioinformatics Pipelines

**The problem:** A genomics pipeline is a DAG of compute stages (sequencing →
alignment → variant calling → annotation) where each stage uses different
specialised hardware. Data movement between stages — through CPU and system RAM
— is the bottleneck. Short read alignment is described in the literature as
"ubiquitous and the principal computational bottleneck."

**Why tensorial helps:** The topological mapper treats a genomics pipeline as
what it architecturally is — a data flow graph over heterogeneous hardware.
Consistency contracts enforce ordering between pipeline stages. The projection
model allows the same genomic data to be simultaneously accessible to a GPU
aligner and a CPU variant caller with defined coherency guarantees.

**Reference point:** GPU-accelerated pipelines already deliver up to 60×
speedup vs CPU-only for variant calling (NVIDIA Clara Parabricks on A100).
Eliminating inter-stage data movement overhead could add a further significant
multiplier.

### 7.4 Autonomous Systems — Partial Fit

**Where it fits:** Sensor fusion, AI inference for perception, planning.
These are soft real-time workloads where the tensorial kernel's distributional
contract guarantees (`Contract<Soft> within 10ms at p99`) are appropriate.

**Where it does not fit:** Safety-critical actuator control (brakes, steering,
flight control surfaces). These require hard real-time guarantees with zero
tolerance for violation. The tensorial kernel's Bayesian contract relaxation
is philosophically incompatible with hard real-time requirements.

### 7.5 Cloud Multi-Tenant Compute (Long-Term, Largest Market)

**The problem:** Multi-tenancy is currently achieved through software
virtualisation (hypervisors, containers) that adds overhead and creates
security boundaries in software rather than hardware.

**Why tensorial helps:** Quarantine domains and capability tensor projections
are a hardware-native multi-tenancy model — isolation enforced by IOMMU and
CXL fabric rather than hypervisor software. Consistency contracts become the
mechanism for performance SLA guarantees rather than best-effort. Research
suggests a 20–25% TCO reduction is achievable from improved memory and data
movement efficiency — worth billions annually at hyperscaler scale.

**Timeline:** This requires broad CXL 3.x deployment. ABI Research projects
commercial availability of sufficient software support by 2027. Production
deployment at scale: 2028–2030.

### Application Priority Matrix

| Domain | Fit | Timeline | Market |
|---|---|---|---|
| AI/ML training | Excellent | 2028–2030 | Enormous |
| High-frequency trading | Excellent | 2027–2029 | Large, concentrated |
| Genomics pipelines | Very good | 2028–2031 | Medium, fast-growing |
| Autonomous (soft RT) | Good | 2030+ | Large |
| Autonomous (hard RT) | **Poor — do not pursue** | Never | — |
| Cloud multi-tenancy | Good | 2030–2035 | Massive |

---

## 9. Honest Evaluation — Pros and Cons

### Genuine Advantages

**Failure modes are first-class.**
In a monolithic kernel, a Byzantine NIC or misconfigured IOMMU produces a
panic or silent data corruption. In the tensorial model, these are typed
contract violations with defined degradation paths. The system knows it is
degraded and can express that. Most kernel security vulnerabilities are
undocumented assumptions that turned out to be false — the contract language
forces those assumptions to be stated explicitly and violated loudly.

**Data movement cost is structurally minimised.**
Every other architecture treats data movement as an application concern. The
tensorial model makes it a kernel-level invariant. A projection that routes
data through an unnecessary memory copy violates the topological mapping
objective function and is automatically eliminated.

**Heterogeneous compute is designed-in, not retrofitted.**
CUDA required a separate driver model. ROCm has different semantics. eBPF
offload is architecturally bolted on. The tensorial kernel treats all compute
elements as equivalent fabric participants from the start.

**Rust ownership maps cleanly onto the core primitive.**
This is not a superficial claim. The projection is a type-level operation. The
borrow checker prevents conflicting projections at compile time. `Send`/`Sync`
become coherency markers. Lifetimes enforce contract scopes. The type system
does real semantic work.

**Testability of invariants.**
Because contracts are explicit and declarative, a test framework can inject
synthetic violations and verify degradation paths. This is nearly impossible
with current kernels whose invariants are implicit.

### Real Limitations

**The contract language has critical open problems.**
As detailed in Section 4, the contract language as currently described has
four critical underspecifications that block publication: the global clock
problem (`within T` is not well-defined in distributed hardware), the
composition problem (Class C contracts compose into Class D), physical aliasing
(the type system cannot detect overlapping tensor physical addresses), and the
load-time vs. runtime verification conflation. These are not minor gaps —
they affect the correctness of the core guarantee claim. The architecture's
value proposition ("contracts are verified at load time, rejecting infeasible
contracts") is currently overstated.

**The contract language is the entire bet.**
The history of formal specification languages suggests that languages
expressive enough to capture real-world requirements become too complex to use
correctly. The tractability/expressiveness tradeoff has no known good solution
for general-purpose OS workloads.

**The brownfield cost is enormous.**
In 2026, the majority of devices are non-tensorial. Every legacy device
interaction pays the tombstone/re-projection overhead. For a general-purpose
workstation, the tensorial kernel would likely be slower than Linux for most
workloads.

**Debugging is pathologically hard.**
When the abstraction is "invariants maintained across distributed hardware," a
violation is extremely hard to localise. The Bayesian fabric model means
system behaviour depends on thermal history, link error history, and witness
confidence values accumulated over time. Reproducing a bug requires reproducing
all of that context.

**No formal foundations.**
seL4 has machine-checked proofs of functional correctness, information-flow
security, and binary correctness. The tensorial model's correctness properties
have not been formalised. Without this, the claimed security properties are
engineering intuitions, not guarantees.

**No ecosystem, no path to one in the near term.**
Linux has 30 years of driver code, tool support, language runtimes, and
operational knowledge. The tensorial kernel has none. The development timeline
to a point where it could outperform Linux on real workloads is measured in
decades.

### The Balanced Verdict

| | Monolithic | Microkernel | Tensorial |
|---|---|---|---|
| **Performance ceiling** | High (today) | Medium | Very high (future hardware) |
| **Performance floor** | High | Lower (IPC cost) | Low (brownfield overhead) |
| **Failure explicitness** | Poor | Medium | Excellent |
| **Debuggability** | Good | Medium | Poor |
| **Hardware requirements** | Minimal | Minimal | Demanding |
| **Formal foundations** | Weak | Strong (seL4) | None |
| **Ecosystem** | Vast | Small | None |
| **Right for 2026** | Yes | Niche | No |
| **Right for 2032+** | Maybe | Maybe | **Maybe** |

---

## 10. Hardware Prerequisites and Timeline

### What the Architecture Requires

| Requirement | Status in 2026 |
|---|---|
| CXL 3.x coherent fabric | Deploying now — 90%+ of new servers CXL-capable |
| CXL fabric switches (multi-device) | First silicon sampling (Panmnesia CXL 3.2) |
| Multi-rack memory pooling | Late 2026–2027 production target |
| Hardware attestation per device | Partial — requires CXL security extensions |
| GPU with direct CXL attachment | NVIDIA Blackwell supports CXL on Grace Hopper |
| NPU with CXL participation | Early stage — research prototypes |

### The CXL Timeline

- **2025:** CXL 2.0 memory expanders in production; CXL 3.x switches sampling
- **2026:** CXL 3.1 mainstream deployment; multi-rack pooling begins;
  CXL 4.0 specification released (128 GT/s, PCIe 7.0)
- **2027:** ABI Research inflection point — CXL 3.0/3.1 with sufficient
  software support for broad commercial adoption
- **2028–2030:** Full CXL-native data centre infrastructure realistic

### The Software Gap

The hardware is arriving faster than the software. As of 2026:
- CXL memory appears as NUMA nodes to Linux — the OS has no native concept
  of consistency contracts
- Applications must be CXL-aware or NUMA-aware to benefit — no transparent
  abstraction exists
- The "software stack for memory tiering" is described as an active gap by
  industry analysts

**This gap is exactly the problem the tensorial kernel is designed to solve.**

---

## 11. Recommended Next Steps

### Most Valuable Near-Term Contribution

A full production tensorial kernel is not the right first step. The most
valuable near-term contribution, in order of priority:

**1. Resolve the four critical contract language gaps (prerequisite for everything else)**
- Ground `within T` in hardware-observable happens-before ordering from the
  CXL formal memory model, not wall-clock time. Treat timing as advisory.
- Define `C₁ ⊕ C₂` composition explicitly and prove the class of the result,
  or restrict composition with a justified argument.
- Implement physical aliasing detection at the kernel level — two tensors over
  overlapping physical ranges must be detected at contract submission time.
- Clearly separate load-time structural verification from runtime physical
  feasibility monitoring, with defined guarantees for each.

**2. Ground the formal semantics in existing theory**
Ground ownership semantics in fractional permissions (Boyland 2003) and
visibility semantics in δ-consistency (Yu & Vahdat 2000) combined with CXL
happens-before ordering. This provides an existing formal foundation rather
than requiring a new one from scratch.

**3. Write a position paper**
Target HotOS (short, position-paper bar) rather than SOSP/EuroSys until the
critical items are resolved. Frame relative to Stramash and M3: "what should
the OS interface look like once the hardware these systems assume becomes
widespread?" The novelty claim is the Rust-ownership-as-coherency-enforcement
mapping and the consistency contract API — be precise about what is novel and
what is synthesis.

**4. Implement a tensorial subsystem within Linux**
Rather than a new kernel, implement the topological mapper and contract
verifier as a Linux kernel module for CXL-aware NUMA memory. This gives a
real implementation surface, real hardware to test on, and a path to adoption.
The IOMMU management and epoch-based teardown are implementable today on CXL
2.0 hardware.

**5. Build the HFT proof of concept**
The highest-confidence near-term application. Implement consistency contracts
for network-to-compute data paths using existing RDMA infrastructure — no CXL
required. This validates the contract language design against a workload where
performance is precisely measurable and the economics justify the effort.

### Academic Venue Targets

- **HotOS** — position papers, early ideas; appropriate venue once Critical
  items are addressed
- **EuroSys** — systems research with implementation evidence; target after
  Linux subsystem prototype exists
- **OSDI / SOSP** — full research papers with strong evaluation; target after
  HFT or genomics proof of concept
- **ASPLOS** — architecture/OS intersection, hardware-software co-design;
  good fit for CXL-native work

### Collaboration Targets

- TU Dresden (M3 team) — heterogeneous OS design, directly related prior work
- CXL Consortium — hardware specification input, formal memory model access
- NVIDIA Research — GPU/NPU participation model
- Boyland / fractional permissions community — formal semantics grounding
- Any group working on CXL software stacks — the gap is acknowledged and open

---

## 12. References and Related Work

### Core Architecture References

- **M3:** Nötzli & Davidson, "M3: A Hardware/OS-Supported Attested Microkernel,"
  ASPLOS 2016. TU Dresden. Direct precedent for CPU-free execution model.

- **Barrelfish / Multikernel:** Baumann et al., "The Multikernel: A New OS
  Architecture for Scalable Multicore Systems," SOSP 2009. Distributed
  consistency model precedent.

- **Arrakis:** Peter et al., "Arrakis: The Operating System Is the Control Plane,"
  OSDI 2014. Control/data plane separation. Application direct I/O access.

- **Stramash:** ASPLOS 2025. Fused-kernel OS for heterogeneous-ISA processors
  over cache-coherent shared memory. Closest current work. Validates core premise.

- **XSched:** OSDI 2025. Heterogeneous XPU scheduling with unified abstraction.
  Complementary approach to same problem.

### Formal Theory References

- **Fractional Permissions:** Boyland, J., "Checking Interference with Fractional
  Permissions," SAS 2003. The formal grounding for the `CapabilityTensor`
  projection model. Read-capable fractions + write-capable whole maps directly
  onto `coherent` vs. `exclusive`. Recommended foundation for contract language
  semantics.

- **δ-Consistency:** Yu, H. & Vahdat, A., "Design and Evaluation of a Continuous
  Consistency Model for Replicated Services," OSDI 2000. Bounded-time visibility
  semantics — the formal predecessor to `coherent(A, B) within T`. Critically,
  this model also assumes a global clock, which is the same gap the tensorial
  contract language must address.

- **Lamport Clocks / Happens-Before:** Lamport, L., "Time, Clocks, and the
  Ordering of Events in a Distributed System," CACM 1978. The correct primitive
  for defining ordering in a distributed hardware system without a global clock.
  The `within T` guarantee should be redefined in terms of happens-before events
  observable by the hardware.

### Formal Verification References

- **seL4:** Klein et al., "seL4: Formal Verification of an OS Kernel," SOSP 2009.
  ACM TOCS 2014 comprehensive version. The formal verification bar to eventually
  reach. 8,700 lines of C, machine-checked functional correctness proof.

### Type System and Safety References

- **RaBAB-NeuSym Kernel:** 2025. Linear types as kernel resource primitive.
  Independent convergence with tensorial model's Rust ownership approach.

- **CapsLock:** CCS 2025. Revoke-on-use hardware capabilities. Independent
  implementation of projection model semantics.

### Hardware and Interconnect References

- **CXL Consortium:** "Opportunities and Challenges for Compute Express Link,"
  CXL Consortium whitepaper, November 2024.

- **CXL 4.0 Specification:** CXL Consortium, November 2025. 128 GT/s via
  PCIe 7.0. Multi-rack fabric. ABI Research: commercial adoption inflection
  point expected 2027.

- **ABI Research CXL Forecast:** CXL 3.0/3.1 solutions commercially available
  with sufficient software support by 2027.

### Application Domain References

- **AI I/O bottleneck:** 70% of training time on I/O, 40% GPU idle waiting
  for data. Memory bandwidth scaling 1.6× per two years vs. compute at 3×.

- **PNM-KV:** Processing-Near-Memory for KV cache. 21.9× throughput improvement
  by offloading token page selection to CXL-attached accelerators.

- **Clara Parabricks:** NVIDIA. Up to 60× speedup on GPU vs CPU-only for
  germline variant calling on A100. Validates GPU acceleration for genomics;
  inter-stage data movement remains unaddressed.

- **HFT latency:** Traditional kernels add 20–50 microseconds. HFT targets
  sub-10 microseconds. Industry already bypasses kernel via DPDK/RDMA.

---

## Appendix: Key Open Research Questions

### Contract Language (Critical — prerequisite for publication)

1. **How should `within T` be formally defined without a global clock?**
   The most promising approach: define `within T` in terms of
   hardware-observable happens-before events from the CXL formal memory model,
   with `T` as an engineering SLO verified against observed latency
   distributions rather than as a hard correctness guarantee. What is lost by
   this weakening, and is it acceptable?

2. **Is there a composition operator `C₁ ⊕ C₂` with a bounded complexity class?**
   Two Class C contracts can compose into a Class D problem. Either a global
   composition checker with polynomial-time complexity must be designed, or
   composition must be restricted. What is the most expressive restriction
   that preserves polynomial-time composition?

3. **How is physical aliasing detected at the kernel level?**
   Two independently constructed `CapabilityTensor` objects over overlapping
   physical ranges must be detected at contract submission time. What data
   structure should the kernel maintain? What is the complexity of overlap
   detection as a function of active contract count?

4. **Can the contract language be grounded in fractional permissions?**
   Boyland's fractional permissions (SAS 2003) provide a sound and complete
   type theory for read/write capability splitting. Can `coherent` and
   `exclusive` be defined as surface syntax over a fractional permission model
   extended with δ-consistency? What extensions are required?

5. **Is coherency transitivity implied or not, and at what cost?**
   If `coherent(A,B) within 50ns` and `coherent(B,C) within 50ns`, is
   `coherent(A,C) within 100ns` guaranteed? If yes, pairwise constraints grow
   O(n²). If no, applications will assume it and have latent bugs. The semantics
   must commit to one answer with formal justification.

### Architecture (Important)

6. **What is the minimum attestation infrastructure required?**
   CXL 3.0 adds integrity and data encryption. Is this sufficient for Stratum 2
   ratification, or does the witness chain require dedicated silicon?

7. **Can the Bayesian fabric model converge fast enough?**
   The jitter classification requires sufficient history to distinguish noise
   from failure. How much history is required? What are the false positive and
   false negative rates for the autocorrelation-based classifier?

8. **How does the topological mapper handle dynamic topology changes?**
   Hotplug of CXL devices, device failure, thermal-induced bandwidth reduction.
   The mapper must renegotiate contracts without disrupting executing workloads.
   What is the renegotiation protocol and its latency?

9. **What is the performance cost of epoch-based teardown?**
   The drain window adds latency to preemption. Is this acceptable for the
   workloads where the tensorial kernel provides the most benefit? What is the
   minimum achievable drain window on CXL 3.x hardware?

10. **Is failure attribution recoverable in the multi-party case?**
    Violations in a multi-party coherency ring are non-attributable without a
    global observer. Is there a partial attribution mechanism that narrows the
    suspect set sufficiently to guide targeted re-mapping?

### Verification (Long-term)

11. **Is there a path to formal verification?**
    seL4 required 20 person-years of proof effort for 8,700 lines of C. The
    tensorial kernel's distributed consistency model is significantly more
    complex. What subset of properties could be verified first (e.g., epoch
    teardown safety, physical aliasing freedom), and what proof techniques are
    appropriate (separation logic, TLA+, Iris)?
