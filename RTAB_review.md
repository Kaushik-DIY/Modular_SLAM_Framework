# RTAB-Map: Architectural Deep-Dive Reference

> **Purpose of this document.** This is the master reference for our future planning on extending the SLAM workspace into a multi-modal, RTAB-Map-inspired system. It describes how RTAB-Map (the upstream library by Labbé & Michaud) is designed and implemented, with extra depth on its memory management and map management. The report draws only from RTAB-Map's own authoritative sources (their papers, the official site, and the `introlab/rtabmap` source headers) and cites each claim. **No comparisons with our system and no integration suggestions are included** — those are deliberately deferred to a follow-up plan.
>
> Scope of sensor coverage: **RGB-D + 2D LiDAR fusion**, since that matches our future end-goal. Stereo/monocular and 3D LiDAR are mentioned only where they affect shared logic.
>
> Source legend used inline:
> - **[JFR'19]** = Labbé & Michaud, *RTAB-Map as an open-source lidar and visual SLAM library for large-scale and long-term online operation*, Journal of Field Robotics 2019 (arXiv 2403.06341)
> - **[IROS'11]** = Labbé & Michaud, *Memory management for real-time appearance-based loop closure detection*, IROS 2011 (arXiv 2407.15890)
> - **[Site]** = official site http://introlab.github.io/rtabmap/
> - **[Src: X]** = header file `corelib/include/rtabmap/core/X.h` from https://github.com/introlab/rtabmap (master branch)
> - **[Unclear]** = the source does not explicitly state this; the report flags the gap rather than guessing.

---

## Table of contents

1. [Overview & positioning](#1-overview--positioning)
2. [End-to-end data flow (one sensor cycle)](#2-end-to-end-data-flow-one-sensor-cycle)
3. [Memory management deep dive (STM / WM / LTM)](#3-memory-management-deep-dive-stm--wm--ltm)
4. [Map management deep dive (graph structure)](#4-map-management-deep-dive-graph-structure)
5. [Sensor input layer](#5-sensor-input-layer)
6. [Signature creation (the node payload)](#6-signature-creation-the-node-payload)
7. [Odometry sources](#7-odometry-sources)
8. [Appearance-based loop-closure detector](#8-appearance-based-loop-closure-detector)
9. [Retrieval mechanism (LTM → WM)](#9-retrieval-mechanism-ltm--wm)
10. [Geometric verification (visual + LiDAR)](#10-geometric-verification-visual--lidar)
11. [Graph optimization back-end](#11-graph-optimization-back-end)
12. [Map regeneration after optimization](#12-map-regeneration-after-optimization)
13. [Database persistence](#13-database-persistence)
14. [Key configuration parameters (appendix)](#14-key-configuration-parameters-appendix)
15. [Author-stated limitations](#15-author-stated-limitations)
16. [Sources](#16-sources)

---

## 1. Overview & positioning

RTAB-Map is described by its authors as *"a RGB-D, Stereo and Lidar Graph-Based SLAM approach based on an incremental appearance-based loop closure detector"* [Site]. It originated in 2011 as a stand-alone *loop closure detector* designed around a memory-management mechanism that bounded the loop-closure compute regardless of map size [IROS'11], and has since grown into a complete graph-SLAM framework that integrates:

- A loop-closure detector (visual bag-of-words + Bayesian filter) [JFR'19 §3.3]
- A graph optimizer (TORO / g2o / GTSAM with Vertigo robust kernels) [JFR'19, Site]
- A three-tier memory manager (STM / WM / LTM) [IROS'11, JFR'19 §3]
- Visual and lidar odometry modules (F2F, F2M for vision; S2S, S2M for lidar) [JFR'19 §3.1]
- Geometric verification (PnP-RANSAC for visual, ICP for point-cloud) [JFR'19 §3.3]
- 2D / 3D / OctoMap / dense-cloud / mesh map outputs assembled from per-node local occupancy grids [JFR'19 §3.4]
- A SQLite-backed database for full session persistence

The **central design problem** that motivates the architecture is stated in [IROS'11]: in large environments, *"only using a certain number of locations for loop closure detection so that real-time constraint can be satisfied"*. The memory manager is RTAB-Map's signature contribution; the rest of the framework is built around it.

---

## 2. End-to-end data flow (one sensor cycle)

The following diagram traces what happens to one synchronized sensor sample as it flows through RTAB-Map. Each arrow is one explicit step in the C++ implementation, and the labels reference the modules / paper sections cited next to them. The cycle runs at the *node-creation rate* `Rtabmap/DetectionRate` (default 2 Hz) [JFR'19], not at full sensor rate.

```
            ┌──────────────────────────────────────────────────────┐
            │ RAW SYNC SAMPLE                                       │
            │   • RGB image  + depth image  (+ calibration, TF)     │
            │   • 2D laser scan  (optional)                         │
            │   • Odometry pose + covariance  (external input)      │
            └─────────────────────────┬────────────────────────────┘
                                      │
                                      ▼
            ┌──────────────────────────────────────────────────────┐
            │ Rtabmap::process(data, odomPose, ...)  [Src: Rtabmap] │
            │   main entry; runs at DetectionRate                   │
            └─────────────────────────┬────────────────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────────┐
         │ Memory::update(data, odomPose)             [Src: Memory]    │
         │   Creates a new Signature for this sample, computes         │
         │   visual words via the incremental dictionary,              │
         │   compresses raw sensor data, and INSERTS the signature     │
         │   into STM.  Also adds a NEIGHBOR Link with the odom         │
         │   transform back to the previous STM node.                  │
         └────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────────┐
         │ Rehearsal check                            [IROS'11 §III-C] │
         │   Compare new signature to previous STM node.               │
         │   If similarity > Mem/RehearsalSimilarity (default 0.2),    │
         │   MERGE old node's weight into new node and delete old.     │
         └────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────────┐
         │ STM aging                                  [JFR'19 §3.2]    │
         │   If STM size > Mem/STMSize (default 30),                   │
         │   move OLDEST STM node into WM.                             │
         │   (That node becomes eligible for loop-closure scoring.)    │
         └────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────────┐
         │ Loop hypothesis computation       [Src: BayesFilter]        │
         │   • Memory::computeLikelihood(): TF-IDF score of new        │
         │     signature against every signature in WM.                │
         │   • BayesFilter updates posterior over "is current location │
         │     the same as past location Li ?".                        │
         │   • Hypothesis with highest p(St = i | Lt) is the candidate.│
         └─────────────┬───────────────────────────────┬──────────────┘
                       │                               │
            no candidate│             candidate above   │
            above Tloop ▼              Tloop = LoopThr  │
                  (idle)                                ▼
                                ┌──────────────────────────────────────┐
                                │ Retrieval from LTM                    │
                                │  Up to 2 neighbors of the candidate   │
                                │  node are pulled back from LTM into   │
                                │  WM. Dictionary is updated with their │
                                │  visual words. [IROS'11 §III-E]       │
                                └────────────────────┬─────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────┐
                                │ Geometric verification                │
                                │  1. RegistrationVis  (PnP-RANSAC)     │
                                │     between current and candidate;    │
                                │     requires >= Vis/MinInliers (20).  │
                                │  2. If scan or point cloud available, │
                                │     RegistrationIcp REFINES the link  │
                                │     transform on the point clouds.    │
                                │  Output: a Link of type kGlobalClosure│
                                │  with transform + information matrix. │
                                └────────────────────┬─────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────┐
                                │ Graph update                          │
                                │  Memory::addLink(loop_closure_link).  │
                                │  Proximity links (kLocalSpaceClosure) │
                                │  may also be added via LiDAR-only     │
                                │  proximity detection between WM nodes │
                                │  whose graph distance is bounded by   │
                                │  RGBD/ProximityMaxGraphDepth (50).    │
                                └────────────────────┬─────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────┐
                                │ Graph optimization                    │
                                │  Optimizer::optimize() runs on the WM │
                                │  subgraph. Constraints = all          │
                                │  neighbor + closure + proximity links.│
                                │  Outlier rejection on link transform  │
                                │  change vs RGBD/OptimizeMaxError.     │
                                └────────────────────┬─────────────────┘
                                                     │
                                                     ▼
            ┌────────────────────────────────────────────────────────────┐
            │ WM aging (transfer)                          [IROS'11 §III-D]│
            │   If processing time exceeds Rtabmap/TimeThr OR             │
            │   WM size > Rtabmap/MemoryThr, transfer the OLDEST of the   │
            │   LOWEST-WEIGHTED nodes from WM to LTM (asynchronous DB     │
            │   write). Words owned only by transferred signatures are    │
            │   removed from the active dictionary.                       │
            └────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────────┐
            │ Map output assembly                          [JFR'19 §3.4]  │
            │   Local occupancy grids attached to each WM signature are   │
            │   transformed by the optimized poses and stitched into the  │
            │   global 2D grid / OctoMap / point cloud / mesh output.     │
            └────────────────────────────────────────────────────────────┘
```

> **Note on the entry path:** Visual odometry (`rgbd_odometry` node) and lidar odometry (`icp_odometry` node) compute the input `odomPose` *before* `Rtabmap::process()` is called [JFR'19 §3.1]. From the perspective of the `Rtabmap` main loop, odometry is an *input*, not a step. The neighbor link's transform is the odometry-derived relative pose.

---

## 3. Memory management deep dive (STM / WM / LTM)

This is RTAB-Map's signature design and warrants the most detail. The three tiers are explicitly visible in the `Memory` class internals:

```cpp
// from corelib/include/rtabmap/core/Memory.h  [Src: Memory]
std::map<int, Signature*> _signatures;   // all in-RAM signatures
std::set<int>            _stMem;         // IDs of signatures in STM
std::map<int, double>    _workingMem;    // ID -> "last accessed" time stamp; signatures in WM
// LTM is implicit: any signature ID known to the DB but not in _signatures
```

Tier membership queries exist on the Memory class: `isInSTM(id)`, `isInWM(id)`, `isInLTM(id)`, `allNodesInWM()`, `memoryChanged()` [Src: Memory].

### 3.1 Short-Term Memory (STM)

**What it is.** A bounded queue of the most recent signatures. *"Locations in the Short-Term Memory (STM) are not used for loop closure detection, to avoid loop closure detection on locations that have just been visited"* [IROS'11].

**Why.** Two consecutive observations of the same place are almost identical and would always score as "loop closures" if compared. STM hides recent nodes from the Bayes filter so the hypothesis space is meaningful.

**Size bound.** Parameter `Mem/STMSize` (default 30 in [JFR'19], experimentally 25 in [IROS'11]). When full, *"the oldest location in STM is moved into WM"* [IROS'11].

**What STM stores.** Each STM entry is a full `Signature` (visual words, compressed sensor data, pose, weight, links). It is also the place where the **rehearsal** merge happens (see §3.4 below).

### 3.2 Working Memory (WM)

**What it is.** The set of signatures that are loaded in RAM and actively scored against each new query for loop closure. *"Keeping the most recent and frequently observed locations in the robot's Working Memory (WM)"* [IROS'11].

**What WM stores.** Same `Signature` objects as STM — the difference is policy, not representation. WM entries are eligible for:

- TF-IDF similarity scoring (computeLikelihood)
- Bayes-filter loop hypothesis posterior
- Geometric verification when chosen as candidate
- Inclusion in the optimized graph subset

**Size bound.** *Two* independent bounds, both off by default and enabled by the user:

- `Rtabmap/TimeThr` (default 0 ms = disabled). The wall-clock target per iteration. When iteration time exceeds it, a transfer happens.
- `Rtabmap/MemoryThr` (default 0 nodes = disabled). Hard cap on WM size in node count.

When either is exceeded, *"the oldest of the smallest weighted nodes are transferred to LTM first"* [JFR'19].

The transfer is asynchronous: *"Saving the transferred locations into the database is done asynchronously using a background thread"* [IROS'11], so the foreground SLAM loop is not blocked on disk I/O.

The implementation of WM as `std::map<int, double>` (signature ID → last-access timestamp) [Src: Memory] is what makes the "oldest among lowest-weight" choice efficient.

### 3.3 Long-Term Memory (LTM)

**What it is.** The on-disk SQLite database. Signatures whose IDs are *not* present in `_signatures` (RAM) but are recorded in the DB are considered to be in LTM [Src: Memory, by construction].

**What LTM holds.** Same data as a WM signature — visual words, compressed sensor data, pose, weight, links — but persisted in compressed form (image, depth, scan all stored as compressed blobs; see §13).

**Transfer-in.** Triggered by the WM aging rule (above). The signature record is written to the DB and removed from `_signatures` and `_workingMem`. *"A signature of a location transferred to LTM removes its word references from the visual dictionary. If a word does not have reference to a signature anymore, it is transferred into LTM"* [IROS'11].

**Retrieval-out.** Triggered by the loop-closure hypothesis (see §9).

### 3.4 Weight assignment and the rehearsal mechanism

Weights are the basis of the transfer policy ("oldest of the lowest-weighted") and are also how RTAB-Map distinguishes a place the robot dwelled in from a place it just passed.

**Initialization.** *"A location Lt is then created with signature zt, a weight initialized to 0"* [IROS'11].

**Rehearsal merge (a STM event).** When the new STM signature `zt` and its immediate predecessor `zc` have similarity above `Mem/RehearsalSimilarity`:

1. Words of `zc` are dropped; `zt` keeps its own words [IROS'11].
2. *"The weight of Lt is increased by the one of Lc plus one, the neighbor links of Lc are added to Lt and Lc is deleted from STM"* [IROS'11].
3. Net effect: a stationary or near-stationary robot collapses repeated observations into one node whose weight grows.

This is the source of the "frequently observed = high weight" property.

**Loop-closure weight bump.** *"The weight of Lt is increased by the one of Li plus one"* when a loop closure is confirmed against past location `Li` [IROS'11].

**Selection for transfer.** *"When transfer occurs, the location with the lowest weight is selected. If there are locations with the same weight, then the oldest one is selected to be transferred"* [IROS'11].

**Heuristic rationale.** *"Identify the locations seen more frequently than others and that are more likely to cause loop closure detections"* [IROS'11].

### 3.5 Memory transition diagram

```
                  ┌────────────────────────┐
NEW sample ─────► │  STM                   │
                  │  (size <= Mem/STMSize)  │
                  │  bounded queue          │
                  │  protected from loop    │
                  │  closure scoring        │
                  └─────────────┬──────────┘
                                │  age-out: oldest signature
                                ▼  (after each new insert when full)
                  ┌────────────────────────┐
                  │  WM                    │
                  │  size = dynamic        │  ◄────── retrieve up to 2
                  │  size cap = TimeThr or │           neighbors when
                  │     MemoryThr (both    │           Bayes filter picks
                  │     opt-in)            │           a candidate that
                  │  used for loop closure │           sits in LTM
                  │  & graph optimization  │
                  └─────────────┬──────────┘
                                │ transfer: oldest of lowest-weight
                                ▼ (async DB write, background thread)
                  ┌────────────────────────┐
                  │  LTM = SQLite database │
                  │  persistent on disk    │
                  │  compressed signatures │
                  └────────────────────────┘
```

### 3.6 Key class methods that implement these rules

From `Memory.h` [Src: Memory]:

| Method | What it does |
|---|---|
| `update(SensorData, odomPose, ...)` | Adds a new signature; runs rehearsal, STM aging |
| `computeLikelihood(signature, ids)` | TF-IDF scoring of a query against listed WM IDs |
| `forget()` / `cleanup()` | Run the WM → LTM transfer step |
| `moveSignatureToWMFromSTM(int)` | STM → WM transition (called on STM overflow) |
| `reactivateSignatures(ids)` | LTM → WM (loads from DB back into RAM) |
| `moveToTrash(Signature*)` | Drop a signature entirely (also frees its dictionary words) |
| `isInSTM(int)`, `isInWM(int)`, `isInLTM(int)` | Tier membership queries |

The `_maxStMemSize`, `_recentWmRatio`, `_similarityThreshold`, `_transferSortingByWeightId` member fields hold the configuration that drives the above [Src: Memory].

---

## 4. Map management deep dive (graph structure)

The RTAB-Map "map" is a *pose graph* whose vertices are signatures and whose edges are typed links.

### 4.1 Node (Signature) payload

From `Signature.h` [Src: Signature], one signature carries:

```cpp
int             _id;                  // unique signature ID
int             _mapId;               // map / session this belongs to
double          _stamp;               // timestamp
int             _weight;              // memory-management weight (see §3.4)
std::string     _label;               // optional human-readable label
Transform       _pose;                // estimated 6-DoF pose
Transform       _groundTruthPose;     // ground truth pose (if any)
std::vector<float> _velocity;         // 6-D velocity (vx,vy,vz,vroll,vpitch,vyaw)

// Visual feature payload
std::multimap<int,int>    _words;          // visual-word ID -> keypoint index
std::vector<cv::KeyPoint> _wordsKpts;      // 2D keypoints
std::vector<cv::Point3f>  _words3;         // 3D positions in base_link frame
cv::Mat                   _wordsDescriptors; // descriptors matrix

// Graph connectivity
std::multimap<int,Link>   _links;          // links to other signatures
std::map<int,Link>        _landmarks;      // landmark associations

// Raw sensor payload
SensorData                _sensorData;     // compressed RGB, depth, scans, etc.

// Bookkeeping
bool _saved, _modified;
```

A signature is therefore *not* just a pose: it carries the full evidence (visual words, raw scans/images, descriptors) needed to recompute a loop-closure or regenerate the occupancy grid for that pose at any later time.

### 4.2 Link types

From `Link.h` [Src: Link]:

```cpp
enum Type {
    kNeighbor,           // sequential, odometry-derived
    kGlobalClosure,      // appearance-based loop closure
    kLocalSpaceClosure,  // proximity-based closure (geometric, often LiDAR)
    kLocalTimeClosure,   // closure between temporally close nodes
    kUserClosure,        // user-manually inserted
    kVirtualClosure,     // synthetic / placeholder
    kNeighborMerged,     // resulting from rehearsal merges
    kPosePrior,          // absolute pose constraint (From == To)
    kLandmark,           // observation of a named landmark
    kGravity,            // gravity-direction constraint (From == To)
    kEnd
};

int       from_;            // source node ID
int       to_;              // target node ID
Transform transform_;       // measured relative pose
Type      type_;            // see enum above
cv::Mat   infMatrix_;       // information matrix = inverse covariance
cv::Mat   _userDataCompressed, _userDataRaw;
```

**Semantic summary** (from the source comments and [JFR'19] §3.2 / §3.3):

| Type | Created by | Constrains |
|---|---|---|
| `kNeighbor` | STM neighbor insertion, from the odometry input | Two consecutive nodes' relative pose |
| `kNeighborMerged` | Rehearsal merge in STM | Compound neighbor link inheriting from a merged node |
| `kGlobalClosure` | Visual loop-closure pipeline (Bayes filter + RANSAC PnP, optionally + ICP) | A revisit constraint between non-consecutive nodes anywhere in WM |
| `kLocalSpaceClosure` | Proximity detection (LiDAR-driven) between WM nodes within `RGBD/ProximityMaxGraphDepth` graph hops | Closure when visual matching fails but LiDAR sees the same geometry |
| `kLocalTimeClosure` | Closure between temporally close nodes | Short-range temporal proximity |
| `kUserClosure` | External / manual input | User-supplied constraint |
| `kVirtualClosure` | Programmatic / placeholder | Synthetic link (does not act as a real measurement) |
| `kPosePrior` | External absolute-pose source | Absolute world-frame pose (From == To, unary) |
| `kLandmark` | Detection of a tagged feature | Pose of base relative to a landmark |
| `kGravity` | IMU gravity direction | Roll/pitch of the base (From == To, unary) |

### 4.3 The graph and which subgraph is optimized

The full graph spans every signature ever recorded (RAM + LTM). The **optimized** subgraph is the one whose nodes are currently in WM, *plus* any LTM neighbors retrieved during the current iteration (§9). This is what makes the per-iteration optimization cost bounded by WM size, not by total map size.

When a new loop closure or proximity link is added, *"graph optimization propagates the computed error to the whole graph, to decrease odometry drift"* [JFR'19], operating on the WM subgraph; transferred LTM nodes inherit corrected positions next time they are retrieved.

### 4.4 Multi-session

Each session has its own `_mapId` on every signature [Src: Signature]. When a robot starts a new session, *"a transformation between the two maps can be computed"* via a loop closure between the new map and a signature retrieved from a previous-session LTM record [JFR'19]. Sessions are first-class — RTAB-Map can resume a database and continue mapping with all prior LTM available.

---

## 5. Sensor input layer

### 5.1 Required inputs to `rtabmap` node [JFR'19 §3]

- One camera input: registered RGB-D images **or** stereo pair **or** multiple RGB-D cameras (must have same image size)
- Calibration messages
- TF (sensor → robot base)
- Odometry (3-DoF or 6-DoF, with covariance)

### 5.2 Optional inputs

- 2D LiDAR scan (enables 2D occupancy grid output and LiDAR-based proximity detection)
- 3D point cloud
- IMU (used by some odometry approaches and for the `kGravity` link)

### 5.3 Synchronization

Two strategies [JFR'19]:

- **Exact sync** — required for same-sensor pairs that publish on the same clock (e.g., stereo left/right).
- **Approximate sync** — used across heterogeneous sensors, with a minimum delay error tolerance.

A separate `rgbd_sync` ROS nodelet *"can be used to synchronize camera topics into a single topic"* upstream of `rtabmap`, so the main node sees exact-sync RGB-D plus approximate-sync LiDAR.

### 5.4 Cadence

The `Rtabmap` main loop runs at `Rtabmap/DetectionRate` (default 2 Hz) [JFR'19]. Sensor data may arrive at much higher rates; the node-creation cadence decouples the SLAM bookkeeping from the sensor stream. Odometry runs at full sensor rate in its own node.

---

## 6. Signature creation (the node payload)

### 6.1 Visual feature extraction

Supported detectors / descriptors [JFR'19]: SURF, SIFT, ORB, BRIEF, BRISK (selectable). The OpenCV implementations are used.

Parameter `Kp/MaxFeatures` (default 500) caps the number of keypoints retained for loop closure per signature. When the active odometry is `F2F` or `F2M`, RTAB-Map *reuses* a subset of the odometry features (the highest-response ones) for the loop-closure signature, avoiding double extraction [JFR'19].

### 6.2 Visual word vocabulary (the incremental dictionary)

Each new descriptor is matched against an *incremental* dictionary: if a near match is found (by NN distance ratio) the descriptor inherits that word ID, otherwise it becomes a new word. This is the basis for TF-IDF scoring (§8). The dictionary lives in RAM and is automatically shrunk when signatures are transferred to LTM (§3.3 — words referenced only by transferred signatures are removed). No external pre-training is needed [JFR'19].

### 6.3 LiDAR attachment

When a 2D laser scan is available, it is attached to the same signature in the `SensorData` payload [Src: Signature]. The same signature therefore carries both a visual-word set and a raw scan — they are not separate map entities. This unification is what enables RTAB-Map's *cross-modal verification*: a visual hypothesis can be confirmed by ICP on the scans of the two signatures (§10).

### 6.4 Local occupancy grid attached at creation

For each new signature, RTAB-Map computes a *local* occupancy grid in the node's own frame. Three modes [JFR'19 §3.4]:

- **2D ray tracing from LiDAR** (`Grid/FromDepth=false`, scan available): per-ray Bresenham trace marking free/occupied cells. Fast.
- **3D processing from depth** (`Grid/3D=true`): voxel-downsampled point cloud, ground-plane segmentation by normal angle (threshold `Grid/MaxGroundAngle`), split into ground vs. obstacle.
- **2D projection of 3D** (`Grid/3D=false`, no scan): ground / obstacle clouds projected to the x-y plane.

The local grid is stored on the signature and persists into LTM. *"Pre-computing local grids greatly decreases the regeneration time of the global occupancy grid when the graph has been optimized"* [JFR'19], because regeneration after a loop closure becomes *transform-and-merge* instead of *re-ray-trace*.

---

## 7. Odometry sources

Odometry is an *input* to `Rtabmap::process()`. RTAB-Map ships its own odometry implementations for the standard sensors but accepts any external source via its odometry-pose API.

### 7.1 Visual odometry [JFR'19 §3.1.1]

Two algorithms:

| Mode | Reference | Use when |
|---|---|---|
| **F2F** Frame-to-Frame | Last keyframe | Lower latency, less drift recovery |
| **F2M** Frame-to-Map | Local feature map (size cap `OdomF2M/MaxSize` = 2000) | Better robustness, slightly heavier |

Pipeline: GFTT keypoints (`Vis/MaxFeatures` = 1000) → optical flow (F2F) or NNDR matching (F2M, `Vis/CorNNDR` = 0.6) → constant-velocity motion prediction (`Vis/CorGuessWinSize` = 20 pixels) → PnP-RANSAC requiring `Vis/MinInliers` = 20 → local bundle adjustment → keyframe update when inlier ratio falls below `Odom/KeyFrameThr` (F2F: 0.6, F2M: 0.3).

Covariance is estimated by the *median absolute deviation* (MAD) between 3D feature correspondences.

Seven external visual-odometry packages can also be plugged in as the odometry source: FOVIS, Viso2, DVO, OKVIS, MSCKF, ORB-SLAM2 (with its loop closure disabled), and Project Tango.

### 7.2 LiDAR odometry [JFR'19 §3.1.2]

Two algorithms, mirroring the visual pair:

| Mode | Reference | Use when |
|---|---|---|
| **S2S** Scan-to-Scan | Last keyframe scan | Simpler |
| **S2M** Scan-to-Map | Rolling point-cloud map (size cap `OdomF2M/ScanMaxSize` = 10000) | Better drift behavior |

Pipeline: filtering + normals → ICP via `libpointmatcher` (point-to-point or point-to-plane) → motion prediction from previous registration or external odometry → structural-complexity check (PCA on normals; if too low → orientation-only update, position from external odometry) → keyframe update when correspondence ratio falls below `Odom/ScanKeyFrameThr` (0.9). In S2M, the new scan is subtracted from the map with radius `OdomF2M/ScanSubtractRadius` (0.05 m) before the remainder is added.

Important property quoted verbatim: *"Lidar odometry cannot recover from being lost when the motion prediction is null... must then be reset"* [JFR'19].

### 7.3 External odometry

Wheel encoders, IMU, an external visual-inertial estimator — any source producing a TF pose with optional covariance can drive the `odomPose` argument of `Rtabmap::process()`.

---

## 8. Appearance-based loop-closure detector

### 8.1 TF-IDF similarity

For each new signature, `Memory::computeLikelihood()` produces a TF-IDF score against every signature currently in WM [Src: Memory]. TF-IDF down-weights words that occur in many signatures (and are therefore uninformative) and up-weights rare distinctive words [JFR'19].

### 8.2 Bayes filter

A discrete Bayes filter maintains a posterior `p(St = i | Lt)` over the hypothesis "the current observation `Lt` was generated at past location `Li`" [JFR'19 §3.3, IROS'11 §III-F]. The state space includes one hypothesis per WM signature plus a *virtual place* hypothesis ("this is a new location"). Each iteration:

1. The TF-IDF likelihood becomes the observation model.
2. A neighborhood-aware prediction propagates probability mass along the graph neighbors of each hypothesis (so revisits to a region accumulate support).
3. The posterior is normalized.
4. The hypothesis with the highest posterior is checked against `Rtabmap/LoopThr` (default 0.11 in [JFR'19]; `Tloop` = 10% in [IROS'11]).
5. A minimum WM-size guard `TminHyp` (15 in [IROS'11]) prevents premature acceptance.

### 8.3 What happens when the candidate sits in LTM

If the top hypothesis points at a signature that is *not currently in WM* but is in LTM, the retrieval mechanism (§9) pulls it back in along with up to two of its graph neighbors, and the geometric verification step runs against the now-loaded signatures.

### 8.4 Outlier rejection at the graph level

A late safeguard, added in [JFR'19]: *"If a link's transformation in the graph after optimization has changed more than [a] factor `RGBD/OptimizeMaxError` of its translational variance, all loop closure and proximity links added by the new node are rejected, keeping the optimized graph as if no loop closure happened"* [JFR'19].

`RGBD/OptimizeMaxError` defaults to 1. This rejects loop closures whose acceptance would force unreasonably large pose corrections — a strong defense against false positives.

---

## 9. Retrieval mechanism (LTM → WM)

When the Bayes filter selects a candidate sitting in LTM, RTAB-Map *reactivates* it:

1. *"Neighbors of location Li in LTM are transferred back into WM if `p(St = i | Lt)` is the highest probability"* [IROS'11].
2. *"A maximum of two locations are retrieved at each iteration"* — this bounds the per-iteration disk-load cost [IROS'11].
3. The visual dictionary is augmented with these signatures' words. Words previously dropped during the original transfer are reloaded; if the same descriptor now hashes to a more recent word ID, the mapping is updated.
4. *"The retrieved locations of the current iteration are not allowed to be transferred"* during the same iteration, preventing a thrash where a retrieved node is immediately re-transferred.

The implementation entry point is `Memory::reactivateSignatures(ids)` [Src: Memory], which reads from the database and re-inserts the signatures into `_signatures` and `_workingMem`.

---

## 10. Geometric verification (visual + LiDAR)

This is the step most directly relevant to the multi-modal future plan. Once the Bayes filter has chosen a candidate, the appearance match must be **confirmed geometrically** before becoming a `kGlobalClosure` link.

### 10.1 Visual verification

Done by `RegistrationVis` [Src reference per JFR'19 §3.3]. The pipeline is the **same** as the visual odometry PnP step:

1. Match descriptors between current signature and candidate signature (NNDR ratio test).
2. Solve PnP via RANSAC.
3. Require ≥ `Vis/MinInliers` (20) inliers.
4. Output: a 6-DoF transform + an information matrix derived from inlier residuals.

If this step fails, the hypothesis is rejected as a false appearance match.

### 10.2 LiDAR refinement (when a scan / point cloud is attached)

Quoted verbatim: *"When a laser scan or a point cloud is available, link's transformation is refined using the same ICP Registration approach than with lidar odometry"* [JFR'19].

`RegistrationIcp` therefore *refines* the link transform output by `RegistrationVis`, using the scans attached to the two signatures. ICP variants used: point-to-point and point-to-plane via `libpointmatcher`. The structural-complexity check (PCA on normals) applies here as well — if the scan environment is degenerate (e.g., long featureless corridor), only orientation is refined and translation is left at the visual estimate.

### 10.3 The cross-modal verification path

```
                ┌────────────────────────────────────────┐
                │ Bayes filter picks candidate node Lc   │
                └──────────────────┬─────────────────────┘
                                   │
                                   ▼
                ┌────────────────────────────────────────┐
                │ RegistrationVis (RANSAC + PnP) on the  │
                │ visual word descriptors                │
                └──────────────────┬─────────────────────┘
                                   │
                       ┌───────────┴────────────┐
                no inliers│            inliers   │
                          ▼                      ▼
                 ┌─────────────────┐   ┌────────────────────────────┐
                 │ reject — no     │   │ initial transform T_vis     │
                 │ loop closure    │   │ + information matrix        │
                 └─────────────────┘   └────────────┬───────────────┘
                                                    │
                              scans available?      │
                                  no                ▼
                                  ┌──────────────────────────────┐
                                  │ RegistrationIcp on attached  │
                                  │ scans, seeded at T_vis        │
                                  │ → refined T_final             │
                                  └────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Emit Link{from=Lt, to=Lc,         │
                              │   transform=T_final or T_vis,     │
                              │   type=kGlobalClosure}            │
                              └──────────────────────────────────┘
```

### 10.4 Proximity detection — a LiDAR-only path that bypasses appearance entirely

Independent of the visual Bayes filter, RTAB-Map runs a *proximity detection* loop: for each pair of WM nodes whose graph distance is less than `RGBD/ProximityMaxGraphDepth` (default 50) and that lie close in current world coordinates, an ICP match is attempted between their scans. If accepted, a `kLocalSpaceClosure` link is added. Quoted use case: *"Traversing back the same corridor in a different direction, during which the camera cannot be used to find loop closures"* [JFR'19].

This mode is the practical reason RTAB-Map remains usable for *"long-term"* operation under viewpoint changes that destroy visual appearance.

---

## 11. Graph optimization back-end

Three optimizer back-ends are supported [JFR'19, Site]:

| Optimizer | Property (per JFR'19) |
|---|---|
| **TORO** | Slower convergence, more robust to poorly estimated odometry covariance, robust to multi-session merging |
| **g2o** | Faster convergence, better single-map quality (especially 6-DoF), supports robust kernels via Vertigo |
| **GTSAM** | *"Slightly more robust to multi-session than g2o, and thus is the strategy now used by default"* |

Ceres is **not** listed in the published RTAB-Map architecture descriptions; the three above are the canonical options.

**When optimization runs.** It is triggered whenever (a) a new loop closure or proximity link is added, or (b) nodes are retrieved from LTM or transferred to LTM (because the optimized subgraph composition changed).

**What is fixed.** The first node of the current map (`_mapId` = first session) is fixed as the world-frame anchor by default. Additional `kPosePrior` (unary) constraints can pin further nodes if available.

**What is optimized.** The subgraph induced by the current WM signatures plus any LTM nodes retrieved in this iteration. This is the bounded-compute property: the optimizer never sees the whole-life graph at once, only a working subset.

**What propagates back.** After convergence, *"graph optimization propagates the computed error to the whole graph, to decrease odometry drift"* [JFR'19] — the corrected poses are written back to the signatures in WM. LTM nodes' poses are not touched at this moment; they get corrected lazily next time they are retrieved (since retrieved poses are recomputed relative to the optimized WM).

**Robust kernels.** g2o and GTSAM are used with Vertigo's switchable constraints, which down-weight a constraint whose residual is implausibly large [Site].

---

## 12. Map regeneration after optimization

This is the property that makes RTAB-Map's map output globally consistent without re-tracing every scan after a loop closure.

### 12.1 Per-node local grids (already created at signature time)

See §6.4. The grids live attached to each signature and are persisted in LTM along with the signature.

### 12.2 Global assembly procedure

For each signature `S_i` in the current WM (with its optimized pose `T_i`):

1. Look up `S_i.local_grid` (already in node-local frame).
2. Transform into world: cells whose centers fall into the global grid are placed at `T_i ⋅ cell_center`.
3. Merge into the global map by clearing and re-adding obstacles in the cells touched.

For the **3D OctoMap** path, the same loop happens but with 3D ray tracing per node from its local point cloud, building an octree.

For the **dense point cloud** output, the per-node clouds are concatenated and then voxel-filtered to merge overlapping surfaces [JFR'19].

### 12.3 What happens after a loop closure

The optimizer changes `T_i` for many nodes. The map output is then *re-assembled* from scratch using the new poses. Because the expensive step (local grid construction, ray tracing of each scan) has already been amortized at signature creation time, the post-loop-closure assembly is cheap — it is just transform-and-merge over WM-size signatures.

The corresponding quote: *"When a loop closure occurs, the global map should be re-assembled according to all new optimized poses for all nodes in the map's graph"* [JFR'19].

### 12.4 Output formats

| Format | Requirement | Source |
|---|---|---|
| 2D occupancy grid | 2D local grids on signatures (from scan or projected 3D) | [JFR'19 §3.4] |
| 3D OctoMap | 3D local grids on signatures | [JFR'19] |
| 2D projection of 3D OctoMap | 3D local grids, projected down at output time | [JFR'19] |
| Dense point cloud | Per-node clouds + voxel filter | [JFR'19] |
| Textured mesh | Surface reconstruction over assembled cloud | [Site] |

---

## 13. Database persistence

RTAB-Map persists the *entire* state to a SQLite database via `DBDriverSqlite3` [Src: DBDriverSqlite3]. The session can be paused, closed, reopened, and resumed; LTM is just "the part of the database not currently loaded into RAM."

### 13.1 What is persisted

Tables inferred from `DBDriverSqlite3`'s public surface [Src: DBDriverSqlite3 — schema not fully documented in this header; the list below is what the method names imply]:

- Nodes / signatures table — ID, map ID, timestamp, pose, ground-truth pose, weight, label, velocity, GPS / environmental sensors when present
- Images table — RGB image, compressed
- Depth images table — depth image, compressed
- Laser scans table — scan data, compressed
- Links table — source ID, target ID, transform, type, information matrix, user data
- Words / visual dictionary table — incremental dictionary entries
- Keypoints table — per-signature 2D + 3D keypoints with descriptors
- Calibration table — camera intrinsics, baselines
- Occupancy grids table — per-signature local grids
- Statistics table — per-iteration diagnostics
- Global descriptors table — optional per-signature global descriptor

> [Unclear] The exact SQL schema (`CREATE TABLE` statements, column types, foreign keys, indexes, versioning) is not in the public header; it lives in `corelib/src/DBDriverSqlite3.cpp`. A second pass into the .cpp implementation would be required to nail down the exact column-level schema.

### 13.2 Compression

All raw sensor payloads (RGB image, depth, LiDAR scan) are stored as compressed blobs. RGB/depth images use OpenCV's standard image-codec compression; LiDAR scans are stored as compressed point arrays. The `cv::Mat` ↔ compressed-blob conversion is centralized in `SensorData`. Decompression happens lazily on retrieval, so DB load is cheap when only metadata is needed.

### 13.3 Asynchronous writes

LTM transfers are written by a *background thread* [IROS'11]. The foreground SLAM loop never blocks on disk I/O.

### 13.4 Multi-session resumption

Each signature carries `_mapId` [Src: Signature]. Reopening a DB exposes all prior sessions' nodes as eligible LTM. When the robot triggers a loop closure into a previous-session node, the two `mapId`s are linked by a `kGlobalClosure` and the optimizer aligns them into one consistent world frame [JFR'19].

---

## 14. Key configuration parameters (appendix)

Grouped by subsystem. Defaults are as published in [JFR'19] unless noted.

### 14.1 Memory management

| Parameter | Default | Effect |
|---|---|---|
| `Rtabmap/DetectionRate` | 2 Hz | Node-creation cadence |
| `Mem/STMSize` | 30 nodes (25 in [IROS'11]) | STM bound |
| `Rtabmap/TimeThr` | 0 ms (disabled) | Per-iteration wall-clock cap → triggers WM transfer |
| `Rtabmap/MemoryThr` | 0 nodes (disabled) | Hard WM size cap → triggers WM transfer |
| `Mem/RehearsalSimilarity` | 0.2 | Threshold for STM rehearsal merge |
| `Mem/IncrementalMemory` | true | If false, RTAB-Map runs in localization-only mode (no DB writes) |

### 14.2 Loop closure (Bayes / appearance)

| Parameter | Default | Effect |
|---|---|---|
| `Rtabmap/LoopThr` | 0.11 | Bayes posterior acceptance threshold |
| `Kp/MaxFeatures` | 500 | Features kept per signature for loop closure |
| `Kp/DetectorStrategy` | (impl-default) | SURF / SIFT / ORB / BRIEF / BRISK |

### 14.3 Visual odometry / verification

| Parameter | Default | Effect |
|---|---|---|
| `Vis/MaxFeatures` | 1000 | Features for visual odometry |
| `Vis/MinInliers` | 20 | PnP-RANSAC inlier minimum |
| `Vis/CorNNDR` | 0.6 | NN distance ratio for descriptor matching |
| `Vis/CorGuessWinSize` | 20 px | Constant-velocity search window |
| `Odom/KeyFrameThr` | 0.6 (F2F) / 0.3 (F2M) | Inlier ratio that triggers keyframe |
| `OdomF2M/MaxSize` | 2000 features | Feature-map size cap (F2M) |

### 14.4 LiDAR odometry / verification

| Parameter | Default | Effect |
|---|---|---|
| `Odom/ScanKeyFrameThr` | 0.9 | Correspondence ratio threshold for scan keyframe |
| `OdomF2M/ScanMaxSize` | 10000 points | S2M map size cap |
| `OdomF2M/ScanSubtractRadius` | 0.05 m | Radius for S2M map subtraction |
| `Icp/PointToPlaneMinComplexity` | (param) | Structural-complexity threshold below which only orientation is solved |

### 14.5 Proximity & graph

| Parameter | Default | Effect |
|---|---|---|
| `RGBD/ProximityMaxGraphDepth` | 50 nodes | Graph-distance bound for proximity pairs |
| `RGBD/OptimizeMaxError` | 1 | Post-opt link-change rejection factor (multiple of translational variance) |

### 14.6 Grid output

| Parameter | Default | Effect |
|---|---|---|
| `Grid/CellSize` | (impl-default) | Local grid resolution |
| `Grid/FromDepth` | (boolean) | Build grids from depth (true) or from laser scan (false) |
| `Grid/3D` | (boolean) | Build 3D grids (true) or 2D (false) |
| `Grid/MaxGroundAngle` | (degrees) | Normal-angle tolerance for ground classification |

### 14.7 Optimizer

| Parameter | Default | Effect |
|---|---|---|
| `Optimizer/Strategy` | GTSAM (per [JFR'19]) | Select TORO / g2o / GTSAM |
| `Optimizer/Robust` | (boolean) | Enable Vertigo robust kernels |

---

## 15. Author-stated limitations

Quoted directly from the source material:

- LiDAR odometry recovery: *"Lidar odometry cannot recover from being lost when the motion prediction is null... must then be reset"* [JFR'19].
- Real-time guarantee mode: *"Online processing: output of the SLAM module should be bounded to a maximum delay after receiving sensor data"* — RTAB-Map enforces this via WM transfers, but if `TimeThr`/`MemoryThr` are left at their defaults (0 = disabled), unbounded WM growth in very long sessions is possible.
- Dictionary rebuild cost: *"The most expensive step of our algorithm is to rebuild the visual dictionary"* [IROS'11]. Long sessions with high retrieval traffic can be bottlenecked here.
- Time-bound guideline: *"Ttime can be set to about 200 to 400 ms smaller to the image acquisition rate at 1 Hz... For an image acquisition rate of 1 Hz, Ttime could be set between 600 ms to 800 ms"* [IROS'11].
- Multi-session merging is *"slightly more robust"* with GTSAM than g2o [JFR'19] — implying g2o can fail on multi-session graphs with poorly estimated covariances.
- Proximity detection has a drift caveat: *"prevents invalid detections when odometry drift causes robot to occupy previously mapped areas"* is the *design* — but the bound `RGBD/ProximityMaxGraphDepth = 50` is a tuning compromise: too high → false closures during drift, too low → missed proximities after long traversals.

---

## 16. Sources

Primary:
- W. Labbé & F. Michaud, **"RTAB-Map as an open-source lidar and visual simultaneous localization and mapping library for large-scale and long-term online operation"**, *Journal of Field Robotics* 36(2), 2019. arXiv: https://arxiv.org/abs/2403.06341 (HTML: https://arxiv.org/html/2403.06341v1)
- W. Labbé & F. Michaud, **"Memory management for real-time appearance-based loop closure detection"**, *Proc. IEEE/RSJ IROS*, 2011. arXiv mirror: https://arxiv.org/abs/2407.15890 (HTML: https://arxiv.org/html/2407.15890v1)
- RTAB-Map official site: http://introlab.github.io/rtabmap/
- IntRoLab project page: https://introlab.3it.usherbrooke.ca/index.php/RTAB-Map

Source-code headers consulted (from https://github.com/introlab/rtabmap, master branch, `corelib/include/rtabmap/core/`):
- `Memory.h` — STM/WM/LTM data structures and methods
- `Rtabmap.h` — main `process()` loop and subsystem ownership
- `Signature.h` — node payload (visual words, scans, pose, weight, links)
- `Link.h` — link types and edge payload
- `DBDriverSqlite3.h` — persistence interface

Secondary (cited contextually only):
- W. Labbé & F. Michaud, "Online global loop closure detection for large-scale multi-session graph-based SLAM," IROS 2014.
- W. Labbé & F. Michaud, "Long-Term Online Multi-Session Graph-Based SPLAM with Memory Management," *Autonomous Robots*, 2018.

> **Open follow-ups for this reference** (gaps that would require additional source-code reads to close, flagged here for completeness):
> 1. Exact SQL schema in `DBDriverSqlite3.cpp` (table-level column definitions, indexes, foreign keys).
> 2. Step-by-step implementation of `RegistrationVis::computeTransformation` and `RegistrationIcp::computeTransformation` (`corelib/src/Registration*.cpp`).
> 3. Concrete behavior of `BayesFilter::computePosterior` — the exact transition model and the virtual-place hypothesis weighting.
> 4. The "neighborhood smoothing" step in the Bayes prediction (how probability mass propagates over the WM subgraph).
>
> These can be added in a future revision of this document if the level of detail is needed for implementation planning.
