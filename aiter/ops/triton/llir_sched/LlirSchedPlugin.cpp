//############################################################################
// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//############################################################################

// Vendored from ROCm/gfx950-gluon-tutorials plugins/llir_scheduler at ef48b5f;
// the only change is the LLIR_SCHED_REQUIRE_OPT_IN gate in LlirSchedPass::run.
//
// Out-of-tree LLVM new-PassManager pass plugin: the gfx950 LLIR scheduler
// (MFMA <-> memory interleave for GEMM hot loops), ported from AMD-Triton
// triton-mi450 PR #73 (the sched.barrier variant). Self-contained: depends only
// on LLVM headers. Load into Triton via LLVM_PASS_PLUGIN_PATH; it auto-inserts
// at the OptimizerLast extension point of make_llir's optimize_module O3 run.
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdlib>

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

// Inlined from PR#73's TritonAMDGPUToLLVM/MfmaUtility.h so the plugin needs no
// triton headers.
namespace mlir::triton::AMD {
inline bool isMFMAorWMMA(const llvm::Instruction &I) {
  const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  const llvm::Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  llvm::StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}
} // namespace mlir::triton::AMD

namespace {

using namespace llvm;

// Classification of an instruction for scheduling purposes.
enum class SchedKind { MFMA, GR, LR, LW, Other };

// LDS resides in address space 3 on AMDGPU.
constexpr unsigned kLDSAddressSpace = 3;

// Structures used for region analysis/scheduling
struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

struct MFMARegionInfo {
  Instruction *RegionStart = nullptr;
  unsigned TotalMFMA = 0;
};

using MFMARegionList = SmallVector<MFMARegionInfo, 8>;
using BBMFMAAnalysisMap = DenseMap<const BasicBlock *, MFMARegionList>;

struct BBRegion {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr; // First instruction in region (inclusive)
  Instruction *End =
      nullptr; // First instruction of next region or nullptr (exclusive)
};

struct MFMARegionCollectResult {
  // MFMA-input prep instructions to hoist to the region start.
  SmallVector<Instruction *, 16> Hoist;
  // MFMA-result users to sink toward the region end.
  SmallVector<Instruction *, 16> Sink;
  // Last memory anchor seen while collecting (used to place trailing MFMAs).
  Instruction *LastAnchor = nullptr;
  // Memory ops (GR/LR/LW) that the region's MFMAs are spaced around.
  SmallVector<AnchorInst, 32> Anchors;
  // The region's MFMA instructions, in program order.
  SmallVector<Instruction *, 32> MFMAInsts;
};

// Stateless helpers shared by region analysis and scheduling.
namespace Utils {
bool isMFMAorWMMA(const Instruction &I) {
  // Shared matrix-core predicate (also used by the scalarize-packed-fops
  // pass). gfx950 only exposes MFMA, but the helper is family-agnostic.
  return mlir::triton::AMD::isMFMAorWMMA(I);
}

bool isHoistTransparentInst(const Instruction &I) {
  return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
}

bool isSinkTransparentInst(const Instruction &I) {
  return isa<ExtractElementInst>(I);
}

// A register-class pin: an empty inline asm whose output is tied to its input
// ("=a,0" / "=v,0"), which Triton's `cd_regclass` MFMA lowering puts around
// each MFMA tile's accumulator. It emits no instruction; it only forces the
// value into AGPRs or VGPRs at that point, so it has to stay right next to its
// MFMA: the pin on C directly before the MFMA, the pin on D directly after.
bool isRegClassPin(const Instruction &I) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->arg_size() != 1 ||
      CI->getType() != CI->getArgOperand(0)->getType())
    return false;
  const auto *IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
  if (!IA || !IA->getAsmString().empty())
    return false;
  StringRef Constraints = IA->getConstraintString();
  return Constraints == "=a,0" || Constraints == "=v,0";
}

// The pin that produces MFMA's accumulator operand C, if that is its only use.
Instruction *getCPin(Instruction *MFMA) {
  auto *CI = cast<CallInst>(MFMA);
  if (CI->arg_size() < 3 || CI->getArgOperand(2)->getType() != CI->getType())
    return nullptr;
  auto *Pin = dyn_cast<Instruction>(CI->getArgOperand(2));
  if (!Pin || !isRegClassPin(*Pin) || !Pin->hasOneUse() ||
      Pin->getParent() != MFMA->getParent())
    return nullptr;
  return Pin;
}

// The pin on MFMA's result D, if it is the result's only user.
Instruction *getDPin(Instruction *MFMA) {
  if (!MFMA->hasOneUse())
    return nullptr;
  auto *Pin = dyn_cast<Instruction>(*MFMA->user_begin());
  if (!Pin || !isRegClassPin(*Pin) || Pin->getParent() != MFMA->getParent())
    return nullptr;
  return Pin;
}

SchedKind classifySchedInst(Instruction &I) {
  if (isMFMAorWMMA(I))
    return SchedKind::MFMA;

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (Function *F = CI->getCalledFunction()) {
      if (F->isIntrinsic()) {
        StringRef Name = F->getName();
        // GR: buffer.load (into regs), buffer.load.lds / .async.lds,
        //     raw.ptr.buffer.store (gmem store from regs)
        if (Name.contains("buffer.load") ||
            Name.contains("raw.ptr.buffer.store"))
          return SchedKind::GR;
        // LR: ds_read (ds.read.*) or ds_load (ds.load.*)
        if (Name.contains("ds.read") || Name.contains("ds.load"))
          return SchedKind::LR;
      }
    }
  }

  // LR: load from LDS (addrspace 3)
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LR;
  }

  // LW: store to LDS (addrspace 3)
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LW;
  }

  return SchedKind::Other;
}

iterator_range<BasicBlock::iterator> instructionsInRegion(const BBRegion &R) {
  BasicBlock *BB = R.BB;
  // Begin is now inclusive (region starts at this instruction)
  auto ItBegin = R.Begin ? R.Begin->getIterator() : BB->begin();
  auto ItEnd = R.End ? R.End->getIterator() : BB->end();
  return make_range(ItBegin, ItEnd);
}

unsigned getMFMACycles(const Instruction &I) {
  if (!isMFMAorWMMA(I))
    return 0;
  const auto *CI = cast<CallInst>(&I);
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return 0;
  StringRef Name = Callee->getName();

  // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
  // cbsz (arg 3) and blgp (arg 4).
  if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4")) {
    // both operands f4 -> 16 cycles, otherwise (either operand f8) -> 32.
    if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 16
                                                                        : 32;
    return 32; // Fallback if cbsz/blgp are not constants
  }
  if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4")) {
    // both operands f4 -> 32 cycles, otherwise (either operand f8) -> 64.
    if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 32
                                                                        : 64;
    return 64; // Fallback if cbsz/blgp are not constants
  }

  // Fixed-cost MFMAs.
  static constexpr struct {
    StringRef Name;
    unsigned Cycles;
  } kFixedCycles[] = {
      {"mfma.f32.16x16x32.f16", 16},  {"mfma.f32.16x16x32.bf16", 16},
      {"mfma.i32.16x16x64.i8", 16},   {"mfma.f32.32x32x16.f16", 32},
      {"mfma.f32.32x32x16.bf16", 32}, {"mfma.i32.32x32x32.i8", 32},
  };
  for (const auto &Entry : kFixedCycles)
    if (Name.contains(Entry.Name))
      return Entry.Cycles;

  // Unknown / unmodeled shape: the scheduler bails on this region and leaves
  // it to the default LLVM schedulers (such kernels are not perf-critical).
  return 0;
}

// Width in bits of the value moved by an LDS-access anchor.
unsigned getLDSAccessBits(const Instruction *I) {
  if (const auto *LI = dyn_cast<LoadInst>(I))
    return LI->getType()->getPrimitiveSizeInBits();
  if (const auto *SI = dyn_cast<StoreInst>(I))
    return SI->getValueOperand()->getType()->getPrimitiveSizeInBits();
  if (const auto *CI = dyn_cast<CallInst>(I))
    return CI->getType()->getPrimitiveSizeInBits();
  return 0;
}

// LDS instruction throughput during steady state, which is proportional to
// the access bits.
unsigned getLDSCoverCycles(const Instruction *I, unsigned MFMACycles) {
  unsigned Bits = getLDSAccessBits(I);
  return Bits ? (Bits / 8) : MFMACycles;
}

// MFMAs to emit at this LDS access under a throughput model: reads and writes
// share the one LDS issue port, so we carry a running cycle balance across
// the region's accesses and emit floor(balance / MFMACycles) MFMAs here,
// keeping the remainder for the next access.
unsigned takeMFMAsForLDS(const Instruction *I, unsigned MFMACycles,
                         unsigned &AccumCycles) {
  AccumCycles += getLDSCoverCycles(I, MFMACycles);
  unsigned N = AccumCycles / MFMACycles; // floor; carry the remainder
  AccumCycles -= N * MFMACycles;
  return N;
}
} // namespace Utils

// Region analysis and scheduling logic grouped into a helper class
class LLIRScheduler {
public:
  explicit LLIRScheduler() = default;

  // Roll a block back to a pre-scheduling snapshot: erase the instructions the
  // scheduler inserted (the region-comment inline-asm calls and the
  // llvm.amdgcn.sched.barrier intrinsics, all void with no uses) and restore
  // the recorded instruction order.
  static void restoreBlock(BasicBlock &BB,
                           const SmallVectorImpl<Instruction *> &snapshot) {
    SmallPtrSet<const Instruction *, 32> orig(snapshot.begin(), snapshot.end());
    SmallVector<Instruction *, 8> inserted;
    for (Instruction &I : BB)
      if (!orig.count(&I))
        inserted.push_back(&I);
    for (Instruction *I : inserted) {
      if (!I->use_empty())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
    for (size_t i = 1; i < snapshot.size(); ++i)
      snapshot[i]->moveAfter(snapshot[i - 1]);
  }

  // Schedule every block in the function. Region detection + the per-region
  // structural invariant make this safe: a block with no eligible MFMA region
  // is simply left untouched (so no loop-finding heuristic is needed, and an
  // odd prologue / multiple loops / no loop are all handled uniformly).
  // Each block is scheduled transactionally: if its schedule fails
  // verification (e.g. an epilogue the main logic can't safely interleave),
  // only that block is rolled back, so good blocks keep their schedule.
  // Returns true if any region was scheduled.
  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "LLIR scheduler analyzing function: " << F.getName()
                      << "\n");
    BBMFMAAnalysisMap BBMFMAMap;
    bool scheduled = false;
    for (BasicBlock &BB : F) {
      LLVM_DEBUG(dbgs() << "BB: " << BB.getName() << "\n");
      analyzeBB(BB, BBMFMAMap);

      // Snapshot the block so we can revert just this block on failure.
      SmallVector<Instruction *, 64> snapshot;
      for (Instruction &I : BB)
        snapshot.push_back(&I);

      if (!scheduleBB(BB, BBMFMAMap))
        continue;

      if (verifyFunction(F, nullptr)) {
        // This block's schedule is invalid; bail gracefully on it alone.
        LLVM_DEBUG(dbgs() << "  reverting unschedulable block " << BB.getName()
                          << "\n");
        restoreBlock(BB, snapshot);
      } else {
        scheduled = true;
      }
    }
    return scheduled;
  }

private:
  // Split a basic block into MFMA regions in a single program-order pass,
  // recording each region's first MFMA (its RegionStart) and its MFMA count.
  // A new region opens at every MFMA that follows a memory op (GR/LR/LW) seen
  // since the region began; by construction an MFMA's input loads land in an
  // earlier region, so intra-region reordering is dependency-safe.
  static void analyzeBB(BasicBlock &BB, BBMFMAAnalysisMap &Out) {
    MFMARegionList Regions;
    unsigned CurRegion = 0;
    bool SeenMemoryOps = false;
    bool InRegion = false;

    for (Instruction &I : BB) {
      SchedKind SK = Utils::classifySchedInst(I);
      if (SK == SchedKind::GR || SK == SchedKind::LR || SK == SchedKind::LW)
        SeenMemoryOps = true;

      if (!Utils::isMFMAorWMMA(I))
        continue;

      // This MFMA opens a new region when we are not already in one, or when a
      // memory op has appeared since the current region's MFMAs began (that op
      // feeds this MFMA, so it belongs to the next region). Otherwise the MFMA
      // just extends the current region.
      bool startNewRegion = !InRegion || SeenMemoryOps;
      if (startNewRegion) {
        if (InRegion)
          CurRegion++; // close the region we were in, open the next
        InRegion = true;
        // Memory ops seen *before* a region's first MFMA are that region's own
        // setup (their data feeds these MFMAs), not a boundary; clear the flag
        // so they don't spuriously split off the next MFMA.
        SeenMemoryOps = false;
        if (CurRegion >= Regions.size())
          Regions.resize(CurRegion + 1);
        Regions[CurRegion].RegionStart = &I; // first MFMA is the region start
      }
      Regions[CurRegion].TotalMFMA++;
    }

    if (Regions.empty())
      return;

    LLVM_DEBUG({
      for (unsigned i = 0; i < Regions.size(); ++i)
        dbgs() << "Region " << i << ": total MFMA: " << Regions[i].TotalMFMA
               << "\n";
    });

    Out[&BB] = std::move(Regions);
  }

  static bool feedsMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (Utils::isMFMAorWMMA(*UI))
            return true;
          if (Utils::isHoistTransparentInst(*UI) || Utils::isRegClassPin(*UI))
            Worklist.push_back(UI);
        }
      }
    }
    return false;
  }

  static bool definedByMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (auto *DefI = dyn_cast<Instruction>(V)) {
        if (Utils::isMFMAorWMMA(*DefI))
          return true;

        if (Utils::isSinkTransparentInst(*DefI) || Utils::isRegClassPin(*DefI)) {
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
        }
      }
    }
    return false;
  }

  static MFMARegionCollectResult
  collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
    MFMARegionCollectResult Res;

    // All instructions in this region. Hoisting moves a prep to the region
    // start (right after R.Begin). That is only safe if the prep's operands are
    // already available there; if an operand is defined later inside the
    // region, hoisting the prep above it would use a value before its
    // definition (an SSA dominance violation / invalid IR). This set lets us
    // detect that case and leave such preps in place.
    SmallPtrSet<const Instruction *, 32> RegionInsts;
    for (Instruction &I : Utils::instructionsInRegion(R))
      RegionInsts.insert(&I);

    // Preps cleared for hoisting so far (collected in program order). An
    // operand that is a same-region prep we already hoisted stays ahead of its
    // use.
    SmallPtrSet<const Instruction *, 16> Hoisted;

    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW) {
        Res.LastAnchor = &I;
        Res.Anchors.push_back({&I, K});
        continue;
      }

      if (K == SchedKind::MFMA) {
        Res.MFMAInsts.push_back(&I);
        continue;
      }

      if (Utils::isHoistTransparentInst(I)) {
        // Hoisting moves I to right after the region start, so it is safe only
        // if every operand still dominates that position: operands defined
        // before the region already do, R.Begin does, and a prep we are also
        // hoisting keeps its relative order ahead of I. An operand defined
        // inside the region that we are NOT hoisting — an LR/LW/GR anchor, or a
        // prep we rejected — would end up after its use, so I must stay put.
        // This covers both shuffle and insertelement, all anchor kinds, and
        // multi-hop chains.
        bool safeToHoist = true;
        for (Value *Op : I.operands()) {
          auto *OpI = dyn_cast<Instruction>(Op);
          if (!OpI || OpI == R.Begin)
            continue;
          if (RegionInsts.count(OpI) && !Hoisted.count(OpI)) {
            safeToHoist = false;
            break;
          }
        }
        if (safeToHoist && feedsMFMA(&I)) {
          Res.Hoist.push_back(&I);
          Hoisted.insert(&I);
        }
        continue;
      }

      if (isa<ExtractElementInst>(I)) {
        if (definedByMFMA(&I))
          Res.Sink.push_back(&I);
      }
    }

    return Res;
  }

  static MFMARegionCollectResult
  preprocessMFMAInstsInRegion(const BBRegion &R) {
    auto Res = collectMFMAAndTransparentInstsInRegion(R);

    if (Res.Hoist.empty() && Res.Sink.empty())
      return Res;

    Instruction *HoistPos = R.Begin; // Region start (the region's first MFMA)
    // Keep the region start's D pin directly after it.
    if (Instruction *DPin = Utils::getDPin(R.Begin))
      HoistPos = DPin;
    Instruction *SinkPos = Res.LastAnchor; // last anchor in region

    if (HoistPos)
      for (Instruction *I : llvm::reverse(Res.Hoist)) {
        // Don't hoist R.Begin after itself
        if (I != HoistPos)
          I->moveAfter(HoistPos);
      }

    // Sinking needs a trailing anchor to move past. A region with MFMAs but no
    // GR/LR/LW anchor (LastAnchor stays null) has no valid sink point, so leave
    // the extractelements in place rather than dereferencing a null insert
    // position. Such a region carries no anchors, so scheduleMFMAWithSpacing
    // bails on it anyway.
    // Sink ALL MFMA-result extractelements to just past the region's last
    // anchor. This is required, not just opportunistic: it clears them out of
    // the MFMA run so the subsequent interleaving (which only reorders
    // instructions before LastAnchor) cannot move an MFMA past one of its own
    // result extracts. A region with MFMAs but no anchor has no sink point, so
    // the SinkPos null-guard leaves those extracts in place (scheduleMFMAWith-
    // Spacing bails on such a region anyway). Any genuinely unsafe sink is
    // caught by the per-block verifyFunction rollback.
    if (SinkPos)
      for (Instruction *I : llvm::reverse(Res.Sink))
        I->moveAfter(SinkPos);

    return Res;
  }

  static StringRef schedKindName(SchedKind K) {
    switch (K) {
    case SchedKind::GR:
      return "GR";
    case SchedKind::LR:
      return "LR";
    case SchedKind::LW:
      return "LW";
    case SchedKind::MFMA:
      return "mfma";
    case SchedKind::Other:
      return "other";
    }
    llvm_unreachable("unknown SchedKind");
  }

  // Helper: move N MFMAs after InsertPt using moveAfter.
  // moveAfter naturally produces correct order: each new MFMA goes right
  // after InsertPt, pushing previous ones further away.
  // Result: InsertPt, MFMA[N-K], ..., MFMA[N-2], MFMA[N-1]
  static unsigned moveMFMAsAfter(SmallVectorImpl<Instruction *> &MFMAInsts,
                                 unsigned &MFMAIdx, unsigned Count,
                                 Instruction *InsertPt) {
    unsigned moved = 0;
    for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j) {
      moveMFMAAfter(MFMAInsts[--MFMAIdx], InsertPt);
      moved++;
    }
    return moved;
  }

  // Move one MFMA right after InsertPt, together with its register-class pins:
  // the C pin stays directly before it and the D pin directly after it.
  static void moveMFMAAfter(Instruction *MFMA, Instruction *InsertPt) {
    Instruction *CPin = Utils::getCPin(MFMA);
    Instruction *DPin = Utils::getDPin(MFMA);
    MFMA->moveAfter(InsertPt);
    if (CPin)
      CPin->moveBefore(MFMA->getIterator());
    if (DPin)
      DPin->moveAfter(MFMA);
  }

  // Interleave MFMA with anchor instructions using moveAfter.
  //
  // Pure throughput model — every count is "how many MFMAs of compute cover
  // this memory op's issue-port occupancy needs", never a latency-hiding
  // reorder:
  //   - GR (global load): ceil(64 / mfma_cycles) MFMAs (1 if immediately
  //                       followed by an LR).
  //   - LR / LW (LDS read/write): floor(carried LDS-cycle balance /
  //   mfma_cycles)
  //                       MFMAs. Reads and writes share the one LDS issue port,
  //                       so both use the same width-proportional pairing.
  //   - 2 MFMAs drain the tail; any leftover compute (more MFMAs than the
  //   memory
  //     ops demand cover for) is split evenly between the region's head and
  //     tail, with an odd MFMA favoring the head.
  static void
  scheduleMFMAWithSpacing(SmallVectorImpl<AnchorInst> &Anchors,
                          SmallVectorImpl<Instruction *> &MFMAInsts) {
    if (Anchors.empty() || MFMAInsts.empty())
      return;

    unsigned mfmaCycles = Utils::getMFMACycles(*MFMAInsts.front());
    if (mfmaCycles == 0)
      return;
    // A global load occupies the global-load path for ~64 cycles, so it needs
    // 64 cycles of MFMA cover — ceil(64 / mfma_cycles) MFMAs (4 for a 16-cycle
    // MFMA, 2 for 32-cycle). Same throughput basis as the LDS-access pairing.
    unsigned mfmaPerGR = llvm::divideCeil(64, mfmaCycles);

    unsigned MFMAIdx = MFMAInsts.size();
    unsigned Total = MFMAIdx;

    // Count anchors by kind. ldsBudget is the total MFMA cover the region's LDS
    // accesses demand, floor(sum(cycles_per_access) / mfma_cycles) — the
    // throughput ratio over reads *and* writes alike, so cheap accesses share
    // an MFMA and wide ones draw several.
    unsigned numGR = 0, numGRBeforeLR = 0, totalLDSCycles = 0;
    for (size_t j = 0; j < Anchors.size(); ++j) {
      if (Anchors[j].Kind == SchedKind::GR) {
        numGR++;
        if (j + 1 < Anchors.size() && Anchors[j + 1].Kind == SchedKind::LR)
          numGRBeforeLR++;
      } else if (Anchors[j].Kind == SchedKind::LR ||
                 Anchors[j].Kind == SchedKind::LW) {
        totalLDSCycles += Utils::getLDSCoverCycles(Anchors[j].I, mfmaCycles);
      }
    }
    unsigned ldsBudget = totalLDSCycles / mfmaCycles;

    // gfx950 scheduling:
    //   GR: mfmaPerGR MFMAs each (except GR→LR gets 1)
    //   LR/LW: cycle-paired at the true throughput ratio (takeMFMAsForLDS) —
    //          cheap accesses share an MFMA, wide ones draw several
    //   2 MFMAs drain the end; leftover compute is split evenly head/tail
    unsigned grBudget = mfmaPerGR * (numGR - numGRBeforeLR);
    unsigned needed = grBudget + numGRBeforeLR + ldsBudget + 2;
    unsigned leftover = (Total > needed) ? Total - needed : 0;

    // Surplus compute (more MFMAs than the memory ops need cover for) is split
    // evenly between the region's head and tail; an odd MFMA favors the head.
    unsigned tailLeftover = leftover / 2;            // floor → tail
    unsigned headLeftover = leftover - tailLeftover; // ceil → head

    LLVM_DEBUG(dbgs() << "  MFMA budget: total=" << Total
                      << ", needed=" << needed << ", leftover=" << leftover
                      << " (head=" << headLeftover << ", tail=" << tailLeftover
                      << ")\n");

    // The 2-MFMA tail drain plus the tail's share of the surplus. The head's
    // share is whatever stays unmoved at the front after the reverse walk.
    unsigned MFMAAtEnd =
        moveMFMAsAfter(MFMAInsts, MFMAIdx, 2 + tailLeftover, Anchors.back().I);
    // Running LDS-cycle balance carried across the region's LDS accesses (LR
    // and LW, processed in reverse) so the MFMA:access pairing follows the true
    // throughput ratio.
    unsigned ldsAccum = 0;
    DenseMap<SchedKind, unsigned> MFMAPerAnchorKind;

    for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
         --i) {
      size_t idx = static_cast<size_t>(i);
      Instruction *InsertPt = Anchors[idx].I;
      SchedKind Kind = Anchors[idx].Kind;

      unsigned Count = 0;
      if (Kind == SchedKind::LR || Kind == SchedKind::LW) {
        // Cycle model: emit floor(balance / mfma_cycles) MFMAs, carrying the
        // remainder — cheap accesses share an MFMA, wide ones draw several.
        // Reads and writes use the same shared LDS-cycle balance.
        Count = Utils::takeMFMAsForLDS(Anchors[idx].I, mfmaCycles, ldsAccum);
      } else if (Kind == SchedKind::GR) {
        bool followedByLR = (idx + 1 < Anchors.size() &&
                             Anchors[idx + 1].Kind == SchedKind::LR);
        Count = followedByLR ? 1 : mfmaPerGR;
      }

      [[maybe_unused]] unsigned moved =
          moveMFMAsAfter(MFMAInsts, MFMAIdx, Count, InsertPt);
      LLVM_DEBUG(MFMAPerAnchorKind[Kind] += moved);
    }

    LLVM_DEBUG({
      dbgs() << "  MFMA insertion summary: total=" << Total
             << ", at_front=" << MFMAIdx << ", at_end=" << MFMAAtEnd;
      for (auto &KV : MFMAPerAnchorKind) {
        dbgs() << ", after_" << schedKindName(KV.first) << "=" << KV.second;
      }
      dbgs() << "\n";
    });
  }

  // Insert an inline asm comment before the given instruction.
  // Emit a non-side-effecting inline-asm comment (a pure annotation, NOT a
  // reorder barrier). The sched.barriers we emit at anchors -- not these region
  // markers -- pin the schedule, so the markers must stay side-effect-free to
  // avoid adding scheduling constraints of their own.
  static void insertAsmComment(Instruction *IP, const std::string &Comment) {
    LLVMContext &Ctx = IP->getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(IP);
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    InlineAsm *IA =
        InlineAsm::get(FTy, ";; " + Comment, "", /*hasSideEffects=*/false);
    Builder.CreateCall(IA);
  }

  // Insert llvm.amdgcn.sched.barrier(Mask) immediately after AfterI so the pre-
  // and post-RA machine schedulers cannot move instructions across this anchor.
  // Mask bits name the instruction classes allowed to cross; 0 is a full
  // barrier.
  static void insertSchedBarrier(Instruction *AfterI, uint32_t Mask) {
    Instruction *Next = AfterI->getNextNode();
    if (!Next)
      return; // anchors are never terminators, but be defensive
    IRBuilder<> Builder(AfterI->getContext());
    Builder.SetInsertPoint(Next);
    Builder.CreateIntrinsic(Intrinsic::amdgcn_sched_barrier,
                            {Builder.getInt32(Mask)});
  }

  static bool scheduleBB(BasicBlock &BB, const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return false;

    const MFMARegionList &Regions = It->second;

    unsigned NumRegions = Regions.size();
    unsigned ScheduledRegionIdx = 0;

    for (unsigned i = 0; i < NumRegions; ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.RegionStart)
        continue;

      if (R.TotalMFMA != 0) {
        BBRegion bbR;
        bbR.BB = &BB;
        bbR.Begin = Regions[i].RegionStart;
        bbR.End = (i + 1 < NumRegions) ? Regions[i + 1].RegionStart : nullptr;

        // Schedulability check, performed BEFORE any mutation: bail on a region
        // whose MFMA shape we don't model (getMFMACycles == 0) or that has no
        // memory anchor to interleave the MFMAs against. Skipping here --
        // before preprocessMFMAInstsInRegion moves anything -- ensures the pass
        // only schedules (and sched.barrier-pins) regions it actually models,
        // and only reports success (which gates the AGPR-form flags) for those;
        // unmodeled regions are left to the default LLVM schedulers.
        unsigned MFMACycles = 0;
        bool SeenMFMA = false, HasAnchor = false;
        for (Instruction &I : Utils::instructionsInRegion(bbR)) {
          SchedKind K = Utils::classifySchedInst(I);
          if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
            HasAnchor = true;
          else if (K == SchedKind::MFMA && !SeenMFMA) {
            SeenMFMA = true;
            MFMACycles = Utils::getMFMACycles(I);
          }
        }
        if (MFMACycles == 0 || !HasAnchor)
          continue;

        MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);

        // --- Build region comment ---
        std::string Comment;
        raw_string_ostream OS(Comment);

        // Count anchors by kind
        unsigned numGR = 0, numLR = 0, numLW = 0;
        for (auto &A : Res.Anchors) {
          if (A.Kind == SchedKind::GR)
            numGR++;
          else if (A.Kind == SchedKind::LR)
            numLR++;
          else if (A.Kind == SchedKind::LW)
            numLW++;
        }
        OS << "Region " << ScheduledRegionIdx << ": " << Res.MFMAInsts.size()
           << " mfma, " << numGR << " GR, " << numLR << " LR, " << numLW
           << " LW";
        ScheduledRegionIdx++;

        // Before the region start's C pin (if any), so the pin stays adjacent.
        Instruction *CPin = Utils::getCPin(bbR.Begin);
        insertAsmComment(CPin ? CPin : bbR.Begin, Comment);

        LLVM_DEBUG({
          dbgs() << "Cluster " << i << " structure:";
          SchedKind RunKind = SchedKind::Other;
          unsigned RunCount = 0;
          for (Instruction &Inst : Utils::instructionsInRegion(bbR)) {
            SchedKind K = Utils::classifySchedInst(Inst);
            if (K != SchedKind::MFMA && K != SchedKind::GR &&
                K != SchedKind::LR && K != SchedKind::LW)
              continue;
            if (K == RunKind) {
              RunCount++;
            } else {
              if (RunCount > 0)
                dbgs() << " " << RunCount << " " << schedKindName(RunKind);
              RunKind = K;
              RunCount = 1;
            }
          }
          if (RunCount > 0)
            dbgs() << " " << RunCount << " " << schedKindName(RunKind);
          dbgs() << "\n";
        });

        scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts);

        // Pin this region's schedule with a full sched.barrier (mask 0) after
        // each memory anchor, so LLVM's pre- and post-RA machine schedulers
        // preserve the MFMA<->mem interleave. This keeps misched enabled for
        // the rest of the function -- prologue/epilogue and any region the pass
        // bailed on still get machine-scheduled -- so no global misched-disable
        // is needed.
        // Each anchor's fence goes in front of it, so every window between
        // fences is `anchor, mfma...` with the memory op already leading. With
        // register-class pins this keeps a load between two pinned tiles
        // (see the plugin README, "Register-class pins").
        for (const AnchorInst &A : Res.Anchors)
          if (Instruction *Prev = A.I->getPrevNode())
            insertSchedBarrier(Prev, /*Mask=*/0);
        // With register-class pins, also fence each MFMA tile after its D pin so
        // LLVM's machine scheduler cannot reorder the pinned MFMAs in the
        // stretches between memory anchors; that reordering leaves pins behind
        // as v_accvgpr copies.
        for (Instruction *MFMA : Res.MFMAInsts)
          if (Instruction *DPin = Utils::getDPin(MFMA))
            insertSchedBarrier(DPin, /*Mask=*/0);
      }
    }
    return ScheduledRegionIdx > 0;
  }
};


// ===========================================================================
// warp_pipeline (Flash-Attention) region analysis.
//
// Unlike the GEMM path (regions inferred from "MFMA after a memory op"), a
// warp-pipeline kernel already carries its cluster boundaries: ConvertWarpPipeline
// lowers each stage boundary to llvm.amdgcn.sched.barrier(i32 0). So here a
// region is simply the instruction span between two sched.barriers, and the
// compute (MFMA) stages are the regions that contain MFMAs.
//
// This first cut only DETECTS regions, checks MFMA<->VALU independence, and
// histograms mfma / valu — no reordering yet. Enable prints with
// LLIRSCHED_WP_DEBUG=1.
// ===========================================================================
// ===========================================================================
// Which scheduling model a region wants
// ===========================================================================
// This plugin carries TWO independent models, and the choice is a property of what
// a region CONTAINS, not of which kernel it came from:
//
//   mfma + memory (ds_read/ds_write/buffer_load), no valu
//       -> THROUGHPUT model: pair each memory op with as many mfmas as its
//          bandwidth needs (`Utils::takeMFMAsForLDS`, the `LLIRScheduler` path).
//          Typical of intra-wave GEMM, where the loop hides LDS latency.
//
//   mfma + valu, no memory
//       -> CO-EXEC model: fill each mfma's 24-cycle shadow with co-issuable VALU
//          and declare it with sched_group_barrier (this namespace). Typical of
//          inter-wave Flash-Attention, whose DOT clusters are register-only --
//          `TRITON_WP_DEBUG` confirms the FA dot stages have zero LDS effects.
//
//   mfma + valu + memory
//       -> NOT HANDLED. Intra-wave FA would land here. The two models disagree
//          about what an mfma is for (covering VALU issue vs covering memory
//          latency), so such a region is skipped rather than fed to a model whose
//          assumptions it breaks. Revisit when that kernel exists.
//
//   no mfma
//       -> nothing to schedule around. FA's mem stages are this case (they carry
//          LDS + global traffic but no mfma), which is why the co-exec model never
//          sees them; `LLIRSCHED_WP_MEMNOP` paces them separately.
//
// Inter-wave GEMM needs no scheduling at all and simply produces no qualifying
// region.
namespace WP {

// The sched.barrier(i32 0) intrinsic marking a warp-pipeline cluster boundary.
static bool isSchedBarrier(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      return F->getName().contains("amdgcn.sched.barrier");
  return false;
}

// A function is a warp-pipeline kernel iff it already contains sched.barriers
// (the GEMM path has none before this plugin runs). This only decides *where the
// region boundaries come from* -- which model each region then gets is decided by
// classifyRegion() below, per region.
static bool isWarpPipelineFunc(Function &F) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isSchedBarrier(I))
        return true;
  return false;
}

enum class RegionModel { None, Throughput, CoExec, Mixed };

// Defined below, once the cost model it belongs to is in scope.
static int valuWeight(const Instruction &I);

// What does this region contain? Counts are returned so the caller can log them.
static RegionModel classifyRegion(Instruction *Begin, BasicBlock::iterator End,
                                  int &numMfma, int &numValu, int &numMem) {
  numMfma = numValu = numMem = 0;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    Instruction &I = *It;
    if (Utils::isMFMAorWMMA(I)) {
      ++numMfma;
      continue;
    }
    // Reuse the throughput model's own classifier so the two paths cannot disagree
    // about what counts as memory: GR = buffer/global, LR/LW = LDS read/write.
    SchedKind K = Utils::classifySchedInst(I);
    if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW) {
      ++numMem;
      continue;
    }
    if (valuWeight(I) > 0)
      ++numValu;
  }
  if (numMfma == 0)
    return RegionModel::None;
  if (numMem > 0)
    return numValu > 0 ? RegionModel::Mixed : RegionModel::Throughput;
  return numValu > 0 ? RegionModel::CoExec : RegionModel::None;
}

static const char *modelName(RegionModel M) {
  switch (M) {
  case RegionModel::None:
    return "no-mfma";
  case RegionModel::Throughput:
    return "mfma+mem -> THROUGHPUT model";
  case RegionModel::CoExec:
    return "mfma+valu -> CO-EXEC model";
  case RegionModel::Mixed:
    return "mfma+valu+mem -> MIXED (not handled)";
  }
  return "?";
}

// Transcendental (v_exp/v_rcp/...): never co-issues with MFMA -- kept separate
// from plain VALU (matches the hardware isNeverCoissue rule).
static bool isTransCompute(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        return N.contains("exp2") || N.contains("exp.") || N.contains("log2") ||
               N.contains("sin") || N.contains("cos") || N.contains("sqrt") ||
               N.contains("rcp");
      }
  return false;
}


// Transitive: does I reach any instruction in `targets` through its operands,
// WITHIN the same iteration? We stop at PHI nodes so loop-carried (backedge)
// values are not followed -- a VALU consuming the *previous* iteration's MFMA
// (the warp-pipeline decoupling) is independent for interleaving purposes; only
// an intra-iteration MFMA->VALU path counts as a real dependency.
static bool dependsOnAny(Instruction *I,
                         const SmallPtrSetImpl<const Instruction *> &targets) {
  SmallVector<const Value *, 32> Work(I->op_begin(), I->op_end());
  SmallPtrSet<const Value *, 32> Seen;
  while (!Work.empty()) {
    const Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    if (const auto *DI = dyn_cast<Instruction>(V)) {
      if (targets.count(DI))
        return true;
      if (isa<PHINode>(DI))
        continue; // don't cross loop-carried / cross-iteration edges
      Work.append(DI->op_begin(), DI->op_end());
    }
  }
  return false;
}

// Co-issue weight of a VALU op for the interleave count (0 if not counted):
//   regular unpacked scalar FP = 1;  packed (vector FP) = 2;
//   transcendental (exp/...) = 2;  v_permlane = 2.
// True if I is an fmaximum/fminimum/maxnum/minnum intrinsic call -- the ops the
// AMD backend folds pairwise (max(max(a,b),c)) into v_maximum3/v_minimum3.
static bool isMaxMin(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        return N.contains("maximum") || N.contains("minimum") ||
               N.contains("maxnum") || N.contains("minnum");
      }
  return false;
}

// Mark the "inner" max/mins that the backend absorbs into a v_maximum3, so they
// count 0 (they vanish at isel). Replicates isel's greedy bottom-up fold: walk
// the region's max/mins in program order; each one absorbs at most one single-
// use, not-yet-committed max/min operand as its inner. Absorbed -> Folded (0);
// the absorber becomes a v_maximum3 (full weight). Matches the measured count
// (LLIR llvm.maximum -> ASM v_maximum3) exactly on the FA softmax reduction.
static void computeMax3Folds(Instruction *Begin, BasicBlock::iterator End,
                             const SmallPtrSetImpl<const Instruction *> &Region,
                             SmallPtrSetImpl<const Instruction *> &Folded) {
  SmallPtrSet<const Instruction *, 32> Committed;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    if (!isMaxMin(*It))
      continue;
    for (Value *Op : It->operands()) {
      auto *In = dyn_cast<Instruction>(Op);
      if (In && isMaxMin(*In) && Region.count(In) && In->hasOneUse() &&
          !Folded.count(In) && !Committed.count(In)) {
        Folded.insert(In);
        Committed.insert(&*It);
        break;
      }
    }
  }
}

static int valuWeight(const Instruction &I) {
  if (Utils::isMFMAorWMMA(I))
    return 0;
  if (isTransCompute(I))
    return 2;
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        StringRef N = F->getName();
        if (N.contains("permlane"))
          return 2;
        // Count maximum/minimum (the softmax row-max reduction) by DEFAULT.
        // The 2:1 v_maximum3 fold is handled at collection time by
        // computeMax3Folds (inner ops -> weight 0), so with fold-aware weighting
        // the reduction contributes its real issued count (validated: 89 vs the
        // naive 164).
        //
        // Counting them is worth ~1.5%: with weight 0 the reduction is INVISIBLE to
        // the interleave, so its ~16 v_maximum3 pile up *before* the stage's first
        // mfma and no window ever covers them (measured on FAv4: 76 cycles of
        // co-exec-capable VALU stranded at the PV stage head while that stage's own
        // windows sat 92 cycles under-filled; counting moves head 76 -> 0 cyc and
        // fill 292 -> 368 / 384).
        if (N.contains("maxnum") || N.contains("minnum") ||
            N.contains("maximum") || N.contains("minimum") ||
            N.contains("fmuladd") || N.contains("fma."))
          return I.getType()->isVectorTy() ? 2 : 1;
        // llvm.fabs is NOT counted, for the same reason as fneg below: it becomes a
        // source modifier, not an instruction. (No current kernel has one, so this
        // only guards the future.)
      }
  // NOTE: an fmul feeding a single fadd/fsub contracts into one v_pk_fma (-> two
  // v_fma), so in principle the pair is 2 issue slots not 4. But zeroing the
  // fmul weight here creates weight-0 runs that make the error-diffusion cluster
  // mfmas (measured a 5-long mfma run, 1035 vs 1043). Left double-counted on
  // purpose -- the denser qk*scale placement it produces is empirically better.
  // A packed convert (fptrunc/fpext of a <2 x T>) lowers to ONE co-issuable
  // v_cvt_pk_f16 (4 cyc) and is NOT scalarized like packed fmul/fadd, so it is a
  // single co-issue unit -> weight 1, not 2. (Counting it 2 under-filled the cvt
  // windows: 4 cvt = weight 8 "full" but only 16 cyc of the 24-cyc window, so the
  // interleave placed 4/window instead of the 6 that fit.)
  // fneg is NOT an instruction on AMDGPU: it folds into its consumer as a source
  // modifier (`v_fma_f32 v0, v0, s44, -v129`). Counting it inflates a declared
  // sched_group_barrier group by an instruction that never exists, and IGroupLP
  // cannot fill that group -- so its pipeline solver gives up and leaves ISel's
  // order for the WHOLE region. Measured on FAv4 with SCALE_ON_Q=0, whose QK stage
  // carries one fneg for fma(qk, qk_scale, -m_new): 7 of 16 groups were placed and
  // the remaining 8 mfma were emitted back-to-back, stranding 12 exp2 plus ~20 valu
  // with no co-exec window (the stage's twin, which had no fneg, scheduled fine).
  if (I.getOpcode() == Instruction::FNeg)
    return 0;
  if (isa<FPTruncInst>(I) || isa<FPExtInst>(I))
    return 1;
  if (I.getType()->isFPOrFPVectorTy() &&
      (isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<SelectInst>(I)))
    return I.getType()->isVectorTy() ? 2 : 1; // packed vector = 2, scalar = 1
  return 0;
}

// Transparent glue that builds an input of one of THIS region's mfmas
// (insertelement/shuffle feeding a region mfma). Region-restricted so we don't
// hoist a prep that actually feeds the *next* region's mfma (which may legally
// use this region's valu).
static bool feedsMFMA(Instruction *I,
                      const SmallPtrSetImpl<const Instruction *> &RegionMfmas) {
  SmallVector<Value *, 8> Work;
  SmallPtrSet<Value *, 16> Seen;
  Work.push_back(I);
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users())
      if (auto *UI = dyn_cast<Instruction>(U)) {
        if (RegionMfmas.count(UI))
          return true;
        if (Utils::isHoistTransparentInst(*UI))
          Work.push_back(UI);
      }
  }
  return false;
}

// extractelement of one of THIS region's mfma results.
static bool definedByMFMA(Instruction *I,
                          const SmallPtrSetImpl<const Instruction *> &RegionMfmas) {
  SmallVector<Value *, 8> Work;
  SmallPtrSet<Value *, 16> Seen;
  Work.push_back(I);
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    if (auto *DI = dyn_cast<Instruction>(V)) {
      if (RegionMfmas.count(DI))
        return true;
      if (Utils::isSinkTransparentInst(*DI))
        for (Value *Op : DI->operands())
          Work.push_back(Op);
    }
  }
  return false;
}

// Pin an instruction's position by emitting llvm.amdgcn.sched.barrier(0)
// immediately after it, so the machine scheduler cannot reorder across it.
static void insertSchedBarrierAfter(Instruction *I) {
  Instruction *Next = I->getNextNode();
  if (!Next)
    return;
  IRBuilder<> B(Next);
  B.CreateIntrinsic(Intrinsic::amdgcn_sched_barrier, {B.getInt32(0)});
}

// IGroupLP scheduling-group masks (AMDGPUIGroupLP / SCHED_GROUP_BARRIER).
// TRANS is a class of its own: canAddMI()'s VALU branch excludes transcendentals,
// so v_exp matches ONLY the TRANS mask.
static constexpr uint32_t kSGBMaskVALU = 0x002;
static constexpr uint32_t kSGBMaskMFMA = 0x008;
static constexpr uint32_t kSGBMaskTRANS = 0x400;
// One mfma opens a co-execution window shorter than its matrix occupancy: a
// 32x32x16 (32-cycle) mfma exposes 24 cycles. Do not hardcode that -- an FA variant
// built on 16x16x32 mfma has a 16-cycle occupancy, and giving it 24-cycle windows
// would over-fill every one of them by 3x.
static constexpr int kWindowCycles = 24;   // the validated 32-cycle-mfma case
static constexpr int kMFMANonOverlap = 8;  // occupancy - window, from that case

// Co-exec window for a region, derived from the mfmas it actually contains.
// Returns 0 when the shape is unmodelled or mixed in a way we should not guess at,
// in which case the caller leaves the region to the default schedulers -- the same
// bail-out `Utils::getMFMACycles` already uses.
static int regionWindowCycles(Instruction *Begin, BasicBlock::iterator End) {
  unsigned MinCycles = 0;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    if (!Utils::isMFMAorWMMA(*It))
      continue;
    unsigned C = Utils::getMFMACycles(*It);
    if (C == 0)
      return 0; // unmodelled mfma: do not invent a window for it
    // Mixed shapes: size the window for the SHORTEST mfma, so no window overfills.
    MinCycles = MinCycles ? std::min(MinCycles, C) : C;
  }
  if (MinCycles == 0)
    return 0;
  return std::max<int>(4, (int)MinCycles - kMFMANonOverlap);
}
// Mem-stage head pacing: 2 x `s_nop 7` measured best on both FA kernels.
static constexpr int kDefaultMemNops = 2;
// v_permlane is a cross-lane shuffle: model it as a fat 20-cycle VALU.
static constexpr int kPermlaneCycles = 20;
// One 4-cycle VALU op per co-exec slot: 24 cycles / 4 = 6 slots per mfma.
static constexpr int kSlotCycles = 4;

// A packed f32 op this pass can split into per-element scalar ops. Deliberately
// narrow: fmul / fadd / fsub / fma(muladd) on a <N x float>.
//
//  * fptrunc/fpext are excluded -- a packed convert IS one v_cvt_pk_f16_f32, and
//    splitting it would double the issue count for no gain.
//  * <N x half> is excluded -- v_pk_*_f16 is the natural form of f16 math, not a
//    fusion of two scalar ops.
static bool isSplittablePackedFP(const Instruction &I) {
  auto *VT = dyn_cast<FixedVectorType>(I.getType());
  if (!VT || VT->getNumElements() < 2 || !VT->getElementType()->isFloatTy())
    return false;
  if (isa<BinaryOperator>(I))
    return I.getOpcode() == Instruction::FMul || I.getOpcode() == Instruction::FAdd ||
           I.getOpcode() == Instruction::FSub;
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic()) {
        Intrinsic::ID Id = F->getIntrinsicID();
        return Id == Intrinsic::fmuladd || Id == Intrinsic::fma;
      }
  return false;
}

// Replace a packed op with one scalar op per element, appending the new scalar ops
// to Out. Same rewrite as Triton's ScalarizePackedFOps, applied to ONE op instead
// of every packed op in the block -- which is the whole point here: only the ops
// that landed in an mfma co-exec window want to be scalar.
static bool scalarizePackedFP(Instruction *I, SmallVectorImpl<Instruction *> &Out) {
  auto *VT = dyn_cast<FixedVectorType>(I->getType());
  if (!VT)
    return false;
  unsigned N = VT->getNumElements();
  IRBuilder<> B(I);
  Value *Vec = UndefValue::get(VT);
  auto *BO = dyn_cast<BinaryOperator>(I);
  auto *CI = dyn_cast<CallInst>(I);
  for (unsigned e = 0; e < N; ++e) {
    Value *R = nullptr;
    if (BO) {
      Value *A = B.CreateExtractElement(BO->getOperand(0), e);
      Value *C = B.CreateExtractElement(BO->getOperand(1), e);
      R = B.CreateBinOp(BO->getOpcode(), A, C);
    } else if (CI) {
      Value *A = B.CreateExtractElement(CI->getArgOperand(0), e);
      Value *C = B.CreateExtractElement(CI->getArgOperand(1), e);
      Value *D = B.CreateExtractElement(CI->getArgOperand(2), e);
      R = B.CreateIntrinsic(VT->getElementType(),
                            CI->getCalledFunction()->getIntrinsicID(), {A, C, D});
    } else {
      return false;
    }
    if (auto *RI = dyn_cast<Instruction>(R)) {
      RI->copyFastMathFlags(I); // keep contraction/reassociation rights identical
      Out.push_back(RI);
    }
    Vec = B.CreateInsertElement(Vec, R, e);
  }
  I->replaceAllUsesWith(Vec);
  I->eraseFromParent();
  return true;
}

// ---------------------------------------------------------------------------
// Over-capacity stages: choose what to hide, and pack what cannot be hidden.
// ---------------------------------------------------------------------------
// declareRegionGroups() below assumes the stage's VALU work FITS in the mfma
// co-exec capacity (24 cyc x M) and spreads it so no group overflows. FAv3 breaks
// that assumption: its QK stage carries ~470 cycles of VALU against 384 cycles of
// capacity and its PV stage ~490 (measured minGroups 20 and 21 against 16 mfmas,
// so the balanced packer's merge loop collapses pairs into 48-cycle groups).
//
// When the work does not fit, no schedule hides it and the question changes to
// which ops get a window and what shape the rest take:
//
//   1. WHICH ops get covered. An op that cannot be packed (exp2, the v_maximum3
//      reduction, v_cvt_pk, permlane) gains nothing from being left alone, so it
//      gets first claim on the windows. Whatever capacity is left goes to the
//      packable ops, taken from the END of the stage backwards -- mfmas inserted in
//      reverse program order -- which keeps the uncovered remainder contiguous and
//      leaves it where it already sits: at the head in FAv3's QK (the rescale
//      muls), in the middle in its PV (the qk_scale fmas, between the max3
//      reduction and the exp2s).
//   2. WHAT SHAPE the rest takes. A covered op should be SCALAR, so it co-issues
//      one 4-cycle slot at a time inside its window. An UNCOVERED op should stay
//      PACKED: nothing hides it either way, and one v_pk_mul_f32 retires two
//      elements in one issue where two v_mul_f32 need two.
//
// This is why FAv3 must NOT set AMDGCN_SCALARIZE_PACKED_FOPS: that pass splits
// every packed op in any block containing an mfma, including the uncovered ones
// this pass deliberately keeps packed.
//
// Weights follow the slot model: 1 mfma = 6 slots, 1 unpacked op = 1 slot, 1
// packed op = 2 slots; exp2 = 2 slots (8 cyc) and permlane = 5 (20 cyc).
static bool declareRegionGroupsOverCap(Instruction *Begin, Instruction *End,
                                       int syncID, int M, int windowCycles,
                                       const SmallPtrSetImpl<const Instruction *> &Max3Folded,
                                       const SmallPtrSetImpl<const Instruction *> &transSet) {
  BasicBlock *BB = Begin->getParent();
  auto ItEnd = End ? End->getIterator() : BB->end();

  struct Op {
    Instruction *I;
    int cyc;
    bool isTrans;
    bool splittable;
    bool covered;
  };
  SmallVector<Op, 64> ops;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (Utils::isMFMAorWMMA(I) || Max3Folded.count(&I))
      continue;
    int w = valuWeight(I);
    if (w <= 0)
      continue;
    bool isTrans = transSet.count(&I) != 0;
    bool isPermlane = false;
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (Function *F = CI->getCalledFunction())
        isPermlane = F->getName().contains("permlane");
    int cyc = isPermlane ? kPermlaneCycles : w * kSlotCycles;
    ops.push_back({&I, cyc, isTrans, isSplittablePackedFP(I), false});
  }
  if (ops.empty() || M < 2)
    return false;

  const int Cap = windowCycles * M;
  int Total = 0, Fixed = 0;
  for (const Op &o : ops) {
    Total += o.cyc;
    if (!o.splittable)
      Fixed += o.cyc;
  }
  if (Total <= Cap)
    return false; // fits -- the balanced packer handles it better

  // A window is one mfma's 24-cycle shadow. It may hold SEVERAL declared groups,
  // of different IGroupLP classes: [MFMA 1][VALU 1][TRANS 1] asks for one mfma and
  // then a sub and an exp2 behind it, which is one mfma instead of two. FAv3's PV
  // stage ends in exactly that pair (a lone sub, then a lone exp2) and used to
  // spend two windows on 12 cycles of work.
  struct Chunk {
    bool isTrans;
    int cyc, n;
  };
  struct Slot {
    bool hasMfma; // false = uncovered: declared with no mfma in front of it
    int cyc;
    SmallVector<Chunk, 4> chunks;
  };

  // Decide coverage for a given packable-op budget, then lay the result out into
  // slots. Returns the number of windows the layout needs.
  auto decideAndLayout = [&](int avail, SmallVectorImpl<Slot> &out, int &boughtOut,
                             int &nSplitOut) {
    for (Op &o : ops)
      o.covered = !o.splittable; // non-splittable ops get first claim
    int bought = 0, nSplit = 0;
    for (int i = (int)ops.size() - 1; i >= 0; --i) {
      Op &o = ops[i];
      if (o.splittable && bought + o.cyc <= avail) {
        o.covered = true;
        bought += o.cyc;
        ++nSplit;
      }
    }
    boughtOut = bought;
    nSplitOut = nSplit;

    out.clear();
    int nWin = 0;
    for (const Op &o : ops) {
      // Instruction count as IGroupLP counts it: a covered packed op is about to be
      // scalarized into one op per element; an uncovered one stays a single issue.
      int n = 1;
      if (o.covered && o.splittable)
        if (auto *VT = dyn_cast<FixedVectorType>(o.I->getType()))
          n = VT->getNumElements();
      bool wantMfma = o.covered;
      Slot *S = out.empty() ? nullptr : &out.back();
      // Extend the current slot when it is the same kind of slot and the op still
      // fits the window. Class may differ from the previous chunk -- that is the
      // point -- so only the cycle budget and the covered/uncovered split bound it.
      bool fits = S && S->hasMfma == wantMfma &&
                  (!wantMfma || S->cyc + o.cyc <= windowCycles);
      if (!fits) {
        out.push_back({wantMfma, 0, {}});
        S = &out.back();
        if (wantMfma)
          ++nWin;
      }
      S->cyc += o.cyc;
      if (!S->chunks.empty() && S->chunks.back().isTrans == o.isTrans) {
        S->chunks.back().cyc += o.cyc;
        S->chunks.back().n += n;
      } else {
        S->chunks.push_back({o.isTrans, o.cyc, n});
      }
    }
    return nWin;
  };

  // Iterate: a window freed by tighter packing is capacity the packable ops can
  // still use, so feed it back into the coverage budget. FAv3's PV frees the window
  // its trailing sub+exp2 pair used to waste, which buys three more v_pk_fma.
  int Avail = Cap - Fixed, Bought = 0, nSplit = 0;
  SmallVector<Slot, 40> slots;
  int nWin = decideAndLayout(Avail, slots, Bought, nSplit);
  for (int iter = 0; iter < 4 && nWin < M; ++iter) {
    int grown = Avail + (M - nWin) * windowCycles;
    SmallVector<Slot, 40> trial;
    int tb = 0, ts = 0;
    int tw = decideAndLayout(grown, trial, tb, ts);
    if (tw > M)
      break; // overshot: keep the layout that still fits
    Avail = grown;
    slots = std::move(trial);
    Bought = tb;
    nSplit = ts;
    if (tw == nWin)
      break; // nothing more to win
    nWin = tw;
  }
  // Re-run the decision so ops[].covered matches the layout we kept.
  nWin = decideAndLayout(Avail, slots, Bought, nSplit);

  // More windows than mfmas? Merge the cheapest adjacent pair, so the excess lands
  // in one deliberately over-full window instead of dropping the tail slots -- which
  // would cost the LAST exp2 groups their windows, the ops least able to afford it.
  while (nWin > M) {
    size_t bi = slots.size();
    int best = INT_MAX;
    for (size_t i = 0; i + 1 < slots.size(); ++i)
      if (slots[i].hasMfma && slots[i + 1].hasMfma) {
        int cost = slots[i].cyc + slots[i + 1].cyc;
        if (cost < best) {
          best = cost;
          bi = i;
        }
      }
    if (bi == slots.size())
      break; // every window is separated by an uncovered slot; nothing to merge
    slots[bi].cyc += slots[bi + 1].cyc;
    for (const Chunk &c : slots[bi + 1].chunks)
      if (!slots[bi].chunks.empty() && slots[bi].chunks.back().isTrans == c.isTrans) {
        slots[bi].chunks.back().cyc += c.cyc;
        slots[bi].chunks.back().n += c.n;
      } else {
        slots[bi].chunks.push_back(c);
      }
    slots.erase(slots.begin() + bi + 1);
    --nWin;
  }

  // Scalarize exactly the covered packed ops. The layout above already accounted
  // for the instruction count each one becomes, so nothing needs re-walking after
  // this -- which matters because the rewrite erases the original op.
  int PackedLeft = 0;
  SmallVector<Instruction *, 16> ToSplit;
  for (Op &o : ops) {
    if (o.covered && o.splittable)
      ToSplit.push_back(o.I);
    else if (o.splittable)
      ++PackedLeft;
  }
  for (Instruction *I : ToSplit) {
    SmallVector<Instruction *, 4> New;
    scalarizePackedFP(I, New);
  }

  // Emit. Program order is always satisfiable (it is the order IGroupLP already
  // sees), so unlike the fitting path this needs no dependency-ordered blocks:
  // coverage, not reordering, is the decision here.
  int gBare = M - nWin;
  if (gBare < 0)
    gBare = 0;
  Instruction *IP = End ? End : BB->getTerminator();
  if (!IP)
    return false;
  IRBuilder<> B(IP);
  auto emit = [&](uint32_t mask, int size) {
    B.CreateIntrinsic(Intrinsic::amdgcn_sched_group_barrier,
                      {B.getInt32(mask), B.getInt32(size), B.getInt32(syncID)});
  };
  for (size_t i = 0; i < slots.size(); ++i) {
    const Slot &S = slots[i];
    // Partnerless mfmas go just before the LAST slot, not at the region head: an
    // unfilled window is free, but having it first delays every co-issued op.
    if (i + 1 == slots.size())
      for (int k = 0; k < gBare; ++k)
        emit(kSGBMaskMFMA, 1);
    if (S.hasMfma)
      emit(kSGBMaskMFMA, 1);
    for (const Chunk &c : S.chunks)
      emit(c.isTrans ? kSGBMaskTRANS : kSGBMaskVALU, c.n);
  }

  if (std::getenv("LLIRSCHED_WP_DEBUG")) {
    errs() << "  [sgb-overcap] sync=" << syncID << " M=" << M << " cap=" << Cap
           << " total=" << Total << "cyc fixed=" << Fixed << "cyc avail=" << Avail
           << "cyc bought=" << Bought << "cyc  scalarized=" << nSplit
           << " packed-left=" << PackedLeft << "  windows=" << nWin
           << "/" << M << "  slots:";
    for (const Slot &S : slots) {
      errs() << " " << (S.hasMfma ? "[" : "*[");
      bool first = true;
      for (const Chunk &c : S.chunks) {
        errs() << (first ? "" : "+") << (c.isTrans ? "T" : "V") << c.n;
        first = false;
      }
      errs() << "]" << S.cyc;
    }
    errs() << "  (* = no mfma) bare=" << gBare << "\n";
  }
  return true;
}

// LLIRSCHED_WP_SGB: declare the co-exec schedule with sched_group_barrier instead
// of physically reordering the region and pinning it with sched_barrier(0).
//
// Why: sched_barrier(0) is only an advisory "do not cross" marker for the machine
// scheduler. Measured on FAv4, codegen still consolidates the last two
// sub-regions of a stage (an mfma migrates toward the region front, so ~5 v_cvt_pk
// or ~3 v_exp end up past the final mfma with no window over them) even though the
// plugin's own IR had every group inside its 24-cycle window. sched_group_barrier
// is the stronger form: AMDGPUIGroupLP *builds* the requested pipeline in the
// machine scheduler rather than merely forbidding motion. This is the mechanism
// ROCm/FlyDSL uses -- it emits no reordering at all, just
// {[MFMA 1][VALU 5..6]} and {[MFMA 1][TRANS 3]} group declarations per cluster on
// stock upstream LLVM.
//
// Group sizing follows the validated split from the FAv3 co-issue work: treat one
// TRANS as two VALU slots (8 vs 4 cycles), so with M mfma, V valu slots and E
// trans ops -> K1 = ceil((V + 2E)/M) valu per mfma, K2 = ceil(K1/2) trans per
// mfma, g0 = round(V/K1) mfmas take VALU groups and the remaining g1 take TRANS
// groups. VALU groups are declared FIRST (valu-first measured 1071 vs exp-first
// 1047 TFLOPS on FAv3).
//
// Placement matters: IGroupLP forms groups scanning UPWARD from the barrier, so
// the whole declaration must sit AFTER every real op of the region -- emitting it
// at the top yields empty groups and silently does nothing.
static bool declareRegionGroups(Instruction *Begin, Instruction *End,
                                int syncID) {
  BasicBlock *BB = Begin->getParent();
  auto ItEnd = End ? End->getIterator() : BB->end();
  while (Begin && isa<PHINode>(Begin))
    Begin = Begin->getNextNode();
  if (!Begin || Begin == End)
    return false;

  SmallPtrSet<const Instruction *, 32> RegionInsts, Max3Folded;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It)
    RegionInsts.insert(&*It);
  computeMax3Folds(Begin, ItEnd, RegionInsts, Max3Folded);

  // Collect the region's co-issuable ops as maximal RUNS of a single IGroupLP
  // class, in program order.
  //
  // Why runs and not one VALU block + one TRANS block: the declaration must be
  // satisfiable, and satisfiability is a dataflow property. FAv4's DOT1 stage after
  // opt7 is `VALU x8 -> TRANS x8 -> VALU x50` (T3's subs, then its exp2s, then the
  // sum-reduction adds and the p->fp16 converts, which CONSUME those exps). A
  // two-block "all VALU then all TRANS" declaration asks IGroupLP to schedule 58
  // VALU before the first TRANS, which the dependency forbids -- the solver then
  // abandons the pipeline and leaves ISel's order, collapsing 8 exps plus their
  // dependent adds into one 132-cycle group and leaving 6 mfmas bare (measured).
  // Emitting one group sequence per run, in order, is always satisfiable because it
  // *is* the program order, and it handles any number of class transitions.
  //
  // Cost model per op (cycles / instruction count as IGroupLP counts them):
  //   * plain valu   4 cyc, 1 instruction
  //   * packed valu  8 cyc, 1 instruction -- two slots' worth of window, but ONE
  //                  entry: sched_group_barrier sizes are instruction counts, and
  //                  SIPreEmitPeephole splits whatever lands in a shadow later
  //   * v_permlane   20 cyc, 1 instruction (cross-lane shuffle; one window can hide
  //                  a permlane plus a single 4-cycle op and no more)
  //   * TRANS (exp2) 8 cyc, 1 instruction, its own mask
  // Blocks are ordered by DEPENDENCY, not by raw program order.
  //
  // Raw program order is too fine: FAv4's PV stage emits sub(T0) exp(T0) sub(T1)
  // exp(T1) ... = 8 alternating runs, and one mandatory group sequence per run does
  // not fit 16 windows (measured: budget widened to 32, 156 cyc of overflow). But
  // those runs are freely reorderable -- sub(T1) does not depend on exp(T0) -- so
  // they belong in ONE VALU block.
  //
  // What is *not* reorderable is a VALU op that consumes a TRANS result. FAv4's DOT1
  // stage has exactly that: T3's subs (independent) -> its exp2s -> the
  // sum-reduction adds and p->fp16 converts, which CONSUME those exps. So classify:
  //
  //   block 0: VALU that does NOT depend on any TRANS in this region
  //   block 1: TRANS
  //   block 2: VALU that DOES depend on a TRANS in this region
  //
  // That is always satisfiable (it is a topological order of the class dependency),
  // collapses to the old two-block form when block 2 is empty, and handles any
  // number of program-order transitions.
  SmallPtrSet<const Instruction *, 32> transSet;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It)
    if (!Utils::isMFMAorWMMA(*It) && !Max3Folded.count(&*It) &&
        isTransCompute(*It))
      transSet.insert(&*It);

  struct ClassRun {
    bool isTrans;
    SmallVector<int, 32> cyc;
  };
  SmallVector<ClassRun, 3> runs;
  runs.push_back({false, {}}); // VALU, TRANS-independent
  runs.push_back({true, {}});  // TRANS
  runs.push_back({false, {}}); // VALU, TRANS-dependent
  int M = 0;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (Utils::isMFMAorWMMA(I)) {
      ++M;
      continue;
    }
    if (Max3Folded.count(&I))
      continue; // folds into a v_maximum3, no issue slot of its own
    if (isTransCompute(I)) {
      runs[1].cyc.push_back(8);
      continue;
    }
    if (int w = valuWeight(I)) {
      bool isPermlane = false;
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *F = CI->getCalledFunction())
          isPermlane = F->getName().contains("permlane");
      int idx = dependsOnAny(&I, transSet) ? 2 : 0;
      // ONE entry per instruction, priced by its weight -- the same convention the
      // TRANS branch above uses (one entry of 8) and the same one
      // declareRegionGroupsOverCap uses for its Chunks. It matters because the entry
      // count becomes the sched_group_barrier group SIZE, and that size is a count of
      // INSTRUCTIONS. Pushing one 4-cycle entry per ELEMENT instead would declare 6
      // where a window holds 3 packed ops; IGroupLP cannot fill the group and its
      // solver then abandons the whole region's pipeline. With instruction counts the
      // declaration is satisfiable as emitted, and SIPreEmitPeephole splits whatever
      // ends up in a shadow -- so no kernel needs Triton's ScalarizePackedFOps.
      if (isPermlane)
        runs[idx].cyc.push_back(kPermlaneCycles);
      else
        runs[idx].cyc.push_back(4 * w);
    }
  }
  // Drop empty blocks so they cost no groups.
  {
    SmallVector<ClassRun, 3> kept;
    for (ClassRun &r : runs)
      if (!r.cyc.empty())
        kept.push_back(std::move(r));
    runs = std::move(kept);
  }
  if (M < 2 || runs.empty())
    return false;

  // Size the co-exec window from the mfmas this region actually contains, rather
  // than assuming the 32x32x16 shape the model was validated on.
  const int Window = regionWindowCycles(Begin, ItEnd);
  if (Window <= 0)
    return false; // unmodelled mfma shape: leave it to the default schedulers
  if (Window != kWindowCycles && std::getenv("LLIRSCHED_WP_DEBUG"))
    errs() << "  [sgb] sync=" << syncID << " window=" << Window
           << "cyc (derived; the validated 32-cycle-mfma case is " << kWindowCycles
           << ")\n";

  // Over capacity? Then no arrangement hides the work and the balanced packer
  // below is solving the wrong problem -- it would spread, find it needs more
  // groups than there are mfmas, and merge cheap pairs into double-width groups.
  // Hand those stages to the packed-aware path, which decides what to cover and
  // keeps the uncovered remainder packed. Opt out with LLIRSCHED_WP_NOOVERCAP.
  {
    int totalCyc = 0;
    for (const ClassRun &r : runs)
      for (int c : r.cyc)
        totalCyc += c;
    if (totalCyc > Window * M &&
        std::getenv("LLIRSCHED_WP_NOOVERCAP") == nullptr &&
        declareRegionGroupsOverCap(Begin, End, syncID, M, Window, Max3Folded, transSet))
      return true;
  }

  // Pack one block into groups. BALANCED, not greedy first-fit.
  //
  // First-fit fills each group to the brim and leaves the block's tail in a stub,
  // which wastes windows: a 20-cycle permlane after a full group opens a group that
  // can then take only one more 4-cycle op. Worse, when the resulting group count
  // exceeded the mfma count the old code widened `budget` for the WHOLE stage, so
  // every group was allowed to run 28 cycles and overflow -- 40 cycles of exposed
  // VALU in a stage whose 364 cycles of work fit inside 384 cycles of capacity.
  //
  // Instead: take the minimum number of groups the cycle total needs,
  // g = ceil(total / budget), then aim for total/g per group. That keeps every group
  // at or under `budget` while spreading the slack evenly, so no group overflows and
  // the tail stub disappears.
  auto packGroups = [](ArrayRef<int> cycles, int budget) {
    SmallVector<int, 32> sizes;
    if (cycles.empty())
      return sizes;
    int total = 0;
    for (int c : cycles)
      total += c;
    // The minimum number of windows this block's cycle total needs. Hitting exactly
    // this count matters: one group too many trips the widen-the-window fallback,
    // which then lets EVERY group in the stage overflow. Measured on FAv4's QK
    // stage: 360 cyc of work, 384 cyc of capacity, 16 windows available and 16
    // needed -- yet fragmentation around the 20-cycle permlane produced 17 groups,
    // the budget widened to 28, and 16 cycles of VALU ended up exposed in a stage
    // that fits perfectly.
    int g = (total + budget - 1) / budget;
    if (g < 1)
      g = 1;
    // Adaptive target: aim each group at the average of what is LEFT over the groups
    // still to come, capped by the window. This self-corrects after a wide op (a
    // permlane forces a short group; the following groups take a little more) and
    // lands on exactly `g` groups instead of fragmenting.
    int remCyc = total, remG = g, cur = 0, n = 0;
    for (int c : cycles) {
      int target = (remG > 0) ? (remCyc + remG - 1) / remG : budget;
      target = std::min(target, budget);
      if (n > 0 && (cur + c > budget || (cur >= target && remG > 1))) {
        sizes.push_back(n);
        remCyc -= cur;
        if (remG > 1)
          --remG;
        cur = 0;
        n = 0;
      }
      cur += c;
      ++n;
    }
    if (n)
      sizes.push_back(n);
    return sizes;
  };
  SmallVector<SmallVector<int, 32>, 8> groups;
  int budget = Window, total = 0;
  groups.clear();
  for (const ClassRun &r : runs) {
    groups.push_back(packGroups(r.cyc, budget));
    total += (int)groups.back().size();
  }
  // Too many groups for the available mfmas? Do NOT widen the window for the whole
  // stage -- that lets every group overflow (measured: 24 cyc exposed in a stage
  // whose 360 cyc of work fits in 384 cyc of capacity). Instead MERGE the cheapest
  // adjacent pair, repeatedly, so the excess is confined to one or two groups.
  //
  // Fragmentation is why the count can exceed the cycle-minimum at all: a 20-cycle
  // permlane cannot share a 24-cycle window with more than one 4-cycle op, so the
  // group before it closes short. That costs 8 cycles of capacity, not 24.
  while (total > M) {
    size_t bi = 0, gi = 0;
    int best = INT_MAX;
    for (size_t b = 0; b < groups.size(); ++b)
      for (size_t g = 0; g + 1 < groups[b].size(); ++g) {
        int cost = groups[b][g] + groups[b][g + 1];
        if (cost < best) {
          best = cost;
          bi = b;
          gi = g;
        }
      }
    if (best == INT_MAX)
      break; // nothing left to merge (every block is a single group)
    groups[bi][gi] += groups[bi][gi + 1];
    groups[bi].erase(groups[bi].begin() + gi + 1);
    --total;
  }
  int gBare = M - total;
  if (gBare < 0)
    gBare = 0;

  Instruction *IP = End ? End : BB->getTerminator();
  if (!IP)
    return false;
  IRBuilder<> B(IP);
  auto emit = [&](uint32_t mask, int size) {
    B.CreateIntrinsic(Intrinsic::amdgcn_sched_group_barrier,
                      {B.getInt32(mask), B.getInt32(size), B.getInt32(syncID)});
  };
  for (size_t i = 0; i < runs.size(); ++i) {
    // Partnerless mfmas go just before the LAST run, not at the region head: an
    // unfilled window is free, but having it first delays every co-issued op
    // behind it.
    if (i + 1 == runs.size())
      for (int k = 0; k < gBare; ++k)
        emit(kSGBMaskMFMA, 1);
    for (int n : groups[i]) {
      emit(kSGBMaskMFMA, 1);
      emit(runs[i].isTrans ? kSGBMaskTRANS : kSGBMaskVALU, n);
    }
  }
  if (std::getenv("LLIRSCHED_WP_DEBUG")) {
    errs() << "  [sgb] sync=" << syncID << " M=" << M << " budget=" << budget;
    {
      int tc = 0, need = 0;
      for (const ClassRun &r : runs) {
        int c = 0;
        for (int x : r.cyc)
          c += x;
        tc += c;
        need += (c + Window - 1) / Window;
        errs() << " [" << (r.isTrans ? "T" : "V") << " " << c << "cyc/"
               << ((c + Window - 1) / Window) << "g]";
      }
      errs() << " total=" << tc << "cyc cap=" << 24 * M << " minGroups=" << need;
    }
    errs() << " runs:";
    for (size_t i = 0; i < runs.size(); ++i) {
      errs() << " " << (runs[i].isTrans ? "TRANS" : "VALU") << "("
             << runs[i].cyc.size() << " ops){";
      for (int n : groups[i])
        errs() << n << " ";
      errs() << "}";
    }
    errs() << " bare=" << gBare << "\n";
  }
  return true;
}

// Interleave MFMA with VALU in one independent region [Begin, End).
//   X = sum of valu weights, Y = #mfma.
//   Reserve the last 6 (weighted) valu for the last mfma; distribute the rest
//   evenly, ceil((X-6)/(Y-1)) per mfma. Traverse valu in reverse and place each
//   mfma before the start of its group (mfmas kept in program order so their
//   accumulator chain stays valid).
static void interleaveRegion(Instruction *Begin, Instruction *End,
                             bool pin = true) {
  BasicBlock *BB = Begin->getParent();
  // PHI nodes must stay grouped at the block top. When this region is the first
  // span of its block (Begin is a phi -- e.g. FA's QK stage in the loop's merge
  // block), advance past all leading phis so no repositioned mfma or hoisted prep
  // is ever placed above/between them (that is invalid IR -> the whole wp pass
  // rolls back and the region is left un-interleaved).
  while (Begin && isa<PHINode>(Begin))
    Begin = Begin->getNextNode();
  if (!Begin || Begin == End)
    return;
  auto ItEnd = End ? End->getIterator() : BB->end();

  // 1. Cleanup: hoist THIS region's mfma-input preps to Begin, sink its
  //    mfma-result extracts to just before End, so mfmas can move freely among
  //    the valu. (llirSched's hoist/sink idea, adapted for wp regions.)
  SmallVector<Instruction *, 32> Mfmas, Valus;
  SmallVector<int, 32> Weights;
  SmallPtrSet<const Instruction *, 32> RegionMfmas, RegionInsts;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    RegionInsts.insert(&*It);
    if (Utils::isMFMAorWMMA(*It))
      RegionMfmas.insert(&*It);
  }
  SmallVector<Instruction *, 16> Hoist, Sink;
  SmallPtrSet<const Instruction *, 16> Hoisted;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (Utils::isMFMAorWMMA(I)) {
      Mfmas.push_back(&I);
    } else if (Utils::isHoistTransparentInst(I) && feedsMFMA(&I, RegionMfmas)) {
      // safe to hoist only if every operand already dominates Begin (defined
      // before the region, or a prep we already hoisted).
      bool Safe = true;
      for (Value *Op : I.operands())
        if (auto *OpI = dyn_cast<Instruction>(Op))
          if (OpI != Begin && RegionInsts.count(OpI) && !Hoisted.count(OpI)) {
            Safe = false;
            break;
          }
      if (Safe) {
        Hoist.push_back(&I);
        Hoisted.insert(&I);
      }
    } else if (isa<ExtractElementInst>(I) && definedByMFMA(&I, RegionMfmas)) {
      Sink.push_back(&I);
    }
  }
  for (Instruction *I : llvm::reverse(Hoist))
    if (I != Begin)
      I->moveAfter(Begin);
  if (End)
    for (Instruction *I : llvm::reverse(Sink))
      I->moveBefore(End->getIterator());

  // 2. Collect valu (weight>0) in current program order, with weights. The max/min
  //    reduction is counted (see valuWeight), and by DEFAULT it is counted
  //    fold-aware: the inner max/mins that isel folds pairwise into a v_maximum3
  //    count 0, so the reduction contributes its REAL issued instruction count
  //    rather than 2x. Without the fold the interleave over-reserves window space
  //    for ops that vanish at isel. LLIRSCHED_WP_NOMAX3FOLD restores naive counting.
  SmallPtrSet<const Instruction *, 32> Max3Folded;
  computeMax3Folds(Begin, ItEnd, RegionInsts, Max3Folded);
  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    if (Max3Folded.count(&*It))
      continue;
    int W = valuWeight(*It);
    if (W > 0) {
      Valus.push_back(&*It);
      Weights.push_back(W);
    }
  }
  int Y = Mfmas.size();
  int N = Valus.size();
  if (Y < 2 || N == 0)
    return;
  int X = 0;
  for (int W : Weights)
    X += W;
  if (X <= 6)
    return;
  // 3. Place each mfma's group-start valu by scanning the valu in reverse.
  //    DEFAULT = REVERSE-6: don't spread evenly. Scan valu in reverse and place
  //    a mfma after each 6-weight group -- up to 7 when a weight-2 packed op
  //    straddles the 6 boundary (close at acc>=6; max overshoot 5+2=7). Groups
  //    are [MFMA][6-7 weight]. When X > 6*Y the leftover valu (X - ~6*Y) sit
  //    BEFORE the first mfma -> a VALU prologue at the stage front. With the
  //    LOCAL-barrier fix the in-stage lgkmcnt stalls are gone, so the di/dt
  //    reason for even-spreading no longer applies; front-loading VALU + tight
  //    mfma groups wins (FAv3: 1073 -> 1076 TFLOPS, 78.1% MFMA-eff/SIMD).
  //
  //    GEMM-safe: interleaveRegion only ever runs on the FAv3 warp-pipeline
  //    steady-state regions -- scheduleRegions calls it solely for INDEPENDENT
  //    mfma/valu spans, and GEMM has none (its epilogue valu all depend on the
  //    region's mfmas -> "DEPENDENT -> skip"), so this path never touches GEMM.

  SmallVector<Instruction *, 32> GroupStart; // reverse: last group first
  int vi = N - 1;
  {
    // Group weight per mfma. Default 6 (REVERSE-6, tuned for VALU-heavy stages
    // like FA's PV: X=105 for 16 mfma -- surplus valu front-load as a prologue).
    // When the region is VALU-light (X < 6*Y, e.g. FA's QK stage: X=51 for 16
    // mfma) a fixed 6 strands (Y - X/6) mfmas in a front cluster, so shrink the
    // group weight to ~X/Y (>=1) and spread ALL Y mfmas -- interleave the whole
    // region regardless of how little valu it carries (mfma/valu already verified
    // independent by the caller).
    // Group weight per mfma. Weight 6 == the full 24-cyc co-exec window (a plain
    // valu is weight 1 == 4 cyc; exp/permlane weight 2 == 8 cyc). For a VALU-light
    // region (X < 6*Y) use the smaller X/Y so all Y mfmas get company instead of
    // (Y - X/6) of them clustering bare at the region front.
    //
    // Window-sized groups (G = 6 everywhere) measured WORSE (1162 vs 1174 TFLOPS on
    // FAv4 @16320): bigger groups leave fewer, larger runs, and the backend's
    // relocation of pure valu then piles a heavier tail into the last sub-region --
    // asm window overflow went 16/20 -> 32/32 cyc for QK/PV.
    int G = 6;
    if (X < 6 * Y) {
      G = X / Y;
      if (G < 1)
        G = 1;
    }
    int acc = 0;
    for (; vi >= 0 && (int)GroupStart.size() < Y; --vi) {
      acc += Weights[vi];
      if (acc >= G) {
        GroupStart.push_back(Valus[vi]);
        acc = 0;
      }
    }
    // The reverse walk only closes a group once it reaches G weight, so when the
    // valu run out mid-group the front-most (unclosed) run is left with no mfma
    // ahead of it -- exactly the stranded head above. If any mfma is still
    // unassigned, spend one on that run: `vi + 1` is its first valu (vi == -1 when
    // the walk consumed every valu, giving Valus[0]).
    if (acc > 0 && (int)GroupStart.size() < Y)
      GroupStart.push_back(Valus[vi + 1]);
  }

  // 4. Place mfmas. mfmas[Y-1] before GroupStart[0] (last group), mfmas[Y-2]
  //    before GroupStart[1], ...  Any mfmas left after the valu run out go at
  //    the region front, in program order.
  int P = GroupStart.size();
  if (std::getenv("LLIRSCHED_WP_DEBUG"))
    errs() << "  [interleave] Y(mfma)=" << Y << " N(valu)=" << N
           << " X(weighted)=" << X << " ideal=(X-6)/(Y-1)=" << (double)(X - 6) / (Y - 1)
           << " groups_formed=" << P << " front_cluster=" << (Y - P) << "\n";
  int mi = Y - 1;
  for (int j = 0; j < P; ++j, --mi)
    Mfmas[mi]->moveBefore(GroupStart[j]->getIterator());
  // remaining Mfmas[0..mi] -> before the earliest placed mfma (or first valu),
  // preserving order.
  Instruction *At = (P > 0) ? Mfmas[mi + 1] : Valus[0];
  for (; mi >= 0; --mi) {
    Mfmas[mi]->moveBefore(At->getIterator());
    At = Mfmas[mi];
  }

  // 5. Pin the schedule: emit sched.barrier(0) after each mfma so the machine
  //    scheduler cannot re-cluster the mfmas (which it does otherwise, undoing
  //    the interleave). One barrier per mfma locks each [mfma][valu-group] pair.
  //    Env-toggle (LLIRSCHED_WP_PIN=0 disables) to A/B the pinning.
  if (pin)
    for (Instruction *M : Mfmas)
      insertSchedBarrierAfter(M);
}

// LLIRSCHED_WP_MEMNOP=k: emit k x `s_nop 7` (8 idle scalar cycles each) at the
// START of every MEM region -- a region between two cluster barriers that carries
// memory ops but no mfma.
//
// Borrowed from ROCm/FlyDSL's gfx950 FA kernel, which opens every one of its mem
// clusters with `_s_nop(7)` immediately before that cluster's sched_barrier(0).
// The point is not to waste time: in an inter-wave ping-pong pipeline the two
// waves alternate dot/mem clusters, and a fixed delay at the head of the mem
// cluster shifts this wave's ds_read burst slightly later, so it does not collide
// with the other wave's LDS traffic / issue slots. It is a phase-tuning knob for
// the two-wave interleave, so the right value is empirical.
//
// FlyDSL writes this as raw inline asm (llvm.inline_asm "s_nop 7", side-effecting)
// because ROCDL has no s.nop op; at the LLVM-IR level we can use the real
// llvm.amdgcn.s.nop(i16) intrinsic instead.
static bool insertMemRegionNops(Function &F, int count) {
  if (count <= 0)
    return false;
  auto isRealBarrier = [](const Instruction &I) {
    if (const auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *F = CI->getCalledFunction())
        return F->getName().contains("amdgcn.s.barrier");
    return false;
  };
  // Flatten the function into layout order. A stage is delimited by the REAL
  // cluster barrier (amdgcn.s.barrier), and a stage can SPAN SEVERAL BASIC BLOCKS:
  // FAv4's mem2 carries the lazy-rescale branch (map_elementwise), which splits the
  // stage across 2-4 blocks. Walking per-block therefore never sees mem2's closing
  // barrier and skipped it entirely (mem1, a single block, was the only stage that
  // got its nops).
  SmallVector<Instruction *, 512> flat;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      flat.push_back(&I);

  bool changed = false;
  for (size_t i = 0; i < flat.size(); ++i) {
    if (!isRealBarrier(*flat[i]))
      continue;
    // Classify the stage that FOLLOWS this barrier, up to the next one.
    bool hasMfma = false, hasMem = false;
    size_t j = i + 1;
    for (; j < flat.size() && !isRealBarrier(*flat[j]); ++j) {
      if (Utils::isMFMAorWMMA(*flat[j])) {
        hasMfma = true;
        break;
      }
      SchedKind K = Utils::classifySchedInst(*flat[j]);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
        hasMem = true;
    }
    if (hasMfma || !hasMem)
      continue;
    // Insert at the head of that mem stage, i.e. right after the barrier.
    Instruction *ip = flat[i]->getNextNode();
    while (ip && isa<PHINode>(ip))
      ip = ip->getNextNode();
    if (!ip)
      continue;
    IRBuilder<> B(ip);
    for (int k = 0; k < count; ++k)
      B.CreateIntrinsic(Intrinsic::amdgcn_s_nop,
                        {B.getInt16(7)}); // s_nop 7 == 8 idle cycles
    changed = true;
  }
  return changed;
}

// Detect sched.barrier-delimited regions; interleave MFMA<->VALU in the ones
// that are independent (loop steady-state); skip dependent regions (prologue /
// coarse) and regions missing mfma or valu. Returns true if anything changed.
static bool scheduleRegions(Function &F) {
  const bool Dbg = std::getenv("LLIRSCHED_WP_DEBUG") != nullptr;
  // Declare the schedule via sched_group_barrier (IGroupLP) rather than physically
  // reordering + sched_barrier(0) pinning. This is the DEFAULT: pinning is only
  // advisory, and codegen was measured still consolidating a stage's last groups
  // despite it, so the declaration is the stronger form. LLIRSCHED_WP_NOSGB falls
  // back to the physical interleave (slower; kept for A/B).
  const bool SGB = std::getenv("LLIRSCHED_WP_NOSGB") == nullptr;
  // Region: mfma-bearing regions seen (the denominator worth reporting).
  // SyncID: identifies a *declared* region to IGroupLP. Keep it independent of how
  // many spans were skipped, so adding a skip reason cannot renumber the
  // declarations of the regions that do get scheduled.
  int Region = 0, Done = 0, SyncID = 1;
  bool Changed = false;
  for (BasicBlock &BB : F) {
    // Region boundaries = sched.barriers; process each span between them.
    Instruction *SpanBegin = &BB.front();
    SmallVector<std::pair<Instruction *, Instruction *>, 16> Spans;
    for (Instruction &I : BB) {
      if (isSchedBarrier(I)) {
        Spans.push_back({SpanBegin, &I});
        SpanBegin = I.getNextNode();
      }
    }
    if (SpanBegin)
      Spans.push_back({SpanBegin, nullptr});

    for (auto &S : Spans) {
      Instruction *B = S.first, *E = S.second;
      if (!B)
        continue;
      auto EIt = E ? E->getIterator() : BB.end();

      // Pick the model from the region's contents, not from the kernel's identity.
      int nMfma = 0, nValu = 0, nMem = 0;
      RegionModel Model = classifyRegion(B, EIt, nMfma, nValu, nMem);
      if (Model != RegionModel::CoExec) {
        // Throughput regions belong to the LLIRScheduler path, Mixed ones to nobody
        // yet; either way this namespace's co-exec model must not touch them --
        // it would spread VALU into windows that exist to cover memory latency.
        if (Model != RegionModel::None) {
          if (Dbg)
            errs() << "[wp-region " << Region << "] mfma=" << nMfma << " valu=" << nValu
                   << " mem=" << nMem << "  " << modelName(Model) << " -> skip\n";
          ++Region; // counts as a candidate: it has mfma, we just do not own it
        }
        continue;
      }

      SmallVector<Instruction *, 32> Mfmas, Valus;
      SmallPtrSet<const Instruction *, 32> MfmaSet;
      for (auto It = B->getIterator(); It != EIt; ++It) {
        if (Utils::isMFMAorWMMA(*It)) {
          Mfmas.push_back(&*It);
          MfmaSet.insert(&*It);
        } else if (valuWeight(*It) > 0) {
          Valus.push_back(&*It);
        }
      }
      if (Mfmas.empty() || Valus.empty())
        continue; // classifyRegion already guarantees both, belt and braces
      // Independence must hold BOTH ways (intra-iteration): no valu uses a
      // region mfma, AND no mfma uses a region valu (e.g. an mfma input built
      // from a fptrunc). Only then can mfmas move freely among the valu.
      SmallPtrSet<const Instruction *, 32> ValuSet(Valus.begin(), Valus.end());
      int Dep = 0;
      for (Instruction *V : Valus)
        if (dependsOnAny(V, MfmaSet))
          ++Dep;
      for (Instruction *M : Mfmas)
        if (dependsOnAny(M, ValuSet))
          ++Dep;
      bool Independent = (Dep == 0);
      if (Dbg)
        errs() << "[wp-region " << Region << "] mfma=" << Mfmas.size()
               << " valu=" << Valus.size() << " mem=" << nMem
               << " valu_dep_on_mfma=" << Dep << "  " << modelName(Model)
               << (Independent ? ", INDEPENDENT -> interleave" : ", DEPENDENT -> skip")
               << "\n";
      ++Region;
      if (Independent) {
        if (SGB) {
          // Pure declaration: do NOT physically reorder. The plugin only
          // *computes* the interleave (group sizes) and hands it to IGroupLP as
          // sched_group_barrier hints, which then builds the schedule itself.
          // syncID must be unique per region so IGroupLP solves each stage's
          // pipeline independently (FlyDSL uses one syncID per cluster too).
          if (declareRegionGroups(B, E, ++SyncID)) {
            ++Done;
            Changed = true;
          }
        } else {
          interleaveRegion(B, E);
          ++Done;
          Changed = true;
        }
      }
    }
  }
  // Mem-stage head pacing, on by default at the measured optimum. k=2 won on both
  // kernels (SOQ=0: 1234.3 vs 1227.4 at k=0 and 1218.8 at k=3; SOQ=1: 1244.9 vs
  // 1239.7 at k=3). LLIRSCHED_WP_MEMNOP=k overrides, and k=0 disables it.
  {
    int k = kDefaultMemNops;
    if (const char *nopEnv = std::getenv("LLIRSCHED_WP_MEMNOP"))
      k = atoi(nopEnv);
    if (k > 0 && insertMemRegionNops(F, k)) {
      Changed = true;
      if (Dbg)
        errs() << "[wp] inserted " << k << " x s_nop 7 at each mem-region head\n";
    }
  }
  if (Dbg)
    errs() << "[wp] " << F.getName().str() << ": interleaved " << Done << "/"
           << Region << " regions\n";
  return Changed;
}

} // namespace WP

// ---- New-PassManager wrapper (out-of-tree port of PR#73's legacy pass) ----
// Transactional: clone the function, schedule, verifyFunction, and roll back to
// the pristine body on failure (so a bad schedule never reaches codegen). The
// schedule is pinned by the sched.barriers the scheduler inserts, so nothing
// downstream (make_amdgcn) needs to disable LLVM's machine scheduler.
static bool runLlirScheduleTransactional(Function &F) {
  if (F.isDeclaration())
    return false;

  ValueToValueMapTy VMap;
  Function *Backup = CloneFunction(&F, VMap);
  Backup->setName(F.getName() + ".llirsched.bak");

  LLIRScheduler Scheduler;
  bool didSchedule = Scheduler.run(F);

  if (verifyFunction(F, /*OS=*/nullptr)) {
        if (std::getenv("LLIRSCHED_WP_DEBUG")) errs() << "[wp] ROLLBACK (invalid IR)\n";
    LLVM_DEBUG(dbgs() << "LLIR schedule produced invalid IR; rolling back.\n");
    auto OrigLinkage = F.getLinkage();
    F.deleteBody();
    F.splice(F.end(), Backup);
    F.setLinkage(OrigLinkage);
    for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
      Backup->getArg(i)->replaceAllUsesWith(F.getArg(i));
    Backup->eraseFromParent();
    return false;
  }

  Backup->eraseFromParent();
  return didSchedule;
}

struct LlirSchedPass : PassInfoMixin<LlirSchedPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
#ifdef LLIR_SCHED_REQUIRE_OPT_IN
    // Opt-in build (AITER): the plugin is loaded process-wide through
    // LLVM_PASS_PLUGIN_PATH, so only touch kernels that ask for it with the
    // function attribute below (set through Triton's `llvm_fn_attrs`). Every
    // other function is left byte-identical.
    if (!F.hasFnAttribute("aiter-llir-sched"))
      return PreservedAnalyses::all();
#endif
    // Warp-pipeline (Flash-Attention) kernels: regions are pre-delimited by
    // sched.barrier. Interleave MFMA<->VALU in the independent (steady-state)
    // regions. Transactional: clone, interleave, verify, roll back on failure.
    if (!F.isDeclaration() && WP::isWarpPipelineFunc(F)) {
      ValueToValueMapTy VMap;
      Function *Backup = CloneFunction(&F, VMap);
      Backup->setName(F.getName() + ".wpsched.bak");
      bool wpChanged = WP::scheduleRegions(F);
      if (verifyFunction(F, std::getenv("LLIRSCHED_WP_DEBUG") ? &errs() : nullptr)) {
        if (std::getenv("LLIRSCHED_WP_DEBUG")) errs() << "[wp] ROLLBACK (invalid IR)\n";
        LLVM_DEBUG(dbgs() << "wp interleave produced invalid IR; rolling back.\n");
        auto OrigLinkage = F.getLinkage();
        F.deleteBody();
        F.splice(F.end(), Backup);
        F.setLinkage(OrigLinkage);
        for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
          Backup->getArg(i)->replaceAllUsesWith(F.getArg(i));
        Backup->eraseFromParent();
        return PreservedAnalyses::all();
      }
      Backup->eraseFromParent();
      if (!wpChanged)
        return PreservedAnalyses::all();
      PreservedAnalyses PA;
      PA.preserveSet<CFGAnalyses>();
      return PA;
    }
    bool changed = runLlirScheduleTransactional(F);
    if (!changed)
      return PreservedAnalyses::all();
    // Only reorders/insert within blocks; CFG is preserved.
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
};

} // end anonymous namespace

llvm::PassPluginLibraryInfo getLlirSchedPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LlirSched", "v0.1",
          [](llvm::PassBuilder &PB) {
            // Auto-insert at the very end of the O3 function pipeline so the
            // pass sees near-final IR (individual mfma / ds_read / buffer_load).
            PB.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel,
                   llvm::ThinOrFullLTOPhase) {
                  MPM.addPass(
                      llvm::createModuleToFunctionPassAdaptor(LlirSchedPass()));
                });
            // Also allow explicit `-passes=llir-sched` for triton-opt/opt.
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "llir-sched") {
                    FPM.addPass(LlirSchedPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLlirSchedPluginInfo();
}
