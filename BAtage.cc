/*
 * Copyright (c) 2014 The University of Wisconsin
 *
 * Copyright (c) 2006 INRIA (Institut National de Recherche en
 * Informatique et en Automatique  / French National Research Institute
 * for Computer Science and Applied Mathematics)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Vignyan Reddy, Dibakar Gope and Arthur Perais,
 * from André Seznec's code.
 */

/* @file
 * Implementation of a BATAGE branch predictor
 */

#include "cpu/pred/BAtage.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/random.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"
#include "debug/BATage.hh"

BATAGE::BATAGE(const BATAGEParams *params) : BPredUnit(params), BAtage(params->BAtage)
{
}

// PREDICTOR UPDATE
void
BATAGE::update(ThreadID tid, Addr branch_pc, bool taken, void* bp_history,
              bool squashed, const StaticInstPtr & inst, Addr corrTarget)
{
    assert(bp_history);

    BATageBranchInfo *bi = static_cast<BATageBranchInfo*>(bp_history);
    BATAGEBase::BranchInfo *BAtage_bi = bi->BAtageBranchInfo;

    assert(corrTarget != MaxAddr);

    if (squashed) {
        // This restores the global history, then update it
        // and recomputes the folded histories.
        BAtage->squash(tid, taken, BAtage_bi, corrTarget);
        return;
    }

    int nrand = random_mt.random<int>() & 3;
    if (bi->BAtageBranchInfo->condBranch) {
        DPRINTF(BATage, "Updating tables for branch:%lx; taken?:%d\n",
                branch_pc, taken);
        BAtage->updateStats(taken, bi->BAtageBranchInfo);
        BAtage->condBranchUpdate(tid, branch_pc, taken, BAtage_bi, nrand,
                               corrTarget, bi->BAtageBranchInfo->BAtagePred);
    }

    // optional non speculative update of the histories
    BAtage->updateHistories(tid, branch_pc, taken, BAtage_bi, false, inst,
                          corrTarget);
    delete bi;
}

void
BATAGE::squash(ThreadID tid, void *bp_history)
{
    BATageBranchInfo *bi = static_cast<BATageBranchInfo*>(bp_history);
    DPRINTF(BATage, "Deleting branch info: %lx\n", bi->BAtageBranchInfo->branchPC);
    delete bi;
}

bool
BATAGE::predict(ThreadID tid, Addr branch_pc, bool cond_branch, void* &b)
{
    BATageBranchInfo *bi = new BATageBranchInfo(*BAtage);//nHistoryTables+1);
    b = (void*)(bi);
    return BAtage->BAtagePredict(tid, branch_pc, cond_branch, bi->BAtageBranchInfo);
}

bool
BATAGE::lookup(ThreadID tid, Addr branch_pc, void* &bp_history)
{
    bool retval = predict(tid, branch_pc, true, bp_history);

    BATageBranchInfo *bi = static_cast<BATageBranchInfo*>(bp_history);

    DPRINTF(BATage, "Lookup branch: %lx; predict:%d\n", branch_pc, retval);

    BAtage->updateHistories(tid, branch_pc, retval, bi->BAtageBranchInfo, true);

    return retval;
}

void
BATAGE::btbUpdate(ThreadID tid, Addr branch_pc, void* &bp_history)
{
    BATageBranchInfo *bi = static_cast<BATageBranchInfo*>(bp_history);
    BAtage->btbUpdate(tid, branch_pc, bi->BAtageBranchInfo);
}

void
BATAGE::uncondBranch(ThreadID tid, Addr br_pc, void* &bp_history)
{
    DPRINTF(BATage, "UnConditionalBranch: %lx\n", br_pc);
    predict(tid, br_pc, false, bp_history);
    BATageBranchInfo *bi = static_cast<BATageBranchInfo*>(bp_history);
    BAtage->updateHistories(tid, br_pc, true, bi->BAtageBranchInfo, true);
}

BATAGE*
BATAGEParams::create()
{
    return new BATAGE(this);
}
