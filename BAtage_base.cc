/*new
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

#include "cpu/pred/BAtage_base.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "debug/Fetch.hh"
#include "debug/BATage.hh"

BATAGEBase::BATAGEBase(const BATAGEBaseParams *p)
   : SimObject(p),
     logRatioBiModalHystEntries(p->logRatioBiModalHystEntries),
     nHistoryTables(p->nHistoryTables),
     tagTableCounterUpBits(p->tagTableCounterUpBits),
     tagTableCounterDownBits(p->tagTableCounterDownBits),
     //tagTableUBits(p->tagTableUBits),
     histBufferSize(p->histBufferSize),
     minHist(p->minHist),
     maxHist(p->maxHist),
     pathHistBits(p->pathHistBits),
     tagTableTagWidths(p->tagTableTagWidths),
     logTagTableSizes(p->logTagTableSizes),
     threadHistory(p->numThreads),
     logCTRResetPeriod(p->logCTRResetPeriod),
     initialTCounterValue(p->initialTCounterValue),
     numUseAltOnNa(p->numUseAltOnNa),
     // useAltOnNaBits(p->useAltOnNaBits),
     maxNumAlloc(p->maxNumAlloc),
     noSkip(p->noSkip),
     speculativeHistUpdate(p->speculativeHistUpdate),
     instShiftAmt(p->instShiftAmt),
     initialized(false)
{
    if (noSkip.empty()) {
        // Set all the table to enabled by default
        noSkip.resize(nHistoryTables + 1, true);
    }
}

BATAGEBase::BranchInfo*
BATAGEBase::makeBranchInfo() {
    return new BranchInfo(*this);
}

void
BATAGEBase::init()
{
    if (initialized) {
       return;
    }
    // Current method for periodically resetting the u counter bits only
    // works for 1 or 2 bits
    // Also make sure that it is not 0
    //assert(tagTableUBits <= 2 && (tagTableUBits > 0));

    // we use int type for the path history, so it cannot be more than
    // its size
    assert(pathHistBits <= (sizeof(int)*8));

    // initialize the counter to half of the period
    assert(logCTRResetPeriod != 0);
    tCounter = initialTCounterValue;

    assert(histBufferSize > maxHist * 2);

    useAltPredForNewlyAllocated.resize(numUseAltOnNa, 0);

    for (auto& history : threadHistory) {
        history.pathHist = 0;
        history.globalHistory = new uint8_t[histBufferSize];
        history.gHist = history.globalHistory;
        memset(history.gHist, 0, histBufferSize);
        history.ptGhist = 0;
    }

    histLengths = new int [nHistoryTables+1];

    calculateParameters();

    assert(tagTableTagWidths.size() == (nHistoryTables+1));
    assert(logTagTableSizes.size() == (nHistoryTables+1));

    // First entry is for the Bimodal table and it is untagged in this
    // implementation
    assert(tagTableTagWidths[0] == 0);

    for (auto& history : threadHistory) {
        history.computeIndices = new FoldedHistory[nHistoryTables+1];
        history.computeTags[0] = new FoldedHistory[nHistoryTables+1];
        history.computeTags[1] = new FoldedHistory[nHistoryTables+1];

        initFoldedHistories(history);
    }

    const uint64_t bimodalTableSize = ULL(1) << logTagTableSizes[0];
    btablePrediction.resize(bimodalTableSize, false);
    btableHysteresis.resize(bimodalTableSize >> logRatioBiModalHystEntries,
                            true);

    gtable = new BATageEntry*[nHistoryTables + 1];
    buildBATageTables();

    tableIndices = new int [nHistoryTables+1];
    tableTags = new int [nHistoryTables+1];
    initialized = true;
}

void
BATAGEBase::initFoldedHistories(ThreadHistory & history)
{
    for (int i = 1; i <= nHistoryTables; i++) {
        history.computeIndices[i].init(
            histLengths[i], (logTagTableSizes[i]));
        history.computeTags[0][i].init(
            history.computeIndices[i].origLength, tagTableTagWidths[i]);
        history.computeTags[1][i].init(
            history.computeIndices[i].origLength, tagTableTagWidths[i]-1);
        DPRINTF(BATage, "HistLength:%d, TTSize:%d, TTTWidth:%d\n",
                histLengths[i], logTagTableSizes[i], tagTableTagWidths[i]);
    }
}

void
BATAGEBase::buildBATageTables()
{
    for (int i = 1; i <= nHistoryTables; i++) {
        gtable[i] = new BATageEntry[1<<(logTagTableSizes[i])];
    }
}

void
BATAGEBase::calculateParameters()
{
    histLengths[1] = minHist;
    histLengths[nHistoryTables] = maxHist;

    for (int i = 2; i <= nHistoryTables; i++) {
        histLengths[i] = (int) (((double) minHist *
                       pow ((double) (maxHist) / (double) minHist,
                           (double) (i - 1) / (double) ((nHistoryTables- 1))))
                       + 0.5);
    }
}

void
BATAGEBase::btbUpdate(ThreadID tid, Addr branch_pc, BranchInfo* &bi)
{
    if (speculativeHistUpdate) {
        ThreadHistory& tHist = threadHistory[tid];
        DPRINTF(BATage, "BTB miss resets prediction: %lx\n", branch_pc);
        assert(tHist.gHist == &tHist.globalHistory[tHist.ptGhist]);
        tHist.gHist[0] = 0;
        for (int i = 1; i <= nHistoryTables; i++) {
            tHist.computeIndices[i].comp = bi->ci[i];
            tHist.computeTags[0][i].comp = bi->ct0[i];
            tHist.computeTags[1][i].comp = bi->ct1[i];
            tHist.computeIndices[i].update(tHist.gHist);
            tHist.computeTags[0][i].update(tHist.gHist);
            tHist.computeTags[1][i].update(tHist.gHist);
        }
    }
}

int
BATAGEBase::bindex(Addr pc_in) const
{
    return ((pc_in >> instShiftAmt) & ((ULL(1) << (logTagTableSizes[0])) - 1));
}

int
BATAGEBase::F(int A, int size, int bank) const
{
    int A1, A2;

    A = A & ((ULL(1) << size) - 1);
    A1 = (A & ((ULL(1) << logTagTableSizes[bank]) - 1));
    A2 = (A >> logTagTableSizes[bank]);
    A2 = ((A2 << bank) & ((ULL(1) << logTagTableSizes[bank]) - 1))
       + (A2 >> (logTagTableSizes[bank] - bank));
    A = A1 ^ A2;
    A = ((A << bank) & ((ULL(1) << logTagTableSizes[bank]) - 1))
      + (A >> (logTagTableSizes[bank] - bank));
    return (A);
}

// gindex computes a full hash of pc, ghist and pathHist
int
BATAGEBase::gindex(ThreadID tid, Addr pc, int bank) const
{
    int index;
    int hlen = (histLengths[bank] > pathHistBits) ? pathHistBits :
                                                    histLengths[bank];
    const unsigned int shiftedPc = pc >> instShiftAmt;
    index =
        shiftedPc ^
        (shiftedPc >> ((int) abs(logTagTableSizes[bank] - bank) + 1)) ^
        threadHistory[tid].computeIndices[bank].comp ^
        F(threadHistory[tid].pathHist, hlen, bank);

    return (index & ((ULL(1) << (logTagTableSizes[bank])) - 1));
}


// Tag computation
uint16_t
BATAGEBase::gtag(ThreadID tid, Addr pc, int bank) const
{
    int tag = (pc >> instShiftAmt) ^
              threadHistory[tid].computeTags[0][bank].comp ^
              (threadHistory[tid].computeTags[1][bank].comp << 1);

    return (tag & ((ULL(1) << tagTableTagWidths[bank]) - 1));
}


// Up-down saturating counter
template<typename T>
void
BATAGEBase::ctrUpdate(T & ctr_up, T & ctr_down, bool taken, int nbits)
{
    //assert(nbits <= sizeof(T) << 3);
    /*if (taken) {
        if (ctr < ((1 << (nbits - 1)) - 1))
            ctr++;
    } else {
        if (ctr > -(1 << (nbits - 1)))
            ctr--;
    }*/
    assert(nbits == 3);
    if (taken) {
        if (ctr_up < ((1 << (nbits)) - 1))
            ctr_up++;
        else if (ctr_down > 0)
            ctr_down--;
    }
    else {
        if (ctr_down < ((1 << (nbits)) - 1))
            ctr_down++;
        else if (ctr_up > 0)
            ctr_up--;
    }

}

// int8_t and int versions of this function may be needed
// template void BATAGEBase::ctrUpdate(int8_t & ctr, bool taken, int nbits);
// template void BATAGEBase::ctrUpdate(int & ctr, bool taken, int nbits);

// Up-down unsigned saturating counter
void
BATAGEBase::unsignedCtrUpdate(uint8_t & ctr_up, uint8_t & ctr_down, bool up, unsigned nbits)
{
    //assert(nbits <= sizeof(uint8_t) << 3);
    /*if (up) {
        if (ctr < ((1 << nbits) - 1))
            ctr++;
    } else {
        if (ctr)
            ctr--;
    }*/
    assert(nbits == 3);
    if (up) {
        if (ctr_up < ((1 << (nbits)) - 1))
            ctr_up++;
        else if (ctr_down > 0)
            ctr_down--;
    }
    else {
        if (ctr_down < ((1 << (nbits)) - 1))
            ctr_down++;
        else if (ctr_up > 0)
            ctr_up--;
    }

}

// Bimodal prediction
bool
BATAGEBase::getBimodePred(Addr pc, BranchInfo* bi) const
{
    return btablePrediction[bi->bimodalIndex];
}


// Update the bimodal predictor: a hysteresis bit is shared among N prediction
// bits (N = 2 ^ logRatioBiModalHystEntries)
void
BATAGEBase::baseUpdate(Addr pc, bool taken, BranchInfo* bi)
{
    int inter = (btablePrediction[bi->bimodalIndex] << 1)
        + btableHysteresis[bi->bimodalIndex >> logRatioBiModalHystEntries];
    if (taken) {
        if (inter < 3)
            inter++;
    } else if (inter > 0) {
        inter--;
    }
    const bool pred = inter >> 1;
    const bool hyst = inter & 1;
    btablePrediction[bi->bimodalIndex] = pred;
    btableHysteresis[bi->bimodalIndex >> logRatioBiModalHystEntries] = hyst;
    DPRINTF(BATage, "Updating branch %lx, pred:%d, hyst:%d\n", pc, pred, hyst);
}

// shifting the global history:  we manage the history in a big table in order
// to reduce simulation time
void
BATAGEBase::updateGHist(uint8_t * &h, bool dir, uint8_t * tab, int &pt)
{
    if (pt == 0) {
        DPRINTF(BATage, "Rolling over the histories\n");
         // Copy beginning of globalHistoryBuffer to end, such that
         // the last maxHist outcomes are still reachable
         // through pt[0 .. maxHist - 1].
         for (int i = 0; i < maxHist; i++)
             tab[histBufferSize - maxHist + i] = tab[i];
         pt =  histBufferSize - maxHist;
         h = &tab[pt];
    }
    pt--;
    h--;
    h[0] = (dir) ? 1 : 0;
}

void
BATAGEBase::calculateIndicesAndTags(ThreadID tid, Addr branch_pc,
                                  BranchInfo* bi)
{
    // computes the table addresses and the partial tags
    for (int i = 1; i <= nHistoryTables; i++) {
        tableIndices[i] = gindex(tid, branch_pc, i);
        bi->tableIndices[i] = tableIndices[i];
        tableTags[i] = gtag(tid, branch_pc, i);
        bi->tableTags[i] = tableTags[i];
    }
}

unsigned
BATAGEBase::getUseAltIdx(BranchInfo* bi, Addr branch_pc)
{
    // There is only 1 counter on the base BATAGE implementation
    return 0;
}


int
BATAGEBase::getConfidence(int ctr_up, int ctr_down)
{
    int minCtr = (ctr_up > ctr_down) ? ctr_down : ctr_up;
    
    double confidence = (1 + minCtr)/(2+ ctr_up + ctr_down);

    if (confidence < 1/3)
        confidence = 0;
    else if (confidence == 1/3)
        confidence = 1;
    else
        confidence = 2;
    return confidence;

}


bool
BATAGEBase::BAtagePredict(ThreadID tid, Addr branch_pc,
              bool cond_branch, BranchInfo* bi)
{
    Addr pc = branch_pc;
    bool pred_taken = true;

    if (cond_branch) {
        // BATAGE prediction

        calculateIndicesAndTags(tid, pc, bi);

        bi->bimodalIndex = bindex(pc);

        bi->hitBank = 0;
        bi->altBank = 0;
        double confidence = 0;
        double bestConf = 4;
        //Look for the bank with longest matching history
        for (int i = nHistoryTables; i > 0; i--) {
            if (noSkip[i] &&
                gtable[i][tableIndices[i]].tag == tableTags[i]) {

                confidence = getConfidence(gtable[i][tableIndices[i]].ctr_up, 
                                            gtable[i][tableIndices[i]].ctr_down);
                if (confidence <= bestConf){
                    bi->hitBank = i;
                    bi->hitBankIndex = tableIndices[bi->hitBank];
                    bestConf = confidence;
                }
                break;
            }
        }
        //Look for the alternate bank
        bestConf = 4;
        for (int i = bi->hitBank - 1; i > 0; i--) {
            if (noSkip[i] &&
                gtable[i][tableIndices[i]].tag == tableTags[i]) {

                confidence = getConfidence(gtable[i][tableIndices[i]].ctr_up, 
                                            gtable[i][tableIndices[i]].ctr_down);

                if (confidence <= bestConf){
                    bi->altBank = i;
                    bi->altBankIndex = tableIndices[bi->altBank];
                    bestConf=confidence;
                }
                
                break;
            }
        }
        //computes the prediction and the alternate prediction
        if (bi->hitBank > 0) {
            if (bi->altBank > 0) {
                bi->altTaken =
                    ((gtable[bi->altBank][tableIndices[bi->altBank]].ctr_up - 
                    gtable[bi->altBank][tableIndices[bi->altBank]].ctr_down) >= 0
                    );
                extraAltCalc(bi);
            }else {
                bi->altTaken = getBimodePred(pc, bi);
            }

            bi->longestMatchPred =
                ((gtable[bi->hitBank][tableIndices[bi->hitBank]].ctr_up - 
                gtable[bi->hitBank][tableIndices[bi->hitBank]].ctr_down) >= 0);
            bi->pseudoNewAlloc =
                abs(2 * (gtable[bi->hitBank][bi->hitBankIndex].ctr_up - gtable[bi->hitBank][bi->hitBankIndex].ctr_down) + 1) <= 1;

            //if the entry is recognized as a newly allocated entry and
            //useAltPredForNewlyAllocated is positive use the alternate
            //prediction
            if ((useAltPredForNewlyAllocated[getUseAltIdx(bi, branch_pc)] < 0)
                || ! bi->pseudoNewAlloc) {
                bi->BAtagePred = bi->longestMatchPred;
                bi->provider = BATAGE_LONGEST_MATCH;
            } else {
                bi->BAtagePred = bi->altTaken;
                bi->provider = bi->altBank ? BATAGE_ALT_MATCH
                                           : BIMODAL_ALT_MATCH;
            }
        } else {
            bi->altTaken = getBimodePred(pc, bi);
            bi->BAtagePred = bi->altTaken;
            bi->longestMatchPred = bi->altTaken;
            bi->provider = BIMODAL_ONLY;
        }
        //end BATAGE prediction

        pred_taken = (bi->BAtagePred);
        DPRINTF(BATage, "Predict for %lx: taken?:%d, BAtagePred:%d, altPred:%d\n",
                branch_pc, pred_taken, bi->BAtagePred, bi->altTaken);
    }
    bi->branchPC = branch_pc;
    bi->condBranch = cond_branch;
    return pred_taken;
}

void
BATAGEBase::adjustAlloc(bool & alloc, bool taken, bool pred_taken)
{
    // Nothing for this base class implementation
}

void
BATAGEBase::handleAllocAndUReset(bool alloc, bool taken, BranchInfo* bi,
                           int nrand)
{
    if (alloc) {
        // is there some "unuseful" entry to allocate
        uint8_t worstConfidence = 0;
        double confidence;
        for (int i = nHistoryTables; i > bi->hitBank; i--) {
            confidence = getConfidence(gtable[i][tableIndices[i]].ctr_up, 
                                            gtable[i][tableIndices[i]].ctr_down);
            if (confidence >= 1)
                worstConfidence = confidence;
            // if (gtable[i][bi->tableIndices[i]].u < min) {
            //     min = gtable[i][bi->tableIndices[i]].u;
            // }
        }

        // we allocate an entry with a longer history
        // to  avoid ping-pong, we do not choose systematically the next
        // entry, but among the 3 next entries
        int Y = nrand &
            ((ULL(1) << (nHistoryTables - bi->hitBank - 1)) - 1);
        int X = bi->hitBank + 1;
        if (Y & 1) {
            X++;
            if (Y & 2)
                X++;
        }
        // No entry available, forces one to be available
        if (worstConfidence < 1) {
            gtable[X][bi->tableIndices[X]].ctr_up = 0;
            gtable[X][bi->tableIndices[X]].ctr_down = 0;
        }


        //Allocate entries
        unsigned numAllocated = 0;
        for (int i = X; i <= nHistoryTables; i++) {
            confidence = getConfidence(gtable[i][tableIndices[i]].ctr_up, 
                                            gtable[i][tableIndices[i]].ctr_down);

            if (confidence >= 1){
                gtable[i][bi->tableIndices[i]].tag = bi->tableTags[i];
                gtable[i][bi->tableIndices[i]].ctr_up = (taken) ? 1: 0;
                gtable[i][bi->tableIndices[i]].ctr_down = (taken) ? 0: 1;
                ++numAllocated;
                if (numAllocated == maxNumAlloc) {
                    break;
                }
            }
            // if ((gtable[i][bi->tableIndices[i]].u == 0)) {
            //     gtable[i][bi->tableIndices[i]].tag = bi->tableTags[i];
            //     gtable[i][bi->tableIndices[i]].ctr = (taken) ? 0 : -1;
            //     ++numAllocated;
            //     if (numAllocated == maxNumAlloc) {
            //         break;
            //     }
            // }
        }
    }

    tCounter++;

    handleUReset();
}

void
BATAGEBase::handleUReset()
{
    //periodic reset of u: reset is not complete but bit by bit
    if ((tCounter & ((ULL(1) << logCTRResetPeriod) - 1)) == 0) {
        // reset least significant bit
        // most significant bit becomes least significant bit
        for (int i = 1; i <= nHistoryTables; i++) {
            for (int j = 0; j < (ULL(1) << logTagTableSizes[i]); j++) {
                resetUctr(gtable[i][j].ctr_up, gtable[i][j].ctr_down);
            }
        }
    }
}

void
BATAGEBase::resetUctr(uint8_t & ctr_up, uint8_t & ctr_down)
{
    //u >>= 1;
    if (ctr_up > ctr_down){
        ctr_up--;
    }
    else if (ctr_down > ctr_up){
        ctr_down--;
    }
}

void
BATAGEBase::condBranchUpdate(ThreadID tid, Addr branch_pc, bool taken,
    BranchInfo* bi, int nrand, Addr corrTarget, bool pred, bool preAdjustAlloc)
{
    // BATAGE UPDATE
    // try to allocate a  new entries only if prediction was wrong
    bool alloc = (bi->BAtagePred != taken) && (bi->hitBank < nHistoryTables);

    if (preAdjustAlloc) {
        adjustAlloc(alloc, taken, pred);
    }

    if (bi->hitBank > 0) {
        // Manage the selection between longest matching and alternate
        // matching for "pseudo"-newly allocated longest matching entry
        bool PseudoNewAlloc = bi->pseudoNewAlloc;
        // an entry is considered as newly allocated if its prediction
        // counter is weak
        if (PseudoNewAlloc) {
            if (bi->longestMatchPred == taken) {
                alloc = false;
            }
            // if it was delivering the correct prediction, no need to
            // allocate new entry even if the overall prediction was false
            if (bi->longestMatchPred != bi->altTaken) {
                ctrUpdate(
                    useAltPredForNewlyAllocated[getUseAltIdx(bi, branch_pc)],
                    useAltPredForNewlyAllocated[getUseAltIdx(bi, branch_pc)],
                    bi->altTaken == taken, 3);
                    // bi->altTaken == taken, useAltOnNaBits);
            }
        }
    }

    if (!preAdjustAlloc) {
        adjustAlloc(alloc, taken, pred);
    }

    handleAllocAndUReset(alloc, taken, bi, nrand);

    handleBATAGEUpdate(branch_pc, taken, bi);
}

void
BATAGEBase::handleBATAGEUpdate(Addr branch_pc, bool taken, BranchInfo* bi)
{
    if (bi->hitBank > 0) {
        
        double confidence;

        DPRINTF(BATage, "Updating tag table entry (%d,%d) for branch %lx\n",
                bi->hitBank, bi->hitBankIndex, branch_pc);
        ctrUpdate(gtable[bi->hitBank][bi->hitBankIndex].ctr_up, gtable[bi->hitBank][bi->hitBankIndex].ctr_down, taken,
                  3);
        // if the provider entry is not certified to be useful also update
        // the alternate prediction

        //if (gtable[bi->hitBank][bi->hitBankIndex].u == 0) {
        confidence = getConfidence(gtable[bi->hitBank][bi->hitBankIndex].ctr_up, 
                                            gtable[bi->hitBank][bi->hitBankIndex].ctr_down);
        if (confidence >=1) {
            if (bi->altBank > 0) {
                ctrUpdate(gtable[bi->altBank][bi->altBankIndex].ctr_up, gtable[bi->hitBank][bi->hitBankIndex].ctr_down, taken,
                          3);
                DPRINTF(BATage, "Updating tag table entry (%d,%d) for"
                        " branch %lx\n", bi->hitBank, bi->hitBankIndex,
                        branch_pc);
            }
            if (bi->altBank == 0) {
                baseUpdate(branch_pc, taken, bi);
            }
        }

        // update the u counter
        if (bi->BAtagePred != bi->altTaken) {
            unsignedCtrUpdate(gtable[bi->hitBank][bi->hitBankIndex].ctr_up,gtable[bi->hitBank][bi->hitBankIndex].ctr_down,
                              bi->BAtagePred == taken, 3);
        }
    } else {
        baseUpdate(branch_pc, taken, bi);
    }
}

void
BATAGEBase::updateHistories(ThreadID tid, Addr branch_pc, bool taken,
                          BranchInfo* bi, bool speculative,
                          const StaticInstPtr &inst, Addr target)
{
    if (speculative != speculativeHistUpdate) {
        return;
    }
    ThreadHistory& tHist = threadHistory[tid];
    //  UPDATE HISTORIES
    bool pathbit = ((branch_pc >> instShiftAmt) & 1);
    //on a squash, return pointers to this and recompute indices.
    //update user history
    updateGHist(tHist.gHist, taken, tHist.globalHistory, tHist.ptGhist);
    tHist.pathHist = (tHist.pathHist << 1) + pathbit;
    tHist.pathHist = (tHist.pathHist & ((ULL(1) << pathHistBits) - 1));

    if (speculative) {
        bi->ptGhist = tHist.ptGhist;
        bi->pathHist = tHist.pathHist;
    }

    //prepare next index and tag computations for user branchs
    for (int i = 1; i <= nHistoryTables; i++)
    {
        if (speculative) {
            bi->ci[i]  = tHist.computeIndices[i].comp;
            bi->ct0[i] = tHist.computeTags[0][i].comp;
            bi->ct1[i] = tHist.computeTags[1][i].comp;
        }
        tHist.computeIndices[i].update(tHist.gHist);
        tHist.computeTags[0][i].update(tHist.gHist);
        tHist.computeTags[1][i].update(tHist.gHist);
    }
    DPRINTF(BATage, "Updating global histories with branch:%lx; taken?:%d, "
            "path Hist: %x; pointer:%d\n", branch_pc, taken, tHist.pathHist,
            tHist.ptGhist);
    assert(threadHistory[tid].gHist ==
            &threadHistory[tid].globalHistory[threadHistory[tid].ptGhist]);
}

void
BATAGEBase::squash(ThreadID tid, bool taken, BATAGEBase::BranchInfo *bi,
                 Addr target)
{
    if (!speculativeHistUpdate) {
        /* If there are no speculative updates, no actions are needed */
        return;
    }

    ThreadHistory& tHist = threadHistory[tid];
    DPRINTF(BATage, "Restoring branch info: %lx; taken? %d; PathHistory:%x, "
            "pointer:%d\n", bi->branchPC,taken, bi->pathHist, bi->ptGhist);
    tHist.pathHist = bi->pathHist;
    tHist.ptGhist = bi->ptGhist;
    tHist.gHist = &(tHist.globalHistory[tHist.ptGhist]);
    tHist.gHist[0] = (taken ? 1 : 0);
    for (int i = 1; i <= nHistoryTables; i++) {
        tHist.computeIndices[i].comp = bi->ci[i];
        tHist.computeTags[0][i].comp = bi->ct0[i];
        tHist.computeTags[1][i].comp = bi->ct1[i];
        tHist.computeIndices[i].update(tHist.gHist);
        tHist.computeTags[0][i].update(tHist.gHist);
        tHist.computeTags[1][i].update(tHist.gHist);
    }
}

void
BATAGEBase::extraAltCalc(BranchInfo* bi)
{
    // do nothing. This is only used in some derived classes
    return;
}

void
BATAGEBase::updateStats(bool taken, BranchInfo* bi)
{
    if (taken == bi->BAtagePred) {
        // correct prediction
        switch (bi->provider) {
          case BIMODAL_ONLY: BAtageBimodalProviderCorrect++; break;
          case BATAGE_LONGEST_MATCH: BAtageLongestMatchProviderCorrect++; break;
          case BIMODAL_ALT_MATCH: bimodalAltMatchProviderCorrect++; break;
          case BATAGE_ALT_MATCH: BAtageAltMatchProviderCorrect++; break;
        }
    } else {
        // wrong prediction
        switch (bi->provider) {
          case BIMODAL_ONLY: BAtageBimodalProviderWrong++; break;
          case BATAGE_LONGEST_MATCH:
            BAtageLongestMatchProviderWrong++;
            if (bi->altTaken == taken) {
                BAtageAltMatchProviderWouldHaveHit++;
            }
            break;
          case BIMODAL_ALT_MATCH:
            bimodalAltMatchProviderWrong++;
            break;
          case BATAGE_ALT_MATCH:
            BAtageAltMatchProviderWrong++;
            break;
        }

        switch (bi->provider) {
          case BIMODAL_ALT_MATCH:
          case BATAGE_ALT_MATCH:
            if (bi->longestMatchPred == taken) {
                BAtageLongestMatchProviderWouldHaveHit++;
            }
        }
    }

    switch (bi->provider) {
      case BATAGE_LONGEST_MATCH:
      case BATAGE_ALT_MATCH:
        BAtageLongestMatchProvider[bi->hitBank]++;
        BAtageAltMatchProvider[bi->altBank]++;
        break;
    }
}

unsigned
BATAGEBase::getGHR(ThreadID tid, BranchInfo *bi) const
{
    unsigned val = 0;
    for (unsigned i = 0; i < 32; i++) {
        // Make sure we don't go out of bounds
        int gh_offset = bi->ptGhist + i;
        assert(&(threadHistory[tid].globalHistory[gh_offset]) <
               threadHistory[tid].globalHistory + histBufferSize);
        val |= ((threadHistory[tid].globalHistory[gh_offset] & 0x1) << i);
    }

    return val;
}

void
BATAGEBase::regStats()
{
    BAtageLongestMatchProviderCorrect
        .name(name() + ".BAtageLongestMatchProviderCorrect")
        .desc("Number of times BATAGE Longest Match is the provider and "
              "the prediction is correct");

    BAtageAltMatchProviderCorrect
        .name(name() + ".BAtageAltMatchProviderCorrect")
        .desc("Number of times BATAGE Alt Match is the provider and "
              "the prediction is correct");

    bimodalAltMatchProviderCorrect
        .name(name() + ".bimodalAltMatchProviderCorrect")
        .desc("Number of times BATAGE Alt Match is the bimodal and it is the "
              "provider and the prediction is correct");

    BAtageBimodalProviderCorrect
        .name(name() + ".BAtageBimodalProviderCorrect")
        .desc("Number of times there are no hits on the BATAGE tables "
              "and the bimodal prediction is correct");

    BAtageLongestMatchProviderWrong
        .name(name() + ".BAtageLongestMatchProviderWrong")
        .desc("Number of times BATAGE Longest Match is the provider and "
              "the prediction is wrong");

    BAtageAltMatchProviderWrong
        .name(name() + ".BAtageAltMatchProviderWrong")
        .desc("Number of times BATAGE Alt Match is the provider and "
              "the prediction is wrong");

    bimodalAltMatchProviderWrong
        .name(name() + ".bimodalAltMatchProviderWrong")
        .desc("Number of times BATAGE Alt Match is the bimodal and it is the "
              "provider and the prediction is wrong");

    BAtageBimodalProviderWrong
        .name(name() + ".BAtageBimodalProviderWrong")
        .desc("Number of times there are no hits on the BATAGE tables "
              "and the bimodal prediction is wrong");

    BAtageAltMatchProviderWouldHaveHit
        .name(name() + ".BAtageAltMatchProviderWouldHaveHit")
        .desc("Number of times BATAGE Longest Match is the provider, "
              "the prediction is wrong and Alt Match prediction was correct");

    BAtageLongestMatchProviderWouldHaveHit
        .name(name() + ".BAtageLongestMatchProviderWouldHaveHit")
        .desc("Number of times BATAGE Alt Match is the provider, the "
              "prediction is wrong and Longest Match prediction was correct");

    BAtageLongestMatchProvider
        .init(nHistoryTables + 1)
        .name(name() + ".BAtageLongestMatchProvider")
        .desc("BATAGE provider for longest match");

    BAtageAltMatchProvider
        .init(nHistoryTables + 1)
        .name(name() + ".BAtageAltMatchProvider")
        .desc("BATAGE provider for alt match");
}

int8_t
BATAGEBase::getCtr_up(int hitBank, int hitBankIndex) const
{
    return gtable[hitBank][hitBankIndex].ctr_up;
}

int8_t
BATAGEBase::getCtr_down(int hitBank, int hitBankIndex) const
{
    return gtable[hitBank][hitBankIndex].ctr_down;
}


unsigned
BATAGEBase::getBATageCtrUpBits() const
{
    return tagTableCounterUpBits;
}

unsigned
BATAGEBase::getBATageCtrDownBits() const
{
    return tagTableCounterDownBits;
}

int
BATAGEBase::getPathHist(ThreadID tid) const
{
    return threadHistory[tid].pathHist;
}

bool
BATAGEBase::isSpeculativeUpdateEnabled() const
{
    return speculativeHistUpdate;
}

size_t
BATAGEBase::getSizeInBits() const {
    size_t bits = 0;
    for (int i = 1; i <= nHistoryTables; i++) {
        bits += (1 << logTagTableSizes[i]) *
            (tagTableCounterUpBits + tagTableCounterDownBits + tagTableTagWidths[i]);
    }
    uint64_t bimodalTableSize = ULL(1) << logTagTableSizes[0];
    // bits += numUseAltOnNa * useAltOnNaBits;
    bits += numUseAltOnNa * 3;
    bits += bimodalTableSize;
    bits += (bimodalTableSize >> logRatioBiModalHystEntries);
    bits += histLengths[nHistoryTables];
    bits += pathHistBits;
    bits += logCTRResetPeriod;
    return bits;
}

BATAGEBase*
BATAGEBaseParams::create()
{
    return new BATAGEBase(this);
}
