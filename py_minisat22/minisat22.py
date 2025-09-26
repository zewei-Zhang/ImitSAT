"""
Python Minisat 2.2 Implementation.

This implementation trys to retain the structure, variable names,
and function names of Minisat 2.2, ensuring consistency with the original C++ version.
This implementation is inspired by the Minisat solver by Niklas Een and Niklas Sorensson.

Original Minisat License:
MIT License

Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--------------------------------------------------------------------------------------------------

Additional modifications and original work by Zewei Zhang.
"""
import math

from typing import List, Optional


# lbool constants as an enum-like structure
# In Minisat: l_True = 0, l_False = 1, l_Undef = 2
class Lbool:
    TRUE = 0
    FALSE = 1
    UNDEF = 2


def lbool(val: Optional[bool]) -> int:
    if val is True:
        return Lbool.TRUE
    if val is False:
        return Lbool.FALSE
    return Lbool.UNDEF


def sign(lit) -> bool:
    return lit & 1 == 1


def var(lit) -> int:
    return lit >> 1


def mkLit(var_index: int, sign_bit: bool) -> int:
    return (var_index << 1) + (1 if sign_bit else 0)


def compute_next_decision_value(x: int) -> int:
    """
    Map a decision value from trace for minisat.

    Args:
        x (int): Input integer.

    Returns:
        int: Computed next value.
    """
    return 2 * (x - 1) if x > 0 else -2 * x - 1


# Clause reference
# In Minisat clauses are referenced by indices in ClauseAllocator
class CRef:
    UNDEF = -1


class Lit:
    def __init__(self, x: int):
        self.x = x

    def __eq__(self, other: 'Lit') -> bool:
        return self.x == other.x

    def __ne__(self, other: 'Lit') -> bool:
        return not self == other

    def __lt__(self, other: 'Lit') -> bool:
        return self.x < other.x

    def __invert__(self) -> 'Lit':
        return Lit(self.x ^ 1)  # Negate the literal

    def __xor__(self, b: bool) -> 'Lit':
        return Lit(self.x ^ int(b))

    @property
    def sign(self) -> bool:
        return self.x & 1

    @property
    def var(self) -> int:
        return self.x >> 1

    @staticmethod
    def mkLit(var: int, sign: bool = False) -> 'Lit':
        return Lit(var + var + int(sign))


class lbool:
    def __init__(self, value: int):
        self.value = value  # Represents l_True, l_False, l_Undef

    def __eq__(self, other: 'lbool') -> bool:
        return (self.value & 2 and other.value & 2) or (self.value == other.value)

    def __ne__(self, other: 'lbool') -> bool:
        return not self == other

    def __and__(self, other: 'lbool') -> 'lbool':
        sel = (self.value << 1) | (other.value << 3)
        v = (0xF7F755F4 >> sel) & 3
        return lbool(v)

    def __or__(self, other: 'lbool') -> 'lbool':
        sel = (self.value << 1) | (other.value << 3)
        v = (0xFCFCF400 >> sel) & 3
        return lbool(v)

    def __xor__(self, b: bool) -> 'lbool':
        return lbool(self.value ^ int(b))

    @staticmethod
    def toInt(l: 'lbool') -> int:
        return l.value

    @staticmethod
    def toLbool(v: int) -> 'lbool':
        return lbool(v)


def toInt(v: int) -> int:
    return v


def toInt(l: Lit) -> int:
    return l.x


def toLit(i: int) -> Lit:
    return Lit(i)


# Replacement for vec<T> by Python list.
# Minisat uses vec<T> extensively, we can just use Python lists.
# We'll keep methods similar in name for easier mapping.

def remove(lst, item):
    for i, x in enumerate(lst):
        if x == item:
            del lst[i]
            return


# Options, simplified
class DoubleOption:
    def __init__(self, category, name, desc, default, drange):
        self.value = default


class IntOption:
    def __init__(self, category, name, desc, default, irange):
        self.value = default


class BoolOption:
    def __init__(self, category, name, desc, default):
        self.value = default


class VarData:
    def __init__(self, reason: int, level: int):
        self.reason = reason  # Reference to a clause
        self.level = level  # Decision level


def mkVarData(reason: int, level: int) -> VarData:
    return VarData(reason, level)


class Clause:
    def __init__(self, literals: List[int], learnt: bool = False, has_extra: bool = False):
        self.lits = literals
        self._learnt = learnt
        self.has_extra = has_extra
        self._activity = 0.0
        self._mark = 0
        self._reloced = False
        self.relocation = None
        self.abstraction = self.calc_abstraction() if has_extra and not learnt else None

        self.original_lits = None

    def calc_abstraction(self):
        abstraction = 0
        for lit in self.lits:
            abstraction |= 1 << (lit >> 1 & 31)
        return abstraction

    def size(self) -> int:
        return len(self.lits)

    def learnt(self) -> bool:
        return self._learnt

    def activity(self) -> float:
        return self._activity

    def mark(self, m: int):
        self._mark = m

    def mark_value(self) -> int:
        return self._mark

    def __getitem__(self, i: int) -> int:
        return self.lits[i]

    def __setitem__(self, i: int, val: int):
        self.lits[i] = val

    def reloced(self) -> bool:
        return self._reloced

    def set_reloced(self, val: bool):
        self._reloced = val

    def set_activity(self, activity: float):
        self._activity = activity

    def relocate(self, cref: int):
        self.set_reloced(True)
        self.relocation = cref


class ClauseAllocator:
    Unit_Size = 1

    def __init__(self, start_cap: int = 1024):
        self.db = []
        self.deleted = []
        self.cap = start_cap
        self.extra_clause_field = False

    def alloc(self, ps: List[int], learnt: bool) -> int:
        c = Clause(ps[:], has_extra=self.extra_clause_field, learnt=learnt)
        cref = len(self.db)
        self.db.append(c)
        c.cref = cref
        return cref

    def free(self, cr: int):
        self.db[cr] = None
        self.deleted.append(cr)

    def __getitem__(self, cr: int) -> Clause:
        if cr < 0 or cr >= len(self.db) or self.db[cr] is None:
            raise ValueError(f"Invalid cref {cr} in ClauseAllocator.")
        return self.db[cr]

    def reloc(self, cr: int, to):
        if cr == CRef.UNDEF:
            return cr
        c = self.db[cr]
        if c is None:
            return cr
        if c.reloced():
            return c.relocation
        new_cref = to.alloc(c.lits, c.learnt())
        nc = to[new_cref]
        nc.set_activity(c.activity())
        nc.mark(c.mark_value())
        if not c.learnt() and nc.has_extra:
            nc.abstraction = nc.calc_abstraction()

        c.relocate(new_cref)
        return new_cref

    def size(self):
        return len(self.db)

    def wasted(self):
        return len(self.deleted)

    def moveTo(self, to):
        to.extra_clause_field = self.extra_clause_field

        to.db = self.db
        to.deleted = self.deleted
        to.cap = self.cap


class Watcher:
    def __init__(self, cref: int, blocker: int):
        self.cref = cref
        self.blocker = blocker

    def __eq__(self, other: 'Watcher') -> bool:
        return self.cref == other.cref

    def __ne__(self, other: 'Watcher') -> bool:
        return not self == other


class WatcherDeleted:
    def __init__(self, ca: ClauseAllocator):
        self.ca = ca

    def __call__(self, w: Watcher) -> bool:
        if w.cref < 0 or w.cref >= len(self.ca.db):
            return True

        clause = self.ca.db[w.cref]
        if clause is None:
            return True
        return clause.mark_value() == 1


class Watches:
    def __init__(self, deleted):
        self.table = {}
        self.deleted = deleted

    def __getitem__(self, lit: int) -> List[Watcher]:
        if lit not in self.table:
            self.table[lit] = []
        return self.table[lit]

    def init(self, lit: int):
        if lit not in self.table:
            self.table[lit] = []

    def update(self, lit: int, watchers: List[Watcher]):
        """Update the watchers for a literal."""
        self.table[lit] = watchers

    def cleanAll(self):
        """Clean all watcher lists by removing deleted watchers."""
        for lit in self.table:
            original_length = len(self.table[lit])
            self.table[lit] = [w for w in self.table[lit] if not self.deleted(w)]

    def smudge(self, lit: int):
        # Lazy detaching simulation: do nothing here
        pass


class VarOrderLt:
    def __init__(self, activity):
        self.activity = activity

    def __call__(self, x, y):
        return self.activity[x] > self.activity[y]


class Heap:
    def __init__(self, comp):
        self.data = []
        self.comp = comp
        self.index = {}  # var -> position in heap, or -1 if not in heap

    def empty(self):
        return len(self.data) == 0

    def size(self):
        return len(self.data)

    def inHeap(self, x):
        return x in self.index and self.index[x] != -1

    def percolateUp(self, i):
        x = self.data[i]
        while i > 0:
            p = (i - 1) // 2
            if self.comp(x, self.data[p]):
                self.data[i] = self.data[p]
                self.index[self.data[p]] = i
                i = p
            else:
                break
        self.data[i] = x
        self.index[x] = i

    def percolateDown(self, i):
        x = self.data[i]
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            best = i

            if l < len(self.data) and self.comp(self.data[l], self.data[best]):
                best = l
            if r < len(self.data) and self.comp(self.data[r], self.data[best]):
                best = r

            if best == i:
                break
            # Swap with the best child
            self.data[i], self.data[best] = self.data[best], self.data[i]
            self.index[self.data[i]] = i
            self.index[self.data[best]] = best
            i = best
        self.data[i] = x
        self.index[x] = i

    def removeMin(self):
        if self.empty():
            return -1
        root = self.data[0]
        last = self.data.pop()
        self.index[root] = -1
        if not self.empty():
            self.data[0] = last
            self.index[last] = 0
            self.percolateDown(0)
        return root

    def build(self, vs):
        self.data = vs[:]
        self.index = {v: i for i, v in enumerate(self.data)}
        for i in range((len(self.data) // 2) - 1, -1, -1):
            self.percolateDown(i)

    def insert(self, x):
        if self.inHeap(x):
            return
        self.data.append(x)
        i = len(self.data) - 1
        self.index[x] = i
        self.percolateUp(i)

    def decrease(self, x):
        # Called after increasing variable's activity to bubble it up
        i = self.index[x]
        self.percolateUp(i)

    def __getitem__(self, i):
        return self.data[i]


def drand(seed_container):
    """
    seed_container is a one-element list or a mutable object holding the seed,
    because we need to modify it in place just like the C++ reference.
    """
    seed = seed_container[0]
    seed *= 1389796
    q = int(seed / 2147483647)
    seed = seed - (q * 2147483647)
    seed_container[0] = seed
    return seed / 2147483647


def irand(seed_container, size):
    return int(drand(seed_container) * size)


class MinisatSolver:
    def __init__(self):
        _cat = "CORE"
        self.opt_var_decay = DoubleOption(_cat, "var-decay", "", 0.95, (0, 1))
        self.opt_clause_decay = DoubleOption(_cat, "cla-decay", "", 0.999, (0, 1))
        self.opt_random_var_freq = DoubleOption(_cat, "rnd-freq", "", 0, (0, 1))
        self.opt_random_seed = DoubleOption(_cat, "rnd-seed", "", 91648253, (0, 1e100))
        self.opt_ccmin_mode = IntOption(_cat, "ccmin-mode", "", 2, (0, 2))
        self.opt_phase_saving = IntOption(_cat, "phase-saving", "", 2, (0, 2))
        self.opt_rnd_init_act = BoolOption(_cat, "rnd-init", "", False)
        self.opt_luby_restart = BoolOption(_cat, "luby", "", True)
        self.opt_restart_first = IntOption(_cat, "rfirst", "", 100, (1, 2 ** 31))
        self.opt_restart_inc = DoubleOption(_cat, "rinc", "", 2, (1, 1e100))
        self.opt_garbage_frac = DoubleOption(_cat, "gc-frac", "", 0.2, (0, 1e100))

        self.verbosity = 1
        self.var_decay = self.opt_var_decay.value
        self.clause_decay = self.opt_clause_decay.value
        self.random_var_freq = self.opt_random_var_freq.value
        self.random_seed = self.opt_random_seed.value
        self.seed_container = [float(self.random_seed)]
        self.luby_restart = self.opt_luby_restart.value
        self.ccmin_mode = self.opt_ccmin_mode.value
        self.phase_saving = self.opt_phase_saving.value
        self.rnd_pol = False
        self.rnd_init_act = self.opt_rnd_init_act.value
        self.garbage_frac = self.opt_garbage_frac.value
        self.restart_first = self.opt_restart_first.value
        self.restart_inc = self.opt_restart_inc.value

        self.learntsize_factor = 1 / 3
        self.learntsize_inc = 1.1

        self.learntsize_adjust_start_confl = 100
        self.learntsize_adjust_inc = 1.5
        self.lit_Undef = -2
        self.var_Undef = -1

        self.solves = 0
        self.starts = 0
        self.decisions = 0
        self.rnd_decisions = 0
        self.propagations = 0
        self.conflicts = 0
        self.dec_vars = 0
        self.clauses_literals = 0
        self.learnts_literals = 0
        self.max_literals = 0
        self.tot_literals = 0

        self.ok = True
        self.cla_inc = 1
        self.var_inc = 1
        self.ca = ClauseAllocator()
        self.watches = Watches(WatcherDeleted(self.ca))
        self.qhead = 0
        self.simpDB_assigns = -1
        self.simpDB_props = 0
        self.activity = []
        self.seen = []
        self.polarity = []
        self.decision = []
        self.trail = []
        self.trail_lim = []
        self.order_heap = Heap(VarOrderLt(self.activity))
        self.progress_estimate = 0
        self.remove_satisfied = True
        self.conflict_budget = -1
        self.propagation_budget = -1
        self.asynch_interrupt = False
        self.trace_enabled = True
        self.suppress_logging = False
        self.model = []
        self.conflict_clause = []
        self.clauses = []
        self.learnts = []
        self.assumptions = []
        self.vardata = []
        self.assigns = []
        self.reason_ = []
        self.analyze_toclear = []
        self.analyze_stack = []
        self.learntsize_adjust_confl = 0
        self.learntsize_adjust_cnt = 0
        self.max_learnts = 0

        self.trace = ''
        self.record_entire_trace = False

        self.use_keytrace = False
        self.keytrace_instructions = []

        self.record_key_trace = True
        self.key_trace_events = []

        self.pop_decision_level = 0   # decisions + assumptions (reason == UNDEF, level > 0)
        self.pop_root_level     = 0   # root-level units      (reason == UNDEF, level == 0)
        self.pop_implied        = 0   # implications (reason != UNDEF)

        # BCP-only propagations (what you want to report as "propagations"):
        self.propagations_bcp   = 0   # = pop_root_level + pop_implied



    def print_order_heap(self, message: str = "order_heap"):
        print(f"{message}:")
        for i in range(self.order_heap.size()):
            var_idx = self.order_heap[i]
            print(f"Var {var_idx} (activity = {self.activity[var_idx]:.6f})")
        print()  # Add an empty line for better readability

    def nAssigns(self) -> int:
        return len(self.trail)
    
    def nVars(self):
        return len(self.assigns)

    def nClauses(self):
        return len(self.clauses)

    def value_var(self, x: int) -> int:
        return self.assigns[x]

    def value_lit(self, l: int) -> int:
        x = var(l)
        s = sign(l)
        val = self.assigns[x]
        if val == Lbool.UNDEF:
            return Lbool.UNDEF
        return val ^ (1 if s else 0)

    def value(self, p: int) -> int:
        return self.value_lit(p)

    def level(self, x: int) -> int:
        return self.vardata[x].level

    def reason(self, x: int) -> int:
        return self.vardata[x].reason

    def decisionLevel(self):
        return len(self.trail_lim)

    def newDecisionLevel(self):
        self.trail_lim.append(len(self.trail))

    def setDecisionVar(self, v: int, b: bool):
        self.decision[v] = b
        if b and self.value_var(v) == Lbool.UNDEF:
            self.order_heap.insert(v)
            self.dec_vars += 1

    def insertVarOrder(self, x: int):
        if self.value_var(x) == Lbool.UNDEF and self.decision[x]:
            self.order_heap.insert(x)

    def varDecayActivity(self):
        self.var_inc *= (1 / self.var_decay)

    def varBumpActivity(self, v: int):
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e100:
            for i in range(len(self.activity)):
                self.activity[i] *= 1e-100
            self.var_inc *= 1e-100
        if self.order_heap.inHeap(v):
            self.order_heap.decrease(v)

    def claDecayActivity(self):
        self.cla_inc *= (1 / self.clause_decay)

    def claBumpActivity(self, c: Clause):
        c._activity += self.cla_inc
        if c._activity > 1e20:
            for cref in self.learnts:
                cc = self.ca[cref]
                cc._activity *= 1e-20
            self.cla_inc *= 1e-20

    def enableTracing(self, enable: bool):
        self.trace_enabled = enable
        if self.trace_enabled:
            print("Tracing enabled.")

    def newVar(self, sign_: bool, dvar: bool):
        v = self.nVars()
        self.watches.init(mkLit(v, False))
        self.watches.init(mkLit(v, True))
        self.assigns.append(Lbool.UNDEF)
        self.vardata.append(mkVarData(CRef.UNDEF, 0))
        if self.rnd_init_act:
            init_act = drand(self.random_seed) * 0.00001
        else:
            init_act = 0
        self.activity.append(init_act)
        self.seen.append(0)
        self.polarity.append(sign_)
        self.decision.append(False)
        self.setDecisionVar(v, dvar)
        return v

    def addClause_(self, ps: List[int]) -> bool:
        assert self.decisionLevel() == 0
        if not self.ok:
            return False
        ps.sort(key=lambda x: (var(x), sign(x)))
        p = self.lit_Undef
        j = 0
        i = 0
        while i < len(ps):
            if self.value(ps[i]) == Lbool.TRUE or (p != self.lit_Undef and ps[i] == (p ^ 1)):
                return True
            elif self.value(ps[i]) != Lbool.FALSE and (p == self.lit_Undef or ps[i] != p):
                ps[j] = ps[i]
                p = ps[i]
                j += 1
            i += 1
        ps = ps[:j]
        if len(ps) == 0:
            self.ok = False
            return False
        elif len(ps) == 1:
            self.uncheckedEnqueue(ps[0])
            self.ok = (self.propagate() == CRef.UNDEF)
            return self.ok
        else:
            cr = self.ca.alloc(ps, False)
            self.clauses.append(cr)
            self.attachClause(cr)
            return True

    def attachClause(self, cr: int):
        c = self.ca[cr]
        assert c.size() > 1
        self.watches[(c[0] ^ 1)].append(Watcher(cr, c[1]))
        self.watches[(c[1] ^ 1)].append(Watcher(cr, c[0]))
        if c.learnt():
            self.learnts_literals += c.size()
        else:
            self.clauses_literals += c.size()

    def detachClause(self, cr: int, strict: bool):
        c = self.ca[cr]
        assert c.size() > 1
        if strict:
            remove(self.watches[(c[0] ^ 1)], Watcher(cr, c[1]))
            remove(self.watches[(c[1] ^ 1)], Watcher(cr, c[0]))
        else:
            self.watches.smudge(c[0] ^ 1)
            self.watches.smudge(c[1] ^ 1)
        if c.learnt():
            self.learnts_literals -= c.size()
        else:
            self.clauses_literals -= c.size()

    def locked(self, clause: Clause) -> bool:
        if clause.size() == 0:
            return False
        first_lit = clause.lits[0]
        r = self.reason(var(first_lit))
        return (
                self.value(first_lit) == Lbool.TRUE
                and r != CRef.UNDEF
                and r == clause.cref  # Compare cref instead of object identity
        )

    def removeClause(self, cr: int):
        c = self.ca[cr]
        self.detachClause(cr, True)
        if self.locked(c):
            self.vardata[var(c[0])].reason = CRef.UNDEF
        c.mark(1)
        self.ca.free(cr)

    def satisfied(self, c: Clause) -> bool:
        for i in range(c.size()):
            if self.value(c[i]) == Lbool.TRUE:
                return True
        return False

    def cancelUntil(self, level: int):
        if self.decisionLevel() > level:
            for c in range(self.nAssigns() - 1, self.trail_lim[level] - 1, -1):
                x = var(self.trail[c])
                self.assigns[x] = Lbool.UNDEF
                if self.phase_saving > 1 or (self.phase_saving == 1 and c > self.trail_lim[level]):
                    self.polarity[x] = sign(self.trail[c])
                self.insertVarOrder(x)
            self.qhead = self.trail_lim[level]
            self.trail = self.trail[:self.trail_lim[level]]
            self.trail_lim = self.trail_lim[:level]

    def standard_pickBranchLit(self) -> int:
        next_ = self.var_Undef
        if drand(self.seed_container) < self.random_var_freq and not self.order_heap.empty():
            next_ = self.order_heap[irand(self.seed_container, self.order_heap.size())]
            if self.value_var(next_) == Lbool.UNDEF and self.decision[next_]:
                self.rnd_decisions += 1
        while next_ == self.var_Undef or self.value_var(next_) != Lbool.UNDEF or not self.decision[next_]:
            if self.order_heap.empty():
                next_ = self.var_Undef
                break
            else:
                next_ = self.order_heap.removeMin()
        if next_ == self.var_Undef:
            return self.lit_Undef
        return mkLit(next_, self.rnd_pol and drand(self.random_seed) < 0.5 or self.polarity[next_])

    def pickBranchLit(self) -> int:
        if self.use_keytrace:
            next_ = compute_next_decision_value(self.keytrace_instructions.pop())
            self.use_keytrace = False if len(self.keytrace_instructions) == 0 else True
            return next_
        else:
            return self.standard_pickBranchLit()

    def analyze(self, confl: int):
        out_learnt = []
        pathC = 0
        p = self.lit_Undef

        out_learnt.clear()
        out_learnt.append(self.lit_Undef)
        index = self.nAssigns() - 1
        while True:
            assert confl != CRef.UNDEF
            c = self.ca[confl]
            if c.learnt():
                self.claBumpActivity(c)
            for j in range(0 if p == self.lit_Undef else 1, c.size()):
                q = c[j]
                if self.seen[var(q)] == 0 and self.level(var(q)) > 0:
                    self.varBumpActivity(var(q))
                    self.seen[var(q)] = 1
                    if self.level(var(q)) >= self.decisionLevel():
                        pathC += 1
                    else:
                        out_learnt.append(q)
            while self.seen[var(self.trail[index])] == 0:
                index -= 1
            p = self.trail[index]
            confl = self.reason(var(p))
            self.seen[var(p)] = 0
            pathC -= 1
            if pathC <= 0:
                break
        out_learnt[0] = p ^ 1

        self.analyze_toclear.clear()
        self.analyze_toclear.extend(out_learnt)
        if self.ccmin_mode == 2:
            abstract_level = 0
            for i in range(1, len(out_learnt)):
                abstract_level |= (1 << (self.level(var(out_learnt[i])) & 31))
            i = 1
            j = 1
            while i < len(out_learnt):
                x = var(out_learnt[i])
                if self.reason(x) == CRef.UNDEF or not self.litRedundant(out_learnt[i], abstract_level):
                    out_learnt[j] = out_learnt[i]
                    j += 1
                i += 1
            out_learnt[:] = out_learnt[:j]
        elif self.ccmin_mode == 1:
            i = 1
            j = 1
            while i < len(out_learnt):
                x = var(out_learnt[i])
                if self.reason(x) == CRef.UNDEF:
                    out_learnt[j] = out_learnt[i]
                    j += 1
                else:
                    rc = self.ca[self.reason(x)]
                    k = 1
                    keep = False
                    while k < rc.size():
                        if self.seen[var(rc[k])] == 0 and self.level(var(rc[k])) > 0:
                            keep = True
                            break
                        k += 1
                    if keep:
                        out_learnt[j] = out_learnt[i]
                        j += 1
                i += 1
            out_learnt[:] = out_learnt[:j]
        else:
            pass
        self.max_literals += len(out_learnt)
        self.tot_literals += len(out_learnt)

        if len(out_learnt) == 1:
            out_btlevel = 0
        else:
            max_i = 1
            for i in range(2, len(out_learnt)):
                if self.level(var(out_learnt[i])) > self.level(var(out_learnt[max_i])):
                    max_i = i
            p = out_learnt[max_i]
            out_learnt[max_i] = out_learnt[1]
            out_learnt[1] = p
            out_btlevel = self.level(var(p))

        for lit_ in self.analyze_toclear:
            self.seen[var(lit_)] = 0
        return out_learnt, out_btlevel

    def litRedundant(self, p: int, abstract_levels: int) -> bool:
        self.analyze_stack.clear()
        self.analyze_stack.append(p)
        top = len(self.analyze_toclear)
        while len(self.analyze_stack) > 0:
            q = self.analyze_stack.pop()
            r = self.reason(var(q))
            assert r != CRef.UNDEF
            c = self.ca[r]
            for i in range(1, c.size()):
                pp = c[i]
                if self.seen[var(pp)] == 0 and self.level(var(pp)) > 0:
                    if self.reason(var(pp)) != CRef.UNDEF and (
                            (1 << (self.level(var(pp)) & 31)) & abstract_levels) != 0:
                        self.seen[var(pp)] = 1
                        self.analyze_stack.append(pp)
                        self.analyze_toclear.append(pp)
                    else:
                        for j in range(top, len(self.analyze_toclear)):
                            self.seen[var(self.analyze_toclear[j])] = 0
                        self.analyze_toclear = self.analyze_toclear[:top]
                        return False
        return True

    def analyzeFinal(self, p: int, out_conflict: List[int]):
        out_conflict.clear()
        out_conflict.append(p)
        if self.decisionLevel() == 0:
            return
        self.seen[var(p)] = 1
        for i in range(self.nAssigns() - 1, self.trail_lim[0] - 1, -1):
            x = var(self.trail[i])
            if self.seen[x]:
                if self.reason(x) == CRef.UNDEF:
                    out_conflict.append(self.trail[i] ^ 1)
                else:
                    cc = self.ca[self.reason(x)]
                    for j in range(1, cc.size()):
                        if self.level(var(cc[j])) > 0:
                            self.seen[var(cc[j])] = 1
                self.seen[x] = 0
        self.seen[var(p)] = 0

    def uncheckedEnqueue(self, p: int, from_=CRef.UNDEF):
        assert self.value(p) == Lbool.UNDEF
        self.assigns[var(p)] = Lbool.TRUE if not sign(p) else Lbool.FALSE
        self.vardata[var(p)] = VarData(from_, self.decisionLevel())
        self.trail.append(p)
        if self.trace_enabled and not self.suppress_logging:
            var_num = var(p) + 1
            val = -var_num if sign(p) else var_num
            if from_ == CRef.UNDEF and self.decisionLevel() > 0:
                level = self.decisionLevel()
                if self.record_entire_trace:
                    self.trace += f'D {val} L {level} '
                if self.record_key_trace:
                    self.key_trace_events.append(("D", val, level))
                if self.verbosity >= 1:
                    print("D {} L {} ".format(val, level), end='')
            else:
                if self.record_entire_trace:
                    self.trace += f"A {val} "
                if self.record_key_trace:
                    self.key_trace_events.append(("A", val, self.decisionLevel()))
                if self.verbosity >= 1:
                    print("A {} ".format(val), end='')

    def propagate(self) -> int:
        confl = CRef.UNDEF
        num_props = 0
        self.watches.cleanAll()
        while self.qhead < self.nAssigns():
            p = self.trail[self.qhead]
            self.qhead += 1

            v = var(p)
            r = self.vardata[v].reason
            lvl = self.vardata[v].level

            if r == CRef.UNDEF:
                if lvl == 0:
                    self.pop_root_level += 1
                    self.propagations_bcp += 1  # root-level unit is a propagation
                else:
                    self.pop_decision_level += 1  # decision or assumption (not a propagation)
            else:
                self.pop_implied += 1
                self.propagations_bcp += 1  # implied assignment is a propagation

            ws = self.watches[p]
            i = 0
            j = 0
            end = len(ws)
            num_props += 1
            while i < end:
                w = ws[i]
                i += 1
                blocker = w.blocker
                if self.value(blocker) == Lbool.TRUE:
                    ws[j] = w
                    j += 1
                    continue
                cr = w.cref
                c = self.ca[cr]
                if c is None:
                    print(f"Error: cref {cr} not found in clause allocator!")
                false_lit = p ^ 1
                if c[0] == false_lit:
                    c[0], c[1] = c[1], c[0]

                assert c[1] == false_lit
                w2 = Watcher(cr, c[0])
                if self.value(c[0]) == Lbool.TRUE:
                    ws[j] = w2
                    j += 1
                    continue
                k = 2
                foundWatch = False
                while k < c.size():
                    if self.value(c[k]) != Lbool.FALSE:
                        c[1] = c[k]
                        c[k] = false_lit
                        self.watches[c[1] ^ 1].append(w2)
                        foundWatch = True
                        break
                    k += 1
                if not foundWatch:
                    ws[j] = w2
                    j += 1
                    if self.value(c[0]) == Lbool.FALSE:
                        confl = cr
                        self.qhead = self.nAssigns()
                        while i < end:
                            ws[j] = ws[i]
                            j += 1
                            i += 1
                    else:
                        self.uncheckedEnqueue(c[0], cr)
            ws[:] = ws[:j]
        self.propagations += num_props
        self.simpDB_props -= num_props
        return confl

    def reduceDB(self):
        if len(self.learnts) == 0:
            return

        extra_lim = self.cla_inc / len(self.learnts)

        self.learnts.sort(key=lambda cr: (self.ca[cr].size() == 2, self.ca[cr].activity()), reverse=False)

        i = 0
        j = 0
        half = len(self.learnts) // 2
        while i < len(self.learnts):
            c = self.ca[self.learnts[i]]
            if c.size() > 2 and not self.locked(c) and (i < half or c.activity() < extra_lim):
                self.removeClause(self.learnts[i])
            else:
                self.learnts[j] = self.learnts[i]
                j += 1
            i += 1

        self.learnts[j:] = []
        self.checkGarbage()

    def removeSatisfied(self, cs: List[int]):
        i = 0
        j = 0
        while i < len(cs):
            c = self.ca[cs[i]]
            if c is not None and self.satisfied(c):
                self.removeClause(cs[i])
            else:
                if c is not None:
                    cs[j] = cs[i]
                    j += 1
            i += 1
        cs[:] = cs[:j]

    def rebuildOrderHeap(self):
        vs = []
        for v in range(self.nVars()):
            if self.decision[v] and self.value_var(v) == Lbool.UNDEF:
                vs.append(v)
        self.order_heap.build(vs)

    def simplify(self) -> bool:
        assert self.decisionLevel() == 0
        if not self.ok or self.propagate() != CRef.UNDEF:
            self.ok = False
            return False
        if self.nAssigns() == self.simpDB_assigns or self.simpDB_props > 0:
            return True
        self.removeSatisfied(self.learnts)
        if self.remove_satisfied:
            self.removeSatisfied(self.clauses)

        self.checkGarbage()
        self.rebuildOrderHeap()
        self.simpDB_assigns = self.nAssigns()
        self.simpDB_props = self.clauses_literals + self.learnts_literals
        return True

    def search(self, nof_conflicts: int):
        assert self.ok
        backtrack_level = 0
        conflictC = 0
        learnt_clause = []
        self.starts += 1
        while True:
            confl = self.propagate()
            if confl != CRef.UNDEF:
                self.conflicts += 1
                conflictC += 1
                if self.decisionLevel() == 0:
                    return Lbool.FALSE

                # self.print_order_heap("order heap before BT analyze")

                learnt_clause.clear()
                learnt_clause, backtrack_level = self.analyze(confl)

                self.cancelUntil(backtrack_level)
                if self.trace_enabled:
                    var_num = var(learnt_clause[0]) + 1
                    val = -var_num if sign(learnt_clause[0]) else var_num
                    level = backtrack_level
                    if self.record_entire_trace:
                        self.trace += f'BT {val} L {level} '
                    if self.record_key_trace:
                        self.key_trace_events = [
                            (etype, eval, elvl)
                            for (etype, eval, elvl) in self.key_trace_events
                            if elvl <= backtrack_level
                        ]
                        self.key_trace_events.append(("BT", val, backtrack_level))
                    if self.verbosity >= 1:
                        print("BT {} L {}".format(val, level), end='')
                    self.suppress_logging = True
                if len(learnt_clause) == 1:
                    self.uncheckedEnqueue(learnt_clause[0])
                else:
                    cr = self.ca.alloc(learnt_clause, True)
                    self.learnts.append(cr)
                    self.attachClause(cr)
                    self.claBumpActivity(self.ca[cr])
                    self.uncheckedEnqueue(learnt_clause[0], cr)
                self.suppress_logging = False
                self.varDecayActivity()
                self.claDecayActivity()

                # self.print_order_heap("order heap after BT")

                self.learntsize_adjust_cnt -= 1
                if self.learntsize_adjust_cnt == 0:
                    self.learntsize_adjust_confl *= self.learntsize_adjust_inc
                    self.learntsize_adjust_cnt = int(self.learntsize_adjust_confl)
                    self.max_learnts *= self.learntsize_inc
                    if self.verbosity >= 2:
                        print("| %9d | %7d %8d %8d | %8d %8d %6.0f | %6.3f %% |" % (
                            self.conflicts,
                            self.dec_vars - (len(self.trail_lim) == 0 and self.nAssigns() or self.trail_lim[0]),
                            self.nClauses(), self.clauses_literals,
                            int(self.max_learnts), len(self.learnts),
                            (self.learnts_literals / (len(self.learnts) + 1e-100)), self.progressEstimate() * 100
                        ))
            else:
                if (nof_conflicts >= 0 and conflictC >= nof_conflicts) or not self.withinBudget():
                    self.progress_estimate = self.progressEstimate()
                    if self.record_entire_trace:
                        self.trace += f'L 0 '
                    if self.record_key_trace:
                        self.key_trace_events = []
                    if self.verbosity >= 1:
                        print("L 0 ", end='')
                    self.cancelUntil(0)
                    return Lbool.UNDEF
                if self.decisionLevel() == 0 and not self.simplify():
                    return Lbool.FALSE
                if len(self.learnts) - self.nAssigns() >= self.max_learnts:
                    # pass
                    self.reduceDB()
                next_ = self.lit_Undef
                while self.decisionLevel() < len(self.assumptions):
                    p = self.assumptions[self.decisionLevel()]
                    if self.value(p) == Lbool.TRUE:
                        self.newDecisionLevel()
                    elif self.value(p) == Lbool.FALSE:
                        self.analyzeFinal(p ^ 1, self.conflict_clause)
                        return Lbool.FALSE
                    else:
                        next_ = p
                        break
                if next_ == self.lit_Undef:
                    self.decisions += 1
                    next_ = self.pickBranchLit()
                    if next_ == self.lit_Undef:
                        return Lbool.TRUE
                self.newDecisionLevel()
                self.uncheckedEnqueue(next_)

    def progressEstimate(self) -> float:
        progress = 0.0
        F = 1.0 / self.nVars() if self.nVars() > 0 else 1.0
        for i in range(self.decisionLevel() + 1):
            beg = 0 if i == 0 else self.trail_lim[i - 1]
            end = self.trail_lim[i] if i < self.decisionLevel() else self.nAssigns()
            progress += math.pow(F, i) * (end - beg)
        return progress / self.nVars() if self.nVars() > 0 else 0.0

    def withinBudget(self) -> bool:
        if self.conflict_budget >= 0 and self.conflicts >= self.conflict_budget:
            return False
        if self.propagation_budget >= 0 and self.propagations >= self.propagation_budget:
            return False
        return True

    def solve_(self):
        self.model.clear()
        self.conflict_clause.clear()
        if not self.ok:
            return Lbool.FALSE
        self.solves += 1
        self.max_learnts = self.nClauses() * self.learntsize_factor
        self.learntsize_adjust_confl = self.learntsize_adjust_start_confl
        self.learntsize_adjust_cnt = int(self.learntsize_adjust_confl)
        status = Lbool.UNDEF
        if self.verbosity >= 1:
            print("============================[ Search Statistics ]==============================")
            print("| Conflicts |          ORIGINAL         |          LEARNT          | Progress |")
            print("|           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |")
            print("===============================================================================")
        curr_restarts = 0
        while status == Lbool.UNDEF:
            rest_base = self.luby(self.restart_inc, curr_restarts) if self.luby_restart else math.pow(self.restart_inc,
                                                                                                      curr_restarts)
            status = self.search(int(rest_base * self.restart_first))
            if not self.withinBudget():
                break
            curr_restarts += 1
        if self.verbosity >= 1:
            print("\n===============================================================================")
        if status == Lbool.TRUE:
            self.model = [self.value_var(i) for i in range(self.nVars())]
        elif status == Lbool.FALSE and len(self.conflict_clause) == 0:
            self.ok = False
        self.cancelUntil(0)
        return status

    def luby(self, y: float, x: int) -> float:
        size = 1
        seq = 0
        while size < x + 1:
            seq += 1
            size = 2 * size + 1
        while size - 1 != x:
            size = (size - 1) >> 1
            seq -= 1
            x = x % size
        return math.pow(y, seq)

    def toDimacs(self, f, assumps: List[int]):
        if not self.ok:
            f.write("p cnf 1 2\n1 0\n-1 0\n")
            return
        map_ = []
        for _ in range(self.nVars()):
            map_.append(-1)
        max_ = 0
        cnt = 0
        for i in range(len(self.clauses)):
            c = self.ca[self.clauses[i]]
            if c is not None and not self.satisfied(c):
                cnt += 1
        for i in range(len(self.clauses)):
            c = self.ca[self.clauses[i]]
            if c is not None and not self.satisfied(c):
                for lit_ in c.lits:
                    if self.value(lit_) != Lbool.FALSE:
                        v = var(lit_)
                        if v >= len(map_) or map_[v] == -1:
                            map_[v] = max_
                            max_ += 1
        cnt += len(assumps)
        f.write("p cnf {} {}\n".format(max_, cnt))
        for a in assumps:
            assert self.value(a) != Lbool.FALSE
            v = var(a)
            if v >= len(map_) or map_[v] == -1:
                map_[v] = max_
                max_ += 1
            f.write("{}{} 0\n".format('-' if sign(a) else '', map_[v] + 1))
        for i in range(len(self.clauses)):
            c = self.ca[self.clauses[i]]
            if c is None or self.satisfied(c):
                continue
            for lit_ in c.lits:
                if self.value(lit_) != Lbool.FALSE:
                    v = var(lit_)
                    if v >= len(map_) or map_[v] == -1:
                        map_[v] = max_
                        max_ += 1
                    f.write("{}{} ".format('-' if sign(lit_) else '', map_[v] + 1))
            f.write("0\n")
        if self.verbosity > 0:
            print("Wrote {} clauses with {} variables.".format(cnt, max_))

    def relocAll(self, to: ClauseAllocator):
        self.watches.cleanAll()
        for v in range(self.nVars()):
            for s in [False, True]:
                p = mkLit(v, s)
                ws = self.watches[p]
                for j in range(len(ws)):
                    new_cref  = self.ca.reloc(ws[j].cref, to)
                    ws[j].cref = new_cref

        for i in range(len(self.trail)):
            v = var(self.trail[i])
            r = self.vardata[v].reason  # reason
            if r != CRef.UNDEF and (self.ca[r].reloced() or self.locked(self.ca[r])):
                new_cref = self.ca.reloc(r, to)
                self.vardata[v].reason = new_cref

        # Relocate all learnt clauses:
        for i in range(len(self.learnts)):
            self.learnts[i] = self.ca.reloc(self.learnts[i], to)

        # Relocate all original clauses:
        for i in range(len(self.clauses)):
            self.clauses[i] = self.ca.reloc(self.clauses[i], to)

    def checkGarbage(self):
        if self.ca.wasted() > self.ca.size() * self.garbage_frac:
            self.garbageCollect()

    def checkClauseInvariant(self, cr):
        c = self.ca[cr]
        if c.size() > 1:
            w0 = self.watches[(c[0] ^ 1)]
            w1 = self.watches[(c[1] ^ 1)]
            assert any(w.cref == cr for w in w0), f"Clause {cr} not watched by c[0]^1"
            assert any(w.cref == cr for w in w1), f"Clause {cr} not watched by c[1]^1"

    def garbageCollect(self):
        needed_cap = self.ca.size() - self.ca.wasted()
        to = ClauseAllocator(needed_cap)

        self.relocAll(to)
        if self.verbosity >= 2:
            print("|  Garbage collection:   %12d bytes => %12d bytes             |" % (
                self.ca.size() * ClauseAllocator.Unit_Size, to.size() * ClauseAllocator.Unit_Size))
        to.moveTo(self.ca)
