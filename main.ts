// Push-optimal Sokoban solver for make.fun's sokoban-solver-01-run-20261121 (MIT).
//
// One program answers a set of 5 Sokoban levels (level.txt, XSB) in a single run.
// Score is all about pushes: an instance earns 1,000 points only at the push-optimal
// count P*, and 500 x P* / pushes above it, so the search is over pushes, not moves.
//
// Method (my own implementation of the standard approach):
// - The generator builds each level as a hallway plus 1-4 independent rooms (a box
//   pushed into the hallway can never come back), so P* is the sum of the rooms'
//   optima and each room is solved on its own, the other rooms' boxes as walls.
// - Each room is solved with A* over pushes: a state is the sorted box squares plus
//   the player's reachable region (named by its lowest square). The heuristic is the
//   minimum assignment of boxes to goals over grid distances (admissible, and
//   consistent since one push moves one box one square), so the first solved state
//   popped is push-optimal.
// - Deadlock pruning: dead squares (a box there can never reach a goal, found by
//   pulling back from the goals) and frozen 2x2 blocks of walls and boxes.
// - Budgets are by work, never by the clock: a fixed expansion cap per room, so two
//   executions write the same bytes. No Date, no randomness, no environment input.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Max push-search expansions per room: work, never the clock, so two runs agree byte for byte. */
const MAX_EXPANSIONS_PER_ROOM = 300_000;
/** Cap for the whole-level fallback (rooms not decomposable); its states cost more. */
const MAX_EXPANSIONS_WHOLE = 150_000;

/** [dx, dy, step letter, push letter]; this fixed order makes every tie deterministic. */
const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  size: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player,
 * + player on a goal), one row a line; short rows and the outer frame count as walls.
 */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, size: w * h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(level.size);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** The squares the player reaches from `from` with `occupied` (a box mask) in the way, marked in `seen` with `stamp`; the lowest. */
function reach(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(level.size).fill(-1);
  const how = new Array<string>(level.size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, the case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box
 * there reaches no goal) out of the floor, and each part left that holds a box is a
 * room. Null when some part holds more or fewer boxes than goals, or a box starts in
 * the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(level.size);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(level.size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < level.size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The push-optimal solution of `level` (A* over pushes), searching at most
 * `maxExp` expansions: {pushes, moves}, or null when the search gives up.
 * With the admissible, consistent assignment heuristic the first solved state
 * popped is push-optimal.
 */
function solvePushAStar(level: Level, maxExp: number): { pushes: number; moves: string } | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = level.size;
  const nB = startBoxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (startBoxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => {
    for (const b of bs) if (!goal[b]) return false;
    return true;
  };
  if (nB === 0) return { pushes: 0, moves: '' };
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // Grid distance from each goal over open floor (boxes ignored).
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    d[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const nq = p + dy * w + dx;
        if (wall[nq] || d[nq] !== -1) continue;
        d[nq] = d[p]! + 1;
        q.push(nq);
      }
    }
    return d;
  });

  // Lower bound on pushes left: the cheapest box-to-goal assignment (exact search
  // over permutations while the room is small, else the sum of nearest goals).
  // Either way admissible: one push moves one box one square.
  const heuristic = (bs: number[]): number => {
    if (nB <= 5) {
      let best = Infinity;
      const used = new Uint8Array(goals.length);
      const rec = (i: number, acc: number): void => {
        if (acc >= best) return;
        if (i === nB) {
          best = acc;
          return;
        }
        for (let j = 0; j < goals.length; j++) {
          if (used[j]) continue;
          const d = dist[j]![bs[i]!]!;
          if (d < 0) continue;
          used[j] = 1;
          rec(i + 1, acc + d);
          used[j] = 0;
        }
      };
      rec(0, 0);
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (let j = 0; j < goals.length; j++) {
        const d = dist[j]![b]!;
        if (d >= 0 && d < m) m = d;
      }
      if (!Number.isFinite(m)) return Infinity;
      sum += m;
    }
    return sum;
  };

  const h0 = heuristic(startBoxes);
  if (!Number.isFinite(h0)) return null;

  // A state's key: the region's lowest square and the boxes' squares. Numeric
  // while the radix product stays exact (rooms: at most 5 squares), else text.
  const keyOf = (bs: number[], low: number): number | string => {
    if (nB <= 5) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return `${low}|${bs.join(',')}`;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The states in the order found: their boxes, the player's square, the push
  // that made them, and their key (for the stale-entry check).
  const sBoxes: number[][] = [startBoxes.slice()];
  const sPlayer: number[] = [startPlayer];
  const sParent: number[] = [-1];
  const sPushFrom: number[] = [-1];
  const sPushTo: number[] = [-1];
  const sKey: (number | string)[] = [];
  const best = new Map<number | string, number>();
  for (const b of startBoxes) occupied[b] = 1;
  const k0 = keyOf(startBoxes, reach(level, occupied, startPlayer, seen, ++stamp, stack));
  for (const b of startBoxes) occupied[b] = 0;
  sKey.push(k0);
  best.set(k0, 0);

  // Open list: a heap on [f, g, seq, state]; f first, deeper first on a tie,
  // insertion order last, so the run is deterministic.
  type Entry = [number, number, number, number];
  const heap: Entry[] = [];
  const less = (a: Entry, b: Entry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: Entry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): Entry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let seq = 1;
  heapPush([h0, 0, 0, 0]);
  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    if (g > best.get(sKey[s]!)!) continue; // stale: a better path to this state was found
    const bs = sBoxes[s]!;
    if (solved(bs)) {
      found = s;
      break;
    }
    if (expanded >= maxExp) return null;
    expanded++;
    for (const b of bs) occupied[b] = 1;
    reach(level, occupied, sPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < nB; bi++) {
      const b = bs[bi]!;
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let nk: number | string = -1;
        let nb: number[] | null = null;
        let nh = 0;
        if (!frozen) {
          nb = bs.slice();
          nb[bi] = t;
          nb.sort((x, y) => x - y);
          nk = keyOf(nb, reach(level, occupied, b, mark, ++stamp, stack));
          const prevG = best.get(nk);
          if (prevG !== undefined && prevG <= g + 1) {
            nb = null;
          } else {
            nh = heuristic(nb);
            if (!Number.isFinite(nh)) nb = null;
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || nb === null) continue;
        best.set(nk, g + 1);
        const ni = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sParent.push(s);
        sPushFrom.push(b);
        sPushTo.push(t);
        sKey.push(nk);
        heapPush([g + 1 + nh, g + 1, seq++, ni]);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of pushes, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes.slice();
  for (const s of path) {
    const from = sPushFrom[s]!;
    const to = sPushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = sBoxes[s]!.slice();
  }
  return { pushes: path.length, moves };
}

/**
 * The level solved room by room: each room's push-optimal solution in turn, the
 * other rooms' boxes standing as walls (on their goals once their room is done),
 * the player walking on from where the last room left it. A box never leaves its
 * room, so the rooms' optima add up to the level's. Falls back to a whole-level
 * search when the level is no set of rooms.
 */
function solveByRooms(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushAStar(level, MAX_EXPANSIONS_WHOLE);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      size: level.size,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushAStar(sub, MAX_EXPANSIONS_PER_ROOM);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** The moves for a level in XSB: '' when the search gives up (a valid, scoreless answer). */
function solve(text: string): string {
  try {
    const level = parseLevel(text);
    const solved = solveByRooms(level);
    return solved === null ? '' : solved.moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
