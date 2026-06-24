// ─── Seeded RNG & Embeddings ───
function hash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) { h = ((h << 5) - h) + s.charCodeAt(i); h |= 0 }
  return Math.abs(h);
}
function mulberry32(a) {
  return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
}
function seededEmb(seedStr, dim = 32) {
  const r = mulberry32(hash(seedStr));
  const v = new Float64Array(dim); let m = 0;
  for (let i = 0; i < dim; i++) { v[i] = r() * 2 - 1; m += v[i] * v[i] }
  m = Math.sqrt(m);
  for (let i = 0; i < dim; i++) v[i] /= m;
  return v;
}
function sim(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }

// ─── HyperbolicMemoryTree ───
let _nextId = 1;
class Node {
  constructor(depth) { this.id = _nextId++; this.depth = depth; this.children = []; this.facts = []; this.key = null }
}
class HyperbolicMemoryTree {
  constructor(branching = 4, dim = 32) {
    this.branching = branching; this.dim = dim; this.root = null; this.size = 0; this.factCount = 0;
  }

  remember(emb, content, topic) {
    if (!this.root) { this.root = new Node(0); this.size = 1 }
    let leaf = this._findLeaf(this.root, emb);
    leaf.facts.push({ emb, content, topic });
    this.factCount++;
    if (leaf.facts.length > this.branching) this._split(leaf);
  }

  _findLeaf(node, emb) {
    if (node.children.length === 0) return node;
    let best = 0, bestSim = -Infinity;
    for (let i = 0; i < node.children.length; i++) {
      const c = node.children[i];
      const s = c.key ? sim(emb, c.key) : 0;
      if (s > bestSim) { bestSim = s; best = i }
    }
    return this._findLeaf(node.children[best], emb);
  }

  _split(node) {
    const facts = node.facts; if (facts.length <= 1) return;
    const anchor = facts[0].emb;
    const groups = [[], []];
    for (const f of facts) { const s = sim(anchor, f.emb); groups[s >= 0 ? 0 : 1].push(f) }
    if (groups[0].length === 0 || groups[1].length === 0) { for (let i = 0; i < facts.length; i++) groups[i % 2].push(facts[i]) }
    node.facts = [];
    for (const g of groups) {
      if (g.length === 0) continue;
      const n = new Node(node.depth + 1);
      n.facts = g;
      n.key = new Float64Array(this.dim);
      for (const f of g) for (let i = 0; i < this.dim; i++) n.key[i] += f.emb[i];
      for (let i = 0; i < this.dim; i++) n.key[i] /= g.length;
      node.children.push(n); this.size++;
    }
  }

  recall(emb, topK = 5) {
    const results = [];
    this._collect(this.root, emb, results);
    results.sort((a, b) => b.sim - a.sim);
    return results.slice(0, topK);
  }

  _collect(node, emb, results) {
    if (!node) return;
    if (node.children.length === 0) {
      for (const f of node.facts) results.push({ content: f.content, topic: f.topic, sim: sim(emb, f.emb), depth: node.depth, id: node.id });
      return;
    }
    const scored = node.children.map((c, i) => ({ idx: i, sim: c.key ? sim(emb, c.key) : 0 }));
    scored.sort((a, b) => b.sim - a.sim);
    for (const s of scored.slice(0, 2)) this._collect(node.children[s.idx], emb, results);
  }

  getStats() {
    if (!this.root) return { maxDepth: 0, leafCount: 0, facts: 0, nodes: 0 };
    const s = this._dfsStats(this.root, 0);
    return { maxDepth: s.maxDepth, leafCount: s.leafCount, facts: this.factCount, nodes: this.size };
  }

  _dfsStats(node, d) {
    if (!node) return { maxDepth: 0, leafCount: 0 };
    if (node.children.length === 0) return { maxDepth: d, leafCount: 1 };
    let md = 0, lc = 0;
    for (const c of node.children) { const s = this._dfsStats(c, d + 1); md = Math.max(md, s.maxDepth); lc += s.leafCount }
    return { maxDepth: md, leafCount: lc };
  }

  reset() { this.root = null; this.size = 0; this.factCount = 0; _nextId = 1 }

  toJSON() {
    const fn = (n) => {
      if (!n) return null;
      return {
        id: n.id, depth: n.depth, key: n.key ? Array.from(n.key) : null,
        children: n.children.map(fn),
        facts: n.facts.map(f => ({ content: f.content, topic: f.topic, emb: Array.from(f.emb) }))
      };
    };
    return { branching: this.branching, dim: this.dim, root: fn(this.root), size: this.size, factCount: this.factCount, _nextId: _nextId };
  }

  static fromJSON(data) {
    _nextId = data._nextId || 1;
    const fn = (d) => {
      if (!d) return null;
      const n = new Node(d.depth);
      n.id = d.id; n.key = d.key ? Float64Array.from(d.key) : null;
      n.children = d.children.map(fn);
      n.facts = d.facts.map(f => ({ content: f.content, topic: f.topic, emb: Float64Array.from(f.emb) }));
      return n;
    };
    const t = new HyperbolicMemoryTree(data.branching, data.dim);
    t.root = fn(data.root); t.size = data.size; t.factCount = data.factCount;
    return t;
  }

  getAllFacts() {
    const facts = [];
    function collect(n) {
      if (!n) return;
      if (n.children.length === 0) { for (const f of n.facts) facts.push(f); return }
      for (const c of n.children) collect(c);
    }
    collect(this.root);
    return facts;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { HyperbolicMemoryTree, seededEmb, sim };
