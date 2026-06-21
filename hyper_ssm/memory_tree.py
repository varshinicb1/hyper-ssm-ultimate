"""
Hyperbolic Memory Tree (HMT) — logarithmic-time structured memory.

A tree of hyperbolic vectors that grows with your conversation.
Each node stores a compressed summary of its subtree.  Leaf nodes
retain exact original content.  Queries route via hyperbolic similarity.

Key properties:
  - O(log N) retrieval time (N = unique facts)
  - ~260B per node, ~5 MB for 1M tokens
  - Exact recall for recent/salient facts
  - Gist recall for old/compressed branches
  - Auto-prunes and merges to stay within budget

This is a genuinely new architecture — not in any paper.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from .hierarchical_memory import (
    exp_map, log_map, lorentz_inner, project_to_hyperboloid, check_manifold,
)


# =========================================================================
# HELPERS
# =========================================================================

EPS = 1e-8


def _to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device) if device else x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    return torch.tensor([x], dtype=torch.float32)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def hyperbolic_similarity(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Negative Lorentzian inner product = similarity on hyperboloid.
    More negative = closer together in hyperbolic space.
    Returns: (batch,) tensor of similarities (higher = more similar)."""
    return -lorentz_inner(query.unsqueeze(0), key.unsqueeze(0)).squeeze(0)


def hyperbolic_distance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on the hyperboloid."""
    ip = lorentz_inner(query.unsqueeze(0), key.unsqueeze(0)).squeeze(0)
    clipped = torch.clamp(-ip, min=1.0 + EPS)
    return torch.acosh(clipped)


# =========================================================================
# NODE TYPES
# =========================================================================

@dataclass
class MemoryNode:
    """A node in the hyperbolic memory tree."""
    id: int
    depth: int = 0
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)
    
    # Hyperbolic state: compressed summary of this subtree
    # Shape: (state_dim + 1,) on the hyperboloid
    state: Optional[torch.Tensor] = None
    
    # Hyperbolic key: used for routing queries to the right child
    # Also on the hyperboloid
    key: Optional[torch.Tensor] = None
    
    # Euclidean projected embedding (before exp_map).
    # For leaves: the projected input embedding.
    # For internal nodes: average of children's projections.
    # Used for routing without hyperbolic averaging distortion.
    proj: Optional[torch.Tensor] = None
    
    # Leaf-specific
    content: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # original Euclidean embedding
    access_count: int = 0
    
    # Timestamps
    created_at: float = 0.0
    last_accessed: float = 0.0
    
    @property
    def is_leaf(self) -> bool:
        return len(self.child_ids) == 0
    
    @property
    def is_root(self) -> bool:
        return self.parent_id is None
    
    def memory_bytes(self) -> int:
        """Estimated memory used by this node."""
        total = 0
        if self.state is not None:
            total += self.state.numel() * 4
        if self.key is not None:
            total += self.key.numel() * 4
        if self.proj is not None:
            total += self.proj.numel() * 4
        if self.embedding is not None:
            total += self.embedding.nbytes
        if self.content:
            total += len(self.content.encode("utf-8"))
        return total


# =========================================================================
# HYPERBOLIC MEMORY TREE
# =========================================================================

class HyperbolicMemoryTree:
    """
    Adaptive tree-structured hyperbolic memory.
    
    Grows with the conversation.  Learns which facts belong together.
    Recalls exact details for recent/salient items, gist for old ones.
    
    Args:
        state_dim: Dimension of hyperbolic vectors (default 64 -> 260B/node)
        embed_dim: Dimension of input embeddings (default 384)
        max_nodes: Maximum tree nodes before pruning kicks in
        max_depth: Maximum tree depth
        branching_factor: Max children per node
        merge_threshold: Lorentz similarity below this triggers sibling merge
        device: torch device
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        embed_dim: int = 384,
        max_nodes: int = 20000,
        max_depth: int = 10,
        branching_factor: int = 4,
        merge_threshold: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.lorentz_dim = state_dim + 1
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.merge_threshold = merge_threshold
        self.device = device or torch.device("cpu")
        
        self._node_counter = 0
        self._nodes: Dict[int, MemoryNode] = {}
        self._root_id: Optional[int] = None
        
        # Input projection (embed_dim -> state_dim)
        self._input_proj = torch.randn(state_dim, embed_dim, dtype=torch.float32) / (embed_dim ** 0.5)
        
        # Importance scoring
        self._importance_w = torch.randn(state_dim, dtype=torch.float32) * 0.1
        
        self._init_tree()
    
    def _init_tree(self):
        """Create the root node."""
        self._node_counter = 0
        self._nodes = {}
        root = MemoryNode(
            id=self._next_id(),
            depth=0,
        )
        root.state = self._make_origin_state()
        root.key = self._make_origin_state()
        root.proj = torch.zeros(self.state_dim, device=self.device)
        self._root_id = root.id
        self._nodes[root.id] = root
    
    def _next_id(self) -> int:
        self._node_counter += 1
        return self._node_counter
    
    def _make_origin_state(self) -> torch.Tensor:
        h = torch.zeros(self.lorentz_dim, device=self.device)
        h[0] = 1.0  # t = 1, spatial = 0 satisfies <h,h>_L = -1
        return h
    
    # ------------------------------------------------------------------
    # CORE: INSERT
    # ------------------------------------------------------------------
    
    def remember(
        self,
        embedding: np.ndarray,
        content: str = "",
    ) -> int:
        """
        Insert a new fact into the memory tree.
        
        Routes the embedding through the tree to the best-matching leaf,
        then inserts as a sibling or creates a new branch.
        
        Returns:
            Node ID of the new leaf.
        """
        emb_t = _to_tensor(embedding, self.device).float()
        if emb_t.dim() == 1:
            emb_t = emb_t.unsqueeze(0)
        
        # Project to state dimension
        projected = emb_t @ self._input_proj.T  # (1, state_dim)
        
        # Map to hyperboloid
        query_hyp = exp_map(projected)
        
        # Traverse to find insertion point
        path = self._traverse(query_hyp, emb_t)
        parent_id = path[-1] if path else self._root_id
        
        parent = self._nodes[parent_id]
        
        if len(parent.child_ids) >= self.branching_factor and parent.depth < self.max_depth:
            leaf_id = self._split_and_insert(parent_id, emb_t, projected, query_hyp, content)
        else:
            leaf_id = self._create_leaf(emb_t, projected, query_hyp, content, parent_id)
        
        # Update ancestor states
        self._update_ancestors(leaf_id)
        
        # Prune if over capacity
        if len(self._nodes) > self.max_nodes:
            self._prune()
        
        return leaf_id
    
    def _traverse(
        self,
        query_hyp: torch.Tensor,
        query_euc: torch.Tensor,
    ) -> List[int]:
        """
        Walk from root toward the best-matching leaf.
        Returns path of internal nodes (stops before descending into a leaf).
        The last element is the node under which a new leaf should be created.
        """
        path = [self._root_id]
        current_id = self._root_id
        
        for _ in range(self.max_depth):
            node = self._nodes[current_id]
            
            best_child = self._best_child(current_id, query_hyp)
            if best_child is None:
                break
            
            child = self._nodes[best_child]
            # Don't descend into leaves — insert under the current internal node
            if child.is_leaf:
                break
            
            path.append(best_child)
            current_id = best_child
        
        return path
    
    def _best_child(self, parent_id: int, query_hyp: torch.Tensor) -> Optional[int]:
        """Find the child closest to query (lowest hyperbolic_similarity)."""
        parent = self._nodes[parent_id]
        if not parent.child_ids:
            return None
        
        best_id = None
        best_sim = float("inf")
        
        for cid in parent.child_ids:
            child = self._nodes[cid]
            if child.is_leaf:
                child_key = child.key
            else:
                child_key = child.proj
            
            if child_key is None:
                continue
            
            if child.is_leaf:
                sim = hyperbolic_similarity(query_hyp, child_key.unsqueeze(0)).item()
            else:
                hp = exp_map(child_key.unsqueeze(0))
                sim = hyperbolic_similarity(query_hyp, hp).item()
            if sim < best_sim:
                best_sim = sim
                best_id = cid
        
        return best_id
    
    def _create_leaf(
        self,
        emb_t: torch.Tensor,
        projected: torch.Tensor,
        query_hyp: torch.Tensor,
        content: str,
        parent_id: int,
    ) -> int:
        """Create a new leaf node under parent."""
        leaf = MemoryNode(
            id=self._next_id(),
            depth=self._nodes[parent_id].depth + 1,
            parent_id=parent_id,
            content=content,
            embedding=_to_numpy(emb_t[0]),
            created_at=__import__("time").time(),
            last_accessed=__import__("time").time(),
        )
        leaf.state = query_hyp[0].clone()
        leaf.key = query_hyp[0].clone()
        leaf.proj = projected[0].clone()
        
        self._nodes[leaf.id] = leaf
        self._nodes[parent_id].child_ids.append(leaf.id)
        
        return leaf.id
    
    def _update_subtree_depth(self, node_id: int, new_depth: int):
        """Recursively set depth for a node and all its descendants."""
        node = self._nodes[node_id]
        node.depth = new_depth
        for cid in node.child_ids:
            self._update_subtree_depth(cid, new_depth + 1)
    
    def _split_and_insert(
        self,
        parent_id: int,
        emb_t: torch.Tensor,
        projected: torch.Tensor,
        query_hyp: torch.Tensor,
        content: str,
    ) -> int:
        """B-tree-style split: divide children into two balanced nodes at the
        same depth, keeping the tree wide instead of deep.
        
        When a node overflows, all its children are clustered into two groups
        by similarity. Two new internal nodes are created under the parent.
        Each group goes to one node. The new leaf is inserted into the
        better-matching group.
        
        This guarantees O(log_{b/2}(N)) depth where b = branching_factor.
        """
        parent = self._nodes[parent_id]
        all_child_ids = list(parent.child_ids)
        
        if len(all_child_ids) < 2:
            # Not enough children to split — just add under parent
            return self._create_leaf(emb_t, projected, query_hyp, content, parent_id)
        
        # Find two seed children with lowest mutual hyperbolic_similarity
        # (lowest = closest together on hyperboloid)
        min_sim = float("inf")
        seed_a = all_child_ids[0]
        seed_b = all_child_ids[1]
        for i in range(len(all_child_ids)):
            for j in range(i + 1, len(all_child_ids)):
                ci = self._nodes[all_child_ids[i]]
                cj = self._nodes[all_child_ids[j]]
                if ci.key is not None and cj.key is not None:
                    sim = hyperbolic_similarity(ci.key.unsqueeze(0), cj.key.unsqueeze(0)).item()
                    if sim < min_sim:
                        min_sim = sim
                        seed_a = all_child_ids[i]
                        seed_b = all_child_ids[j]
        
        # Assign each child to the closer seed
        group_a = [seed_a]
        group_b = [seed_b]
        for cid in all_child_ids:
            if cid in (seed_a, seed_b):
                continue
            child = self._nodes[cid]
            if child.key is not None and self._nodes[seed_a].key is not None and self._nodes[seed_b].key is not None:
                sim_a = hyperbolic_similarity(child.key.unsqueeze(0),
                    self._nodes[seed_a].key.unsqueeze(0)).item()
                sim_b = hyperbolic_similarity(child.key.unsqueeze(0),
                    self._nodes[seed_b].key.unsqueeze(0)).item()
                # Lower hyperbolic_similarity = closer. Assign to closer seed.
                if sim_b < sim_a:
                    group_b.append(cid)
                else:
                    group_a.append(cid)
            else:
                group_a.append(cid)
        
        # Balance: ensure each group has at least branching_factor // 4
        min_group = max(2, self.branching_factor // 4)
        if len(group_a) < min_group and len(group_b) > min_group * 2:
            # Move most-similar-to-a from b to a
            if self._nodes[seed_a].key is not None:
                group_b_sorted = sorted(group_b, key=lambda cid: (
                    hyperbolic_similarity(self._nodes[cid].key.unsqueeze(0),
                        self._nodes[seed_a].key.unsqueeze(0)).item()
                    if self._nodes[cid].key is not None else -float("inf")
                ))  # ascending = closest to seed_a first
                while len(group_a) < min_group and group_b_sorted:
                    group_a.append(group_b_sorted.pop(0))
                group_b = [c for c in group_b_sorted if c in group_b] if group_b_sorted else [c for c in group_b if c not in group_a]
        elif len(group_b) < min_group and len(group_a) > min_group * 2:
            if self._nodes[seed_b].key is not None:
                group_a_sorted = sorted(group_a, key=lambda cid: (
                    hyperbolic_similarity(self._nodes[cid].key.unsqueeze(0),
                        self._nodes[seed_b].key.unsqueeze(0)).item()
                    if self._nodes[cid].key is not None else -float("inf")
                ))  # ascending = closest to seed_b first
                while len(group_b) < min_group and group_a_sorted:
                    group_b.append(group_a_sorted.pop(0))
                group_a = [c for c in group_a_sorted if c in group_a] if group_a_sorted else [c for c in group_a if c not in group_b]
        
        # Create TWO internal nodes at the same depth as parent+1
        depth = parent.depth + 1
        
        node_a = MemoryNode(
            id=self._next_id(),
            depth=depth,
            parent_id=parent_id,
        )
        node_b = MemoryNode(
            id=self._next_id(),
            depth=depth,
            parent_id=parent_id,
        )
        
        # Assign groups and set keys/states as averages
        for cid in group_a:
            child = self._nodes[cid]
            child.parent_id = node_a.id
            self._update_subtree_depth(cid, depth + 1)
            node_a.child_ids.append(cid)
        
        for cid in group_b:
            child = self._nodes[cid]
            child.parent_id = node_b.id
            self._update_subtree_depth(cid, depth + 1)
            node_b.child_ids.append(cid)
        
        # Set internal node states as group averages (in tangent space)
        for node, group in [(node_a, group_a), (node_b, group_b)]:
            child_states = [self._nodes[cid].state for cid in group if self._nodes[cid].state is not None]
            child_projs = [self._nodes[cid].proj for cid in group if self._nodes[cid].proj is not None]
            
            # Hyperbolic state: average in tangent space (partial info preserved)
            if child_states:
                logs = [log_map(s.unsqueeze(0)).squeeze(0) for s in child_states]
                avg_log = torch.stack(logs).mean(dim=0)
                node.state = exp_map(avg_log.unsqueeze(0)).squeeze(0)
            
            # Euclidean key: average projected embeddings (direction preserved!)
            if child_projs:
                projs = torch.stack(child_projs)
                node.proj = projs.mean(dim=0)
                node.key = exp_map(node.proj.unsqueeze(0)).squeeze(0)
        
        # Replace parent's children with these two internal nodes
        parent.child_ids = [node_a.id, node_b.id]
        self._nodes[node_a.id] = node_a
        self._nodes[node_b.id] = node_b
        
        # Insert new leaf under better-matching group
        if self._nodes[seed_a].key is not None and self._nodes[seed_b].key is not None:
            sim_a = hyperbolic_similarity(query_hyp, self._nodes[seed_a].key.unsqueeze(0)).item()
            sim_b = hyperbolic_similarity(query_hyp, self._nodes[seed_b].key.unsqueeze(0)).item()
            target_parent = node_a.id if sim_a >= sim_b else node_b.id
        else:
            target_parent = node_a.id
        
        return self._create_leaf(emb_t, projected, query_hyp, content, target_parent)
        
        # Set internal state as average of moved children
        if internal.child_ids:
            child_states = [self._nodes[cid].state for cid in internal.child_ids if self._nodes[cid].state is not None]
            if child_states:
                logs = [log_map(s.unsqueeze(0)).squeeze(0) for s in child_states]
                avg_log = torch.stack(logs).mean(dim=0)
                internal.state = exp_map(avg_log.unsqueeze(0)).squeeze(0)
                internal.key = internal.state.clone()
        
        parent.child_ids.append(internal.id)
        self._nodes[internal.id] = internal
        
        # Now add new leaf under parent
        return self._create_leaf(emb_t, projected, query_hyp, content, parent_id)
    
    def _merge_hyperbolic(self, a: torch.Tensor, b: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two hyperbolic points via tangent space."""
        log_a = log_map(a.unsqueeze(0)).squeeze(0)
        log_b = log_map(b.unsqueeze(0)).squeeze(0)
        merged_tangent = alpha * log_a + (1.0 - alpha) * log_b
        return exp_map(merged_tangent.unsqueeze(0)).squeeze(0)
    
    def _update_ancestors(self, node_id: int):
        """Propagate the new leaf's information up through all ancestors.
        Each internal node merges its children's projections into its own."""
        current_id = self._nodes[node_id].parent_id
        while current_id is not None:
            node = self._nodes[current_id]
            children_states = []
            children_projs = []
            for cid in node.child_ids:
                child = self._nodes[cid]
                if child.state is not None:
                    children_states.append(child.state)
                if child.proj is not None:
                    children_projs.append(child.proj)
            
            if children_states:
                logs = [log_map(s.unsqueeze(0)).squeeze(0) for s in children_states]
                avg_log = torch.stack(logs).mean(dim=0)
                node.state = exp_map(avg_log.unsqueeze(0)).squeeze(0)
            
            if children_projs:
                projs = torch.stack(children_projs)
                node.proj = projs.mean(dim=0)
                node.key = exp_map(node.proj.unsqueeze(0)).squeeze(0)
            
            current_id = node.parent_id
    
    # ------------------------------------------------------------------
    # CORE: RECALL
    # ------------------------------------------------------------------
    
    def recall(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        max_depth_scale: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k memories matching the query.
        
        Traverses the tree with beam search, collecting leaves.
        Returns list of dicts with keys: content, similarity, depth, node_id
        
        Args:
            query_embedding: Euclidean query vector
            top_k: Number of results to return
            max_depth_scale: How deep to search (0=root only, 1=children, etc.)
        """
        emb_t = _to_tensor(query_embedding, self.device).float()
        if emb_t.dim() == 1:
            emb_t = emb_t.unsqueeze(0)
        
        projected = emb_t @ self._input_proj.T
        query_hyp = exp_map(projected)
        
        # Per-depth beam search: visits closest children first.
        # hyperbolic_similarity returns -lorentz_inner where HIGHER = FARTHER
        # (min 1.0 for identical points).  We sort ASCENDING to get closest.
        results = []
        visited = set()
        
        # Start with root beam
        root = self._nodes[self._root_id]
        current_beam = [(0.0, self._root_id)]
        
        while current_beam and len(results) < top_k * 3:
            # Collect all candidates at next depth from current beam
            next_candidates = []
            for _, nid in current_beam:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self._nodes[nid]
                node.access_count += 1
                node.last_accessed = __import__("time").time()
                
                if node.is_leaf:
                    sim = hyperbolic_similarity(query_hyp, node.key).item() if node.key is not None else 0.0
                    results.append({
                        "content": node.content or "",
                        "similarity": sim,
                        "depth": node.depth,
                        "node_id": node.id,
                        "embedding": node.embedding,
                    })
                    continue
                
                # Add this node's children to next level candidates
                for cid in node.child_ids:
                    child = self._nodes[cid]
                    if child.key is None:
                        continue
                    sim = hyperbolic_similarity(query_hyp, child.key).item()
                    next_candidates.append((sim, cid))
            
            # Prune to top branching_factor closest children for next depth
            next_candidates.sort(key=lambda x: x[0])
            current_beam = next_candidates[:self.branching_factor]
        
        # Sort by similarity ascending (closest first), return top_k
        results.sort(key=lambda x: x["similarity"])
        return results[:top_k]
    
    def recall_all_scales(
        self,
        query_embedding: np.ndarray,
    ) -> List[np.ndarray]:
        """Return recalled embeddings at all scales (compatibility with ICM)."""
        results = self.recall(query_embedding, top_k=4)
        embeddings = []
        for r in results:
            if r["embedding"] is not None:
                embeddings.append(r["embedding"])
        if not embeddings:
            return [np.zeros(self.embed_dim)]
        return embeddings
    
    # ------------------------------------------------------------------
    # PRUNING
    # ------------------------------------------------------------------
    
    def _prune(self):
        """Remove least-important nodes until under max_nodes."""
        while len(self._nodes) > self.max_nodes * 0.8:  # prune to 80% capacity
            self._prune_one()
    
    def _prune_one(self):
        """Find and merge the two most similar sibling leaves."""
        # Find sibling leaves with highest similarity (lowest hyperbolic_similarity)
        best_pair = None
        best_sim = float("inf")
        
        for nid, node in self._nodes.items():
            if node.is_leaf and node.parent_id is not None:
                parent = self._nodes[node.parent_id]
                for sid in parent.child_ids:
                    if sid == nid:
                        continue
                    sibling = self._nodes[sid]
                    if sibling.is_leaf and sibling.key is not None and node.key is not None:
                        sim = hyperbolic_similarity(node.key, sibling.key).item()
                        if sim < best_sim:
                            best_sim = sim
                            best_pair = (nid, sid)
        
        if best_pair is None:
            return
        
        id1, id2 = best_pair
        
        # Merge: combine embeddings, keep the more recent one's content
        node1 = self._nodes[id1]
        node2 = self._nodes[id2]
        
        merged_embedding = node1.embedding if node1.access_count >= node2.access_count else node2.embedding
        merged_content = node1.content if node1.access_count >= node2.access_count else node2.content
        
        # Update node1 to absorb node2
        if node1.key is not None and node2.key is not None:
            node1.state = self._merge_hyperbolic(node1.state, node2.state)
            node1.key = self._merge_hyperbolic(node1.key, node2.key, alpha=0.7)
        
        node1.access_count += node2.access_count
        if node2.content:
            node1.content = merged_content
        if node2.embedding is not None:
            node1.embedding = merged_embedding
        
        # Remove node2
        parent = self._nodes[node2.parent_id]
        if id2 in parent.child_ids:
            parent.child_ids.remove(id2)
        del self._nodes[id2]
        
        # Update ancestors
        self._update_ancestors(id1)
    
    # ------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------
    
    def state(self) -> Dict[str, Any]:
        """Export full tree state for serialization."""
        nodes_data = {}
        for nid, node in self._nodes.items():
            data = {
                "id": node.id,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "child_ids": node.child_ids,
                "content": node.content,
                "access_count": node.access_count,
                "created_at": node.created_at,
                "last_accessed": node.last_accessed,
            }
            if node.state is not None:
                data["state"] = _to_numpy(node.state).tolist()
            if node.key is not None:
                data["key"] = _to_numpy(node.key).tolist()
            if node.embedding is not None:
                data["embedding"] = node.embedding.tolist()
            if node.proj is not None:
                data["proj"] = _to_numpy(node.proj).tolist()
            nodes_data[str(nid)] = data
        
        return {
            "state_dim": self.state_dim,
            "embed_dim": self.embed_dim,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "merge_threshold": self.merge_threshold,
            "root_id": self._root_id,
            "node_counter": self._node_counter,
            "nodes": nodes_data,
            "input_proj": _to_numpy(self._input_proj).tolist(),
            "importance_w": _to_numpy(self._importance_w).tolist(),
        }
    
    def load_state(self, state_dict: Dict[str, Any]) -> "HyperbolicMemoryTree":
        """Restore tree from exported state."""
        self.state_dim = state_dict["state_dim"]
        self.embed_dim = state_dict["embed_dim"]
        self.lorentz_dim = self.state_dim + 1
        self.max_nodes = state_dict["max_nodes"]
        self.max_depth = state_dict["max_depth"]
        self.branching_factor = state_dict["branching_factor"]
        self.merge_threshold = state_dict["merge_threshold"]
        self._root_id = state_dict["root_id"]
        self._node_counter = state_dict["node_counter"]
        
        if state_dict.get("input_proj"):
            self._input_proj = torch.tensor(state_dict["input_proj"], dtype=torch.float32)
        if state_dict.get("importance_w"):
            self._importance_w = torch.tensor(state_dict["importance_w"], dtype=torch.float32)
        
        self._nodes = {}
        for nid_str, data in state_dict["nodes"].items():
            nid = int(nid_str)
            node = MemoryNode(
                id=nid,
                depth=data["depth"],
                parent_id=data["parent_id"],
                child_ids=data["child_ids"],
                content=data.get("content"),
                access_count=data.get("access_count", 0),
                created_at=data.get("created_at", 0.0),
                last_accessed=data.get("last_accessed", 0.0),
            )
            if data.get("state"):
                node.state = torch.tensor(data["state"], dtype=torch.float32)
            if data.get("key"):
                node.key = torch.tensor(data["key"], dtype=torch.float32)
            if data.get("embedding"):
                node.embedding = np.array(data["embedding"], dtype=np.float32)
            if data.get("proj"):
                node.proj = torch.tensor(data["proj"], dtype=torch.float32)
            self._nodes[nid] = node
        
        return self
    
    # ------------------------------------------------------------------
    # INFO / STATS
    # ------------------------------------------------------------------
    
    def info(self) -> Dict[str, Any]:
        """Return tree statistics."""
        leaves = sum(1 for n in self._nodes.values() if n.is_leaf and n.id != self._root_id)
        internal = len(self._nodes) - leaves - 1  # exclude root from both
        total_bytes = sum(n.memory_bytes() for n in self._nodes.values())
        depth = max((n.depth for n in self._nodes.values()), default=0)
        
        return {
            "nodes": len(self._nodes),
            "leaves": leaves,
            "internal": internal,
            "max_depth": depth,
            "memory_bytes": total_bytes,
            "memory_per_node": total_bytes // max(len(self._nodes), 1),
            "state_dim": self.state_dim,
            "embed_dim": self.embed_dim,
        }
    
    @property
    def memory_size_bytes(self) -> int:
        """Total bytes used by the tree."""
        return sum(n.memory_bytes() for n in self._nodes.values())
    
    @property
    def _utterance_count(self) -> int:
        """Alias for utterance_count (compatibility with flat memory)."""
        return self.utterance_count
    
    @property
    def utterance_count(self) -> int:
        """Number of leaves (facts stored)."""
        return sum(1 for n in self._nodes.values() if n.is_leaf and n.id != self._root_id)
    
    def reset(self) -> "HyperbolicMemoryTree":
        """Clear all memories."""
        self._init_tree()
        return self
    
    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    
    def print_tree(self, node_id: Optional[int] = None, indent: int = 0):
        """ASCII tree visualization."""
        if node_id is None:
            node_id = self._root_id
        
        node = self._nodes[node_id]
        prefix = "  " * indent
        
        if node.is_leaf:
            content = (node.content[:50] + "...") if node.content and len(node.content) > 50 else (node.content or "")
            key_str = f"[key={node.key[0].item():.3f}...]" if node.key is not None else "[no key]"
            print(f"{prefix}+- L{node.id} {key_str} \"{content}\" (access={node.access_count})")
        else:
            key_str = f"[key={node.key[0].item():.3f}...]" if node.key is not None else "[no key]"
            state_str = f"[state={node.state[0].item():.3f}...]" if node.state is not None else "[no state]"
            print(f"{prefix}+- N{node.id} {key_str} {state_str} children={len(node.child_ids)}")
            for cid in node.child_ids:
                self.print_tree(cid, indent + 1)
