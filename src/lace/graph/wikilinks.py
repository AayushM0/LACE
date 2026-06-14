"""Generate and inject wikilinks into memory markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Set
import logging

from lace.core.config import get_lace_home, load_config
from lace.graph.graph import build_graph
from lace.memory.markdown import load_all_memories, markdown_to_memory
from lace.memory.models import MemoryObject

logger = logging.getLogger(__name__)


def extract_existing_wikilinks(content: str) -> Set[str]:
    """Extract all [[wikilink]] references from markdown content."""
    import re
    pattern = r"\[\[([^\]]+)\]\]"
    matches = re.findall(pattern, content)
    return set(matches)


def get_high_value_concepts(
    memory_id: str,
    graph,
    max_links: int = 5,
) -> list[str]:
    """Get the most valuable concepts to link for a memory.
    
    Strategy:
    1. Direct tags (already linked via tagged_with edges) — SKIP (already in frontmatter)
    2. Concepts with strong co-occurrence (weight > 2)
    3. Concepts linked from related memories (2-hop traversal)
    
    Filters out:
    - Generic/common concepts (appear in >50% of memories)
    - Single-letter concepts
    - Concepts identical to existing tags
    
    Returns top N by relevance score.
    """
    if memory_id not in graph:
        return []
    
    # Get memory's direct tags (skip these — already in frontmatter)
    direct_tags = set()
    for target in graph.successors(memory_id):
        edge_data = graph.get_edge_data(memory_id, target)
        if edge_data and edge_data.get("relation") == "tagged_with":
            direct_tags.add(target)
    
    # Collect candidate concepts with scores
    candidates = {}
    
    # Strategy 1: Strong co-occurrence (concepts that frequently appear together)
    for concept in graph.nodes():
        if graph.nodes[concept].get("type") != "concept":
            continue
        if concept in direct_tags:
            continue  # Skip tags
        
        # Check if this concept co-occurs with our memory's concepts
        for direct_tag in direct_tags:
            if graph.has_edge(direct_tag, concept):
                edge_data = graph.get_edge_data(direct_tag, concept)
                if edge_data.get("relation") == "co_occurs":
                    weight = edge_data.get("weight", 1)
                    if weight >= 2:  # Only strong co-occurrence
                        candidates[concept] = candidates.get(concept, 0) + weight
    
    # Strategy 2: 2-hop concepts (concepts from related memories)
    # Find memories that share tags with this memory
    related_memories = set()
    for tag in direct_tags:
        for pred in graph.predecessors(tag):
            if (graph.nodes[pred].get("type") == "memory" and 
                pred != memory_id):
                related_memories.add(pred)
    
    # Get concepts from those related memories
    for related_mem in related_memories:
        for concept in graph.successors(related_mem):
            if graph.nodes[concept].get("type") == "concept":
                if concept not in direct_tags:
                    # Boost score if multiple related memories share this concept
                    candidates[concept] = candidates.get(concept, 0) + 0.5
    
    # Filter out noise
    filtered = {}
    total_memories = sum(1 for n in graph.nodes() if graph.nodes[n].get("type") == "memory")
    
    for concept, score in candidates.items():
        # Skip single-letter or very short concepts
        if len(concept) <= 2:
            continue
        
        # Skip overly common concepts (appear in >50% of memories)
        concept_freq = sum(1 for pred in graph.predecessors(concept) 
                          if graph.nodes[pred].get("type") == "memory")
        if total_memories > 0 and concept_freq / total_memories > 0.5:
            continue
        
        filtered[concept] = score
    
    # Return top N by score
    sorted_concepts = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return [concept for concept, score in sorted_concepts[:max_links]]


def inject_wikilinks_into_memory(
    memory: MemoryObject,
    graph,
    markdown_path: Path,
) -> bool:
    """Inject high-value wikilinks into a memory markdown file.
    
    Only adds links for:
    - Concepts with strong co-occurrence (not just tags)
    - Concepts from related memories
    
    Does NOT duplicate tags as wikilinks (tags are already in frontmatter).
    """
    if not markdown_path.exists():
        logger.warning(f"File not found: {markdown_path}")
        return False
    
    content = markdown_path.read_text(encoding="utf-8")
    original_content = content
    
    # Split frontmatter and content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) < 3:
            logger.warning(f"Invalid frontmatter in {markdown_path}")
            return False
        frontmatter = parts[1]
        body = parts[2].lstrip("\n")
    else:
        frontmatter = ""
        body = content
    
    # Get high-value concepts (NOT just tags)
    related = get_high_value_concepts(memory.id, graph, max_links=5)
    
    if not related:
        # Remove existing **Related:** section if no valuable links
        if "**Related:**" in body:
            import re
            pattern = r"\n\*\*Related:\*\*.*?(?=\n\n|$)"
            new_body = re.sub(pattern, "", body, flags=re.DOTALL).rstrip()
            if frontmatter:
                new_content = f"---{frontmatter}---\n{new_body}"
            else:
                new_content = new_body
            
            if new_content != original_content:
                markdown_path.write_text(new_content, encoding="utf-8")
                logger.info(f"Removed low-value wikilinks from: {markdown_path.name}")
                return True
        return False
    
    # Create wikilinks section
    wikilinks_text = "\n\n**Related:**\n" + " ".join([f"[[{c}]]" for c in related])
    
    # Check if wikilinks section already exists
    if "**Related:**" in body:
        # Replace existing section
        import re
        pattern = r"\n\*\*Related:\*\*.*?(?=\n\n|$)"
        new_body = re.sub(pattern, wikilinks_text, body, flags=re.DOTALL)
    else:
        # Append new section
        new_body = body.rstrip() + wikilinks_text
    
    # Reconstruct file
    if frontmatter:
        new_content = f"---{frontmatter}---\n{new_body}"
    else:
        new_content = new_body
    
    # Write back if changed
    if new_content != original_content:
        markdown_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Added wikilinks to: {markdown_path.name}")
        return True
    
    return False


def inject_wikilinks_all(lace_home: Path | None = None) -> dict:
    """Inject wikilinks into all memory files.
    
    Returns:
        Dict with counts of updated files
    """
    lace_home = lace_home or get_lace_home()
    config = load_config(lace_home)
    vault_path = config.vault_path(lace_home)
    
    # Load graph
    from lace.core.engine import GraphManager
    gm = GraphManager(lace_home=lace_home, config=config)
    graph = gm.get_graph()
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty")
        return {"updated": 0, "total": 0}
    
    # Load all memories
    memories = load_all_memories(vault_path)
    
    updated = 0
    total = 0
    
    # Find all .md files recursively in vault
    for md_path in vault_path.rglob("mem_*.md"):
        total += 1
        
        # Extract memory ID from filename
        import re
        match = re.search(r"mem_([a-f0-9]{12})", md_path.name)
        if not match:
            continue
        
        memory_id = "mem_" + match.group(1)
        
        # Find corresponding memory object
        memory = next((m for m in memories if m.id == memory_id), None)
        if not memory:
            logger.warning(f"Memory {memory_id} not found in store")
            continue
        
        if inject_wikilinks_into_memory(memory, graph, md_path):
            updated += 1
    
    return {
        "updated": updated,
        "total": total,
    }