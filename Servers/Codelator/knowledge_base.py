import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import math
from collections import Counter

@dataclass
class CodeSnippet:
    id: str
    title: str
    code: str
    language: str  # python, java, javascript, etc.
    tags: List[str]  # specific features/patterns
    description: Optional[str] = ""
    
@dataclass 
class KnowledgeEntry:
    framework: str  # e.g., "spark", "react", "django"
    concept: str    # e.g., "mllib", "hooks", "authentication"
    snippets: List[CodeSnippet]
    file_path: str

class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(exist_ok=True)
        self.index_file = self.knowledge_dir / "index.json"
        self.entries: Dict[str, KnowledgeEntry] = {}  # key: "framework/concept"
        self.load_index()
    
    def load_index(self):
        """Load all knowledge entries from disk"""
        self.entries = {}
        
        # Scan all framework folders
        for framework_dir in self.knowledge_dir.iterdir():
            if framework_dir.is_dir() and not framework_dir.name.startswith('.'):
                framework = framework_dir.name
                
                # Scan all concept files in framework
                for md_file in framework_dir.glob("*.md"):
                    concept = md_file.stem
                    snippets = self._parse_concept_file(md_file)
                    
                    key = f"{framework}/{concept}"
                    self.entries[key] = KnowledgeEntry(
                        framework=framework,
                        concept=concept,
                        snippets=snippets,
                        file_path=str(md_file)
                    )
    
    def _parse_concept_file(self, file_path: Path) -> List[CodeSnippet]:
        """Parse a concept markdown file to extract snippets"""
        if not file_path.exists():
            return []
        
        snippets = []
        content = file_path.read_text()
        
        # Pattern: ## Title [language] {tags}
        # Followed by optional description
        # Followed by code block
        pattern = r'## (.*?)\s*\[(.*?)\]\s*\{(.*?)\}\s*\n(.*?)```.*?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for title, language, tags_str, description, code in matches:
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            snippet_id = hashlib.md5(f"{title}{code}".encode()).hexdigest()[:8]
            
            snippets.append(CodeSnippet(
                id=snippet_id,
                title=title.strip(),
                code=code.strip(),
                language=language.strip().lower(),
                tags=tags,
                description=description.strip()
            ))
        
        return snippets
    
    def add_snippet(self, framework: str, concept: str, snippet: CodeSnippet):
        """Add a snippet to the knowledge base"""
        # Create framework directory if needed
        framework_dir = self.knowledge_dir / framework.lower()
        framework_dir.mkdir(exist_ok=True)
        
        # Determine file path
        file_path = framework_dir / f"{concept.lower()}.md"
        
        # Add to file
        self._append_snippet_to_file(file_path, snippet, framework, concept)
        
        # Reload this entry
        key = f"{framework}/{concept}"
        self.entries[key] = KnowledgeEntry(
            framework=framework,
            concept=concept,
            snippets=self._parse_concept_file(file_path),
            file_path=str(file_path)
        )
    
    def _append_snippet_to_file(self, file_path: Path, snippet: CodeSnippet, 
                                framework: str, concept: str):
        """Append a snippet to a concept file"""
        # Create file with header if new
        if not file_path.exists():
            header = f"# {framework.title()} - {concept.title()}\n\n"
            header += f"Code snippets for {concept} in {framework}.\n\n"
            file_path.write_text(header)
        
        # Append snippet
        content = f"\n## {snippet.title} [{snippet.language}] "
        content += f"{{{', '.join(snippet.tags)}}}\n\n"
        
        if snippet.description:
            content += f"{snippet.description}\n\n"
        
        content += f"```{snippet.language}\n{snippet.code}\n```\n"
        
        with open(file_path, 'a') as f:
            f.write(content)
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[CodeSnippet, str, float]]:
        """BM25 search returning (snippet, framework/concept, score)"""
        all_results = []
        
        for key, entry in self.entries.items():
            for snippet in entry.snippets:
                # Create searchable document
                doc = f"{snippet.title} {snippet.title} {snippet.title} "  # 3x weight
                doc += f"{' '.join(snippet.tags)} {' '.join(snippet.tags)} "  # 2x weight  
                doc += f"{entry.framework} {entry.concept} "
                doc += f"{snippet.language} "
                doc += f"{snippet.description} "
                doc += snippet.code
                
                # Simple scoring for MVP
                score = 0
                query_terms = query.lower().split()
                doc_lower = doc.lower()
                
                for term in query_terms:
                    # Count occurrences
                    count = doc_lower.count(term)
                    if count > 0:
                        # Basic TF-IDF style scoring
                        score += count * (1.0 / (1.0 + math.log(1 + len(self.entries))))
                
                if score > 0:
                    all_results.append((snippet, key, score))
        
        # Sort and return top results
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:limit]
    
    def list_frameworks(self) -> List[Dict[str, any]]:
        """List all frameworks with their concepts"""
        frameworks = {}
        
        for key, entry in self.entries.items():
            if entry.framework not in frameworks:
                frameworks[entry.framework] = {
                    'name': entry.framework,
                    'concepts': [],
                    'snippet_count': 0
                }
            
            frameworks[entry.framework]['concepts'].append(entry.concept)
            frameworks[entry.framework]['snippet_count'] += len(entry.snippets)
        
        return list(frameworks.values())
    
    def get_concept(self, framework: str, concept: str) -> Optional[KnowledgeEntry]:
        """Get all snippets for a framework/concept"""
        key = f"{framework}/{concept}"
        return self.entries.get(key)