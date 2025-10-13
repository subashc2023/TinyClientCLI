import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from knowledge_base import KnowledgeBase, CodeSnippet

# Initialize
server = Server("code-translator")
kb = KnowledgeBase()

# Clear instructions for LLM
EXTRACTION_INSTRUCTIONS = """
# Code Snippet Extraction Instructions

When processing content, extract code snippets and organize them using this structure:

## Organization Rules:
1. **Framework**: The main technology/library (e.g., "spark", "react", "tensorflow")
2. **Concept**: The specific feature/pattern (e.g., "mllib", "hooks", "layers")  
3. **Snippet**: Individual code example with:
   - Title: Clear, descriptive name
   - Language: Programming language (python, java, javascript, etc.)
   - Tags: Specific features used (e.g., "clustering", "k-means", "dataframe")
   - Code: The actual code

## Example Extraction:

Given content about Spark MLlib, you might extract:

{
  "framework": "spark",
  "concept": "mllib",
  "snippets": [
    {
      "title": "K-Means Clustering with DataFrames",
      "language": "python",
      "tags": ["clustering", "k-means", "dataframe", "unsupervised"],
      "description": "Cluster data using K-Means with DataFrame API",
      "code": "from pyspark.ml.clustering import KMeans\n..."
    },
    {
      "title": "K-Means Clustering with RDD",
      "language": "java",
      "tags": ["clustering", "k-means", "rdd", "unsupervised"],  
      "description": "Cluster data using K-Means with RDD API",
      "code": "import org.apache.spark.mllib.clustering.KMeans;\n..."
    }
  ]
}

## More Examples:

React content → framework: "react", concepts: "hooks", "components", "routing"
Django content → framework: "django", concepts: "models", "views", "authentication"
PyTorch content → framework: "pytorch", concepts: "tensors", "models", "training"
Express content → framework: "express", concepts: "routing", "middleware", "authentication"
"""

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="retrieve",
            description="Extract and store code snippets from content",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to process"
                    },
                    "extracted": {
                        "type": "object",
                        "description": "Extracted snippets (if already processed by LLM)",
                        "properties": {
                            "framework": {"type": "string"},
                            "concept": {"type": "string"},
                            "snippets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "language": {"type": "string"},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                        "description": {"type": "string"},
                                        "code": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="translate",
            description="Find relevant code examples for translation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Code/description to find examples for"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Optional: Preferred programming language"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_frameworks",
            description="List all frameworks and their concepts in the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="get_concept",
            description="Get all snippets for a specific framework/concept",
            inputSchema={
                "type": "object", 
                "properties": {
                    "framework": {"type": "string"},
                    "concept": {"type": "string"}
                },
                "required": ["framework", "concept"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Optional[Dict] = None) -> List[types.TextContent]:
    if name == "retrieve":
        return await handle_retrieve(arguments)
    elif name == "translate":
        return await handle_translate(arguments)
    elif name == "list_frameworks":
        return await handle_list_frameworks()
    elif name == "get_concept":
        return await handle_get_concept(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_retrieve(args: Dict) -> List[types.TextContent]:
    """Process content and store snippets"""
    content = args.get("content", "")
    extracted = args.get("extracted", None)
    
    if not extracted:
        # Return instructions for LLM
        response = "# Snippet Extraction Required\n\n"
        response += EXTRACTION_INSTRUCTIONS
        response += "\n\n## Content to Process:\n\n"
        response += f"```\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```\n\n"
        response += "Please call `retrieve` again with the `extracted` parameter containing the organized snippets."
        return [types.TextContent(type="text", text=response)]
    
    # Process extracted snippets
    framework = extracted.get("framework", "unknown")
    concept = extracted.get("concept", "general")
    snippets_data = extracted.get("snippets", [])
    
    response = f"# Storing Snippets: {framework}/{concept}\n\n"
    
    for snippet_data in snippets_data:
        try:
            snippet = CodeSnippet(
                id="",  # Auto-generated
                title=snippet_data.get("title", "Untitled"),
                code=snippet_data.get("code", ""),
                language=snippet_data.get("language", "unknown"),
                tags=snippet_data.get("tags", []),
                description=snippet_data.get("description", "")
            )
            
            kb.add_snippet(framework, concept, snippet)
            
            response += f"✓ **{snippet.title}** [{snippet.language}]\n"
            response += f"  Tags: {', '.join(snippet.tags)}\n\n"
            
        except Exception as e:
            response += f"✗ Error: {e}\n\n"
    
    response += f"\n**Location:** `knowledge/{framework}/{concept}.md`"
    
    return [types.TextContent(type="text", text=response)]

async def handle_translate(args: Dict) -> List[types.TextContent]:
    """Find relevant code examples"""
    query = args.get("query", "")
    target_lang = args.get("target_language", None)
    limit = args.get("limit", 5)
    
    results = kb.search(query, limit=limit)
    
    response = f"# Code Translation References\n\n"
    response += f"**Query:** {query}\n"
    if target_lang:
        response += f"**Preferred Language:** {target_lang}\n"
    response += "\n"
    
    if results:
        # Group by framework/concept
        by_concept = {}
        for snippet, key, score in results:
            if key not in by_concept:
                by_concept[key] = []
            by_concept[key].append((snippet, score))
        
        for key, snippets in by_concept.items():
            framework, concept = key.split('/')
            response += f"## {framework.title()} - {concept.title()}\n\n"
            
            for snippet, score in snippets:
                # Highlight if matches target language
                lang_marker = " ⭐" if target_lang and snippet.language == target_lang else ""
                
                response += f"### {snippet.title} [{snippet.language}]{lang_marker}\n"
                response += f"*Relevance: {score:.2f} | Tags: {', '.join(snippet.tags)}*\n\n"
                
                if snippet.description:
                    response += f"{snippet.description}\n\n"
                
                # Show code (truncated if long)
                code = snippet.code
                if len(code) > 500:
                    code = code[:500] + "\n# ... (truncated)"
                
                response += f"```{snippet.language}\n{code}\n```\n\n"
        
        # Translation hints
        response += "## Translation Tips\n\n"
        
        # Find examples in different languages for same concept
        languages_seen = set()
        for snippet, _, _ in results:
            languages_seen.add(snippet.language)
        
        if len(languages_seen) > 1:
            response += f"✓ Found examples in: {', '.join(languages_seen)}\n"
            response += "Compare the patterns across languages above.\n\n"
        
        if target_lang and target_lang not in languages_seen:
            response += f"⚠ No {target_lang} examples found. Consider:\n"
            response += "1. Adapting the patterns from available languages\n"
            response += "2. Adding {target_lang} examples using `retrieve`\n"
            
    else:
        response += "No relevant examples found.\n\n"
        response += "Try:\n"
        response += "1. Using more specific keywords\n"
        response += "2. Adding examples with `retrieve` first\n"
        response += "3. Listing frameworks with `list_frameworks`\n"
    
    return [types.TextContent(type="text", text=response)]

async def handle_list_frameworks() -> List[types.TextContent]:
    """List all frameworks and concepts"""
    frameworks = kb.list_frameworks()
    
    response = "# Knowledge Base Structure\n\n"
    
    if frameworks:
        for fw in sorted(frameworks, key=lambda x: x['name']):
            response += f"## {fw['name'].title()}\n"
            response += f"*{fw['snippet_count']} snippets across {len(fw['concepts'])} concepts*\n\n"
            
            for concept in sorted(fw['concepts']):
                response += f"- `{concept}`\n"
            response += "\n"
    else:
        response += "Knowledge base is empty. Use `retrieve` to add content.\n"
    
    return [types.TextContent(type="text", text=response)]

async def handle_get_concept(args: Dict) -> List[types.TextContent]:
    """Get all snippets for a framework/concept"""
    framework = args.get("framework", "").lower()
    concept = args.get("concept", "").lower()
    
    entry = kb.get_concept(framework, concept)
    
    if not entry:
        return [types.TextContent(
            type="text", 
            text=f"No snippets found for {framework}/{concept}"
        )]
    
    response = f"# {framework.title()} - {concept.title()}\n\n"
    response += f"*{len(entry.snippets)} snippets*\n\n"
    
    # Group by language
    by_language = {}
    for snippet in entry.snippets:
        if snippet.language not in by_language:
            by_language[snippet.language] = []
        by_language[snippet.language].append(snippet)
    
    for language, snippets in by_language.items():
        response += f"## {language.title()} Examples\n\n"
        
        for snippet in snippets:
            response += f"### {snippet.title}\n"
            response += f"*Tags: {', '.join(snippet.tags)}*\n\n"
            
            if snippet.description:
                response += f"{snippet.description}\n\n"
            
            response += f"```{language}\n{snippet.code[:500]}\n```\n\n"
    
    return [types.TextContent(type="text", text=response)]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="code-translator",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())