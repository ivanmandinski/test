"""
WordPress content fetcher and processor.
"""
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging
from config import settings

logger = logging.getLogger(__name__)


class WordPressContentFetcher:
    """Fetches and processes content from WordPress REST API."""
    
    def __init__(self):
        self.base_url = settings.wordpress_api_url
        # Only use auth if credentials are provided
        auth = None
        if settings.wordpress_username and settings.wordpress_password and \
           settings.wordpress_username != "your_wp_username" and \
           settings.wordpress_password != "your_wp_app_password":
            auth = (settings.wordpress_username, settings.wordpress_password)
        
        self.client = httpx.AsyncClient(
            auth=auth,
            timeout=30.0,
            headers={"User-Agent": "HybridSearchBot/1.0"}
        )
    
    async def fetch_all_posts(self) -> List[Dict[str, Any]]:
        """Fetch all published posts from WordPress."""
        posts = []
        page = 1
        per_page = 100
        
        while True:
            try:
                response = await self.client.get(
                    f"{self.base_url}/posts",
                    params={
                        "per_page": per_page,
                        "page": page,
                        "status": "publish",
                        "_embed": True
                    }
                )
                response.raise_for_status()
                
                batch_posts = response.json()
                if not batch_posts:
                    break
                    
                posts.extend(batch_posts)
                page += 1
                
                logger.info(f"Fetched {len(batch_posts)} posts from page {page-1}")
                
            except httpx.HTTPError as e:
                logger.error(f"Error fetching posts: {e}")
                break
        
        logger.info(f"Total posts fetched: {len(posts)}")
        return posts
    
    async def fetch_all_pages(self) -> List[Dict[str, Any]]:
        """Fetch all published pages from WordPress."""
        pages = []
        page = 1
        per_page = 100
        
        while True:
            try:
                response = await self.client.get(
                    f"{self.base_url}/pages",
                    params={
                        "per_page": per_page,
                        "page": page,
                        "status": "publish",
                        "_embed": True
                    }
                )
                response.raise_for_status()
                
                batch_pages = response.json()
                if not batch_pages:
                    break
                    
                pages.extend(batch_pages)
                page += 1
                
                logger.info(f"Fetched {len(batch_pages)} pages from page {page-1}")
                
            except httpx.HTTPError as e:
                logger.error(f"Error fetching pages: {e}")
                break
        
        logger.info(f"Total pages fetched: {len(pages)}")
        return pages
    
    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length to prevent issues
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML content: {e}")
            # Return a safe fallback
            return "Content processing error"
    
    def process_content_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single WordPress content item."""
        try:
            # Extract basic information with safe defaults
            processed = {
                "id": str(item.get("id", "")),
                "title": self._safe_get_text(item.get("title", {}), "rendered", ""),
                "slug": str(item.get("slug", "")),
                "type": str(item.get("type", "post")),
                "url": str(item.get("link", "")),
                "date": str(item.get("date", "")),
                "modified": str(item.get("modified", "")),
                "author": self._safe_get_author(item),
                "categories": [],
                "tags": [],
                "excerpt": "",
                "content": "",
                "word_count": 0
            }
            
            # Clean and extract content
            raw_content = self._safe_get_text(item.get("content", {}), "rendered", "")
            processed["content"] = self.clean_html_content(raw_content)
            processed["word_count"] = len(processed["content"].split())
            
            # Extract excerpt
            excerpt_raw = self._safe_get_text(item.get("excerpt", {}), "rendered", "")
            if excerpt_raw:
                processed["excerpt"] = self.clean_html_content(excerpt_raw)
            
            # Extract categories and tags safely
            try:
                if "_embedded" in item and "wp:term" in item["_embedded"]:
                    for term_group in item["_embedded"]["wp:term"]:
                        for term in term_group:
                            if term.get("taxonomy") == "category":
                                processed["categories"].append({
                                    "id": str(term.get("id", "")),
                                    "name": str(term.get("name", "")),
                                    "slug": str(term.get("slug", ""))
                                })
                            elif term.get("taxonomy") == "post_tag":
                                processed["tags"].append({
                                    "id": str(term.get("id", "")),
                                    "name": str(term.get("name", "")),
                                    "slug": str(term.get("slug", ""))
                                })
            except Exception as e:
                logger.error(f"Error processing categories/tags: {e}")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing content item: {e}")
            # Return minimal safe structure
            return {
                "id": str(item.get("id", "unknown")),
                "title": "Processing Error",
                "slug": "",
                "type": "post",
                "url": "",
                "date": "",
                "modified": "",
                "author": "Unknown",
                "categories": [],
                "tags": [],
                "excerpt": "",
                "content": "Content processing error",
                "word_count": 0
            }
    
    def _safe_get_text(self, obj: Dict, key: str, default: str = "") -> str:
        """Safely get text from nested dictionary."""
        try:
            value = obj.get(key, default)
            if isinstance(value, str):
                # Remove any problematic characters
                return value.replace('\x00', '').replace('\r', '').replace('\n', ' ')[:5000]
            return str(value)[:5000] if value else default
        except:
            return default
    
    def _safe_get_author(self, item: Dict) -> str:
        """Safely get author name."""
        try:
            embedded = item.get("_embedded", {})
            authors = embedded.get("author", [])
            if authors and len(authors) > 0:
                return str(authors[0].get("name", "Unknown"))
            return "Unknown"
        except:
            return "Unknown"
    
    async def get_all_content(self) -> List[Dict[str, Any]]:
        """Fetch and process all WordPress content."""
        logger.info("Starting content fetch from WordPress...")
        
        try:
            # Fetch posts and pages concurrently
            posts_task = self.fetch_all_posts()
            pages_task = self.fetch_all_pages()
            
            posts, pages = await asyncio.gather(posts_task, pages_task)
            
            # Process all content with error handling
            all_content = []
            
            # Process posts
            for i, post in enumerate(posts):
                try:
                    processed_post = self.process_content_item(post)
                    all_content.append(processed_post)
                    if i % 100 == 0:
                        logger.info(f"Processed {i} posts...")
                except Exception as e:
                    logger.error(f"Error processing post {i}: {e}")
                    continue
            
            # Process pages
            for i, page in enumerate(pages):
                try:
                    processed_page = self.process_content_item(page)
                    all_content.append(processed_page)
                except Exception as e:
                    logger.error(f"Error processing page {i}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(all_content)} content items")
            return all_content
            
        except Exception as e:
            logger.error(f"Error in get_all_content: {e}")
            # Return empty list to prevent complete failure
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
