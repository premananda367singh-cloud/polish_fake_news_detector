from typing import Dict, Any, List
import re
from datetime import datetime
import json

class CredibilityScorer:
    """Score credibility of news sources and content"""
    
    def __init__(self, credibility_db_path: str = None):
        """
        Args:
            credibility_db_path: Path to credibility database JSON file
        """
        # Known credible and non-credible sources
        self.trusted_domains = [
            'gov.pl', 'edu.pl', 'uw.edu.pl', 'uj.edu.pl',
            'tvp.info', 'tvn24.pl', 'polsatnews.pl',  # Major news portals
            'wyborcza.pl', 'rp.pl', 'dziennik.pl'     # Newspapers
        ]
        
        self.untrusted_domains = [
            'fakenews.pl', 'niezalezna.pl',  # Example
        ]
        
        # Load external credibility database if provided
        self.external_db = {}
        if credibility_db_path:
            try:
                with open(credibility_db_path, 'r', encoding='utf-8') as f:
                    self.external_db = json.load(f)
            except:
                print("Could not load credibility database")
        
        # Keywords indicating low credibility
        self.sensational_keywords = [
            'szok', 'sensacja', 'niewiarygodne', 'obalono', 'ukrywają',
            'spisek', 'tajemnica', 'zatajono', 'eksplozja', 'katastrofa'
        ]
        
        # Author credibility indicators
        self.credible_authors = []
        self.non_credible_authors = []
    
    def calculate_score(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate credibility score based on metadata
        
        Args:
            metadata: Dictionary containing:
                - source: News source/domain
                - author: Author name
                - date: Publication date
                - url: Article URL
                - references: List of references
                - has_images: Whether article has images
                - quotes_experts: Whether quotes experts
        
        Returns:
            Dictionary with credibility score and factors
        """
        factors = {}
        total_score = 0.0
        max_possible = 0.0
        
        # 1. Source credibility (0-30 points)
        source_score, source_factors = self._score_source(metadata.get('source', ''))
        factors['source'] = source_factors
        total_score += source_score
        max_possible += 30
        
        # 2. Author credibility (0-20 points)
        author_score, author_factors = self._score_author(metadata.get('author', ''))
        factors['author'] = author_factors
        total_score += author_score
        max_possible += 20
        
        # 3. Content indicators (0-25 points)
        content_score, content_factors = self._score_content_indicators(metadata)
        factors['content'] = content_factors
        total_score += content_score
        max_possible += 25
        
        # 4. References and citations (0-15 points)
        ref_score, ref_factors = self._score_references(metadata.get('references', []))
        factors['references'] = ref_factors
        total_score += ref_score
        max_possible += 15
        
        # 5. Timeliness (0-10 points)
        time_score, time_factors = self._score_timeliness(metadata.get('date'))
        factors['timeliness'] = time_factors
        total_score += time_score
        max_possible += 10
        
        # Calculate final normalized score (0-1)
        final_score = total_score / max_possible if max_possible > 0 else 0.5
        
        # Determine credibility level
        if final_score >= 0.7:
            level = 'high'
        elif final_score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'score': final_score,
            'raw_score': total_score,
            'max_possible': max_possible,
            'level': level,
            'factors': factors,
            'metadata_used': {k: v for k, v in metadata.items() if k in ['source', 'author', 'date']}
        }
    
    def _score_source(self, source: str) -> tuple:
        """Score based on news source"""
        score = 15  # Default neutral score
        factors = {
            'domain_recognized': False,
            'trusted_domain': False,
            'untrusted_domain': False
        }
        
        if not source:
            return 10, {'error': 'No source provided'}
        
        # Extract domain
        domain = source.lower()
        if 'http' in domain:
            import urllib.parse
            try:
                domain = urllib.parse.urlparse(source).netloc
            except:
                pass
        
        # Check against known lists
        factors['domain_recognized'] = True
        
        for trusted in self.trusted_domains:
            if trusted in domain:
                score = 28
                factors['trusted_domain'] = True
                break
        
        for untrusted in self.untrusted_domains:
            if untrusted in domain:
                score = 2
                factors['untrusted_domain'] = True
                break
        
        # Check external database
        if domain in self.external_db:
            db_score = self.external_db[domain].get('score', 15)
            score = max(score, db_score)
            factors['external_db_match'] = True
        
        return score, factors
    
    def _score_author(self, author: str) -> tuple:
        """Score based on author information"""
        score = 10  # Default neutral
        
        if not author or author.strip() == '':
            return 5, {'has_author': False}
        
        factors = {
            'has_author': True,
            'author_verified': False,
            'author_known': False
        }
        
        # Check if author is in credible list
        if author in self.credible_authors:
            score = 20
            factors['author_verified'] = True
            factors['author_known'] = True
        
        # Check if author is in non-credible list
        if author in self.non_credible_authors:
            score = 0
            factors['author_verified'] = True
            factors['author_known'] = True
        
        # Simple heuristics
        if 'anonim' in author.lower() or 'anonymous' in author.lower():
            score = 3
            factors['anonymous'] = True
        
        return score, factors
    
    def _score_content_indicators(self, metadata: Dict[str, Any]) -> tuple:
        """Score based on content quality indicators"""
        score = 12  # Default neutral
        factors = {
            'has_images': metadata.get('has_images', False),
            'quotes_experts': metadata.get('quotes_experts', False),
            'sensational_language': False,
            'multiple_perspectives': metadata.get('multiple_perspectives', False)
        }
        
        # Check for images
        if factors['has_images']:
            score += 3
        
        # Check for expert quotes
        if factors['quotes_experts']:
            score += 5
        
        # Check for multiple perspectives
        if factors['multiple_perspectives']:
            score += 3
        
        # Check text for sensational language
        text = metadata.get('text', '')
        if text:
            sensational_count = sum(1 for keyword in self.sensational_keywords 
                                 if keyword in text.lower())
            if sensational_count > 2:
                score -= min(sensational_count * 2, 10)
                factors['sensational_language'] = True
        
        # Ensure score is within bounds
        score = max(0, min(25, score))
        
        return score, factors
    
    def _score_references(self, references: List[str]) -> tuple:
        """Score based on references and citations"""
        if not references:
            return 2, {'has_references': False, 'reference_count': 0}
        
        reference_count = len(references)
        
        # Score based on number and quality of references
        score = min(15, reference_count * 3)
        
        # Check if references include credible sources
        credible_refs = 0
        for ref in references:
            for trusted in self.trusted_domains:
                if trusted in ref.lower():
                    credible_refs += 1
                    break
        
        if credible_refs > 0:
            score += min(credible_refs * 2, 5)
        
        factors = {
            'has_references': True,
            'reference_count': reference_count,
            'credible_references': credible_refs
        }
        
        return score, factors
    
    def _score_timeliness(self, date_str: str) -> tuple:
        """Score based on timeliness of information"""
        score = 5  # Default neutral
        
        if not date_str:
            return 3, {'has_date': False}
        
        try:
            # Parse date
            if isinstance(date_str, str):
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d']:
                    try:
                        article_date = datetime.strptime(date_str[:10], fmt)
                        break
                    except:
                        continue
                else:
                    return 3, {'has_date': True, 'date_valid': False}
            else:
                article_date = date_str
            
            # Calculate age in days
            today = datetime.now()
            age_days = (today - article_date).days
            
            if age_days < 7:
                score = 10  # Very recent
            elif age_days < 30:
                score = 8   # Recent
            elif age_days < 365:
                score = 6   # Within year
            else:
                score = 2   # Outdated
            
            factors = {
                'has_date': True,
                'date_valid': True,
                'age_days': age_days,
                'is_recent': age_days < 30
            }
            
        except Exception as e:
            factors = {
                'has_date': True,
                'date_valid': False,
                'error': str(e)
            }
            score = 3
        
        return score, factors
    
    def add_trusted_source(self, domain: str, score: int = 30):
        """Add a trusted source to the database"""
        self.trusted_domains.append(domain.lower())
    
    def add_untrusted_source(self, domain: str):
        """Add an untrusted source to the database"""
        self.untrusted_domains.append(domain.lower())
    
    def export_credibility_db(self, path: str):
        """Export current credibility database to JSON"""
        db = {
            'trusted_domains': self.trusted_domains,
            'untrusted_domains': self.untrusted_domains,
            'credible_authors': self.credible_authors,
            'non_credible_authors': self.non_credible_authors
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)